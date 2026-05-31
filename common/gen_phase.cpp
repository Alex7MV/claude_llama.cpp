#include "gen_phase.h"
#include "common.h"

#include <cmath>
#include <limits>

// ---------------------------------------------------------------
// R6: Vocab discovery — resolve trigger token IDs from model vocab
// ---------------------------------------------------------------

llama_gen_phase_tokens resolve_model_signatures(const struct llama_model * model) {
    const auto * vocab = llama_model_get_vocab(model);
    llama_gen_phase_tokens tokens;

    // Tokenize with parse_special=true to match special tokens as single token IDs
    auto tokenize_one = [&](const std::string & text) -> llama_token {
        auto t = common_tokenize(vocab, text, false, true);
        return t.size() == 1 ? t[0] : LLAMA_TOKEN_NULL;
    };

    // Primary: DeepSeek-style markers
    tokens.id_thought = tokenize_one("<|thought|>");
    tokens.id_call    = tokenize_one("<|call|>");

    // R6: Fallback to other known marker formats
    if (tokens.id_call == LLAMA_TOKEN_NULL) {
        tokens.id_call = tokenize_one("<tool_call>");
    }
    if (tokens.id_call == LLAMA_TOKEN_NULL) {
        tokens.id_call = tokenize_one("<|tool_call|>");
    }

    // Always resolve structural JSON tokens (needed for brace-depth fallback)
    tokens.id_json_start = tokenize_one("{");
    tokens.id_json_end   = tokenize_one("}");

    // R6: EOS token — used as last-resort exit from TOOL_INVOCATION
    tokens.id_end = llama_vocab_eos(vocab);

    return tokens;
}

// ---------------------------------------------------------------
// Sampler context — holds all state for the gen_phase state machine
// ---------------------------------------------------------------

struct gen_phase_sampler_context {
    llama_gen_phase          phase = llama_gen_phase::TEXT;
    llama_gen_phase_tokens   tokens;
    std::string              tool_call_grammar_str;
    const struct llama_vocab * vocab;

    // R3: Grammar sampler created lazily on entering TOOL_INVOCATION,
    //     destroyed on exit or reset.
    struct llama_sampler *   grmr = nullptr;

    // R5: Brace depth tracker (fallback exit from TOOL_INVOCATION)
    int                      brace_depth = 0;
};

// ---------------------------------------------------------------
// AOCC-compatible inline attribute for hot-path methods
// ---------------------------------------------------------------

#if defined(__GNUC__) || defined(__clang__)
#define GEN_PHASE_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define GEN_PHASE_ALWAYS_INLINE __forceinline
#else
#define GEN_PHASE_ALWAYS_INLINE inline
#endif

// ---------------------------------------------------------------
// R5: Grammar completion probe — tests if grammar is in accepting state
// ---------------------------------------------------------------

static GEN_PHASE_ALWAYS_INLINE bool grammar_in_accepting_state(
    struct llama_sampler * grmr,
    llama_token             eos_id)
{
    if (!grmr) {
        return false;
    }
    // R1: Stack-only allocation — no heap in hot path
    llama_token_data single = {eos_id, 1.0f, 0.0f};
    llama_token_data_array single_arr = {&single, 1, -1, false};
    llama_sampler_apply(grmr, &single_arr);
    return single_arr.data[0].logit != -INFINITY;
}

// ---------------------------------------------------------------
// Sampler interface implementation
// ---------------------------------------------------------------

static const char * gen_phase_name(const struct llama_sampler * /*smpl*/) {
    return "gen-phase";
}

// R1: Zero allocations in apply(). Only integer comparison + logit write.
// R4: Runs before any distribution-transform sampler in the chain.
static GEN_PHASE_ALWAYS_INLINE void gen_phase_apply(
    struct llama_sampler *        smpl,
    llama_token_data_array *      cur_p)
{
    auto * ctx = (gen_phase_sampler_context *)smpl->ctx;

    if (ctx->phase == llama_gen_phase::TOOL_INVOCATION) {
        if (ctx->grmr) {
            llama_sampler_apply(ctx->grmr, cur_p);
        }
        return;
    }

    // TEXT or REASONING: mask <|call|> token
    if (ctx->tokens.id_call != LLAMA_TOKEN_NULL) {
        for (size_t i = 0; i < cur_p->size; i++) {
            if (cur_p->data[i].id == ctx->tokens.id_call) {
                cur_p->data[i].logit = -std::numeric_limits<float>::max();
                break;
            }
        }
    }
}

// R1: Zero allocations in accept(). Integer comparisons only.
// R3: Guarantees grmr is freed on every exit from TOOL_INVOCATION.
// R5: Grammar-is-finished is primary exit; brace-depth is fallback.
static GEN_PHASE_ALWAYS_INLINE void gen_phase_accept(
    struct llama_sampler *        smpl,
    llama_token                   token)
{
    auto * ctx = (gen_phase_sampler_context *)smpl->ctx;

    switch (ctx->phase) {
        case llama_gen_phase::TEXT: {
            if (token == ctx->tokens.id_thought) {
                ctx->phase = llama_gen_phase::REASONING;
            } else if (token == ctx->tokens.id_call) {
                ctx->phase = llama_gen_phase::TOOL_INVOCATION;
                // Lazy grammar creation on first call token
                if (!ctx->grmr && !ctx->tool_call_grammar_str.empty()) {
                    ctx->grmr = llama_sampler_init_grammar(
                        ctx->vocab,
                        ctx->tool_call_grammar_str.c_str(),
                        "root");
                }
                ctx->brace_depth = 0;
            }
            break;
        }
        case llama_gen_phase::REASONING: {
            if (token == ctx->tokens.id_call) {
                ctx->phase = llama_gen_phase::TOOL_INVOCATION;
                if (!ctx->grmr && !ctx->tool_call_grammar_str.empty()) {
                    ctx->grmr = llama_sampler_init_grammar(
                        ctx->vocab,
                        ctx->tool_call_grammar_str.c_str(),
                        "root");
                }
                ctx->brace_depth = 0;
            } else if (token == ctx->tokens.id_end) {
                ctx->phase = llama_gen_phase::TEXT;
            }
            break;
        }
        case llama_gen_phase::TOOL_INVOCATION: {
            // Track brace depth for fallback exit detection
            if (token == ctx->tokens.id_json_start) {
                ctx->brace_depth++;
            } else if (token == ctx->tokens.id_json_end) {
                ctx->brace_depth--;
            }

            // Accept token into inner grammar if active
            if (ctx->grmr) {
                llama_sampler_accept(ctx->grmr, token);
            }

            // R5: Ordered exit checks
            bool should_exit = false;

            // PRIMARY: grammar says JSON is complete (EOS is valid next token)
            if (ctx->grmr && grammar_in_accepting_state(ctx->grmr, ctx->tokens.id_end)) {
                should_exit = true;
            }
            // FALLBACK: brace depth reached 0 (simple JSON completion)
            if (!should_exit && ctx->brace_depth == 0 && token == ctx->tokens.id_json_end) {
                should_exit = true;
            }
            // LAST RESORT: end-of-generation token
            if (token == ctx->tokens.id_end) {
                should_exit = true;
            }

            if (should_exit) {
                ctx->phase = llama_gen_phase::TEXT;
                // R3: explicit free + nullify to prevent leaks
                if (ctx->grmr) {
                    llama_sampler_free(ctx->grmr);
                    ctx->grmr = nullptr;
                }
                ctx->brace_depth = 0;
            }
            break;
        }
    }
}

// R3: Guaranteed reset — phase to TEXT, grammar freed, brace depth cleared.
// Called on slot release, interruption, or n_predict limit.
static void gen_phase_reset(struct llama_sampler * smpl) {
    auto * ctx = (gen_phase_sampler_context *)smpl->ctx;
    ctx->phase = llama_gen_phase::TEXT;
    ctx->brace_depth = 0;
    if (ctx->grmr) {
        llama_sampler_free(ctx->grmr);
        ctx->grmr = nullptr;
    }
}

// R3: Full cleanup — frees inner grammar, then context, then sampler.
static void gen_phase_free(struct llama_sampler * smpl) {
    auto * ctx = (gen_phase_sampler_context *)smpl->ctx;
    if (ctx->grmr) {
        llama_sampler_free(ctx->grmr);
        ctx->grmr = nullptr;
    }
    delete ctx;
    delete smpl;
}

// ---------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------

struct llama_sampler * sampler_gen_phase_init(
    const struct llama_vocab *     vocab,
    const llama_gen_phase_tokens & tokens,
    const std::string &            tool_call_grammar_str)
{
    auto * ctx = new gen_phase_sampler_context();
    ctx->vocab                = vocab;
    ctx->tokens               = tokens;
    ctx->tool_call_grammar_str = tool_call_grammar_str;
    ctx->phase                = llama_gen_phase::TEXT;

    // R6: If no markers found, state machine is a no-op pass-through.
    // The apply() check `if (id_call != LLAMA_TOKEN_NULL)` will be false,
    // so no masking occurs and no phase transitions happen.

    auto * s = new llama_sampler {
        new llama_sampler_i {
            /* .name             = */ gen_phase_name,
            /* .accept           = */ gen_phase_accept,
            /* .apply            = */ gen_phase_apply,
            /* .reset            = */ gen_phase_reset,
            /* .clone            = */ nullptr,
            /* .free             = */ gen_phase_free,
            /* .backend_init     = */ nullptr,
            /* .backend_accept   = */ nullptr,
            /* .backend_apply    = */ nullptr,
            /* .backend_set_input = */ nullptr,
        },
        /* .ctx = */ ctx,
    };
    return s;
}

void sampler_gen_phase_reset(struct llama_sampler * smpl) {
    if (smpl) {
        gen_phase_reset(smpl);
    }
}
