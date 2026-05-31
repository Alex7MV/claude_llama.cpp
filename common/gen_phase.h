#pragma once

#include "llama.h"

#include <string>
#include <unordered_map>

enum class llama_gen_phase {
    TEXT,             // Free text generation, grammar inactive, <|call|> masked
    REASONING,        // Inside <|thought|> block, grammar inactive, <|call|> masked
    TOOL_INVOCATION   // After <|call|> token, grammar active with JSON constraint
};

struct llama_gen_phase_tokens {
    llama_token id_thought    = LLAMA_TOKEN_NULL;  // <|thought|>
    llama_token id_call       = LLAMA_TOKEN_NULL;  // <|call|> or <tool_call>
    llama_token id_end        = LLAMA_TOKEN_NULL;  // EOS / <|im_end|>
    llama_token id_json_start = LLAMA_TOKEN_NULL;  // '{'
    llama_token id_json_end   = LLAMA_TOKEN_NULL;  // '}'
};

// Resolve trigger token IDs from model vocab at load time.
// Probes known token strings in priority order:
//   <|thought|>, <|call|> → <tool_call> → <|tool_call|>
// Returns a struct with LLAMA_TOKEN_NULL for any unfound markers.
// R6: LLAMA_TOKEN_NULL = pass-through mode (state machine is a no-op).
llama_gen_phase_tokens resolve_model_signatures(const struct llama_model * model);

// Create the gen_phase sampler instance.
// The sampler holds resolved token IDs, the GBNF grammar string for tool calls,
// and manages internal phase state and grammar lifecycle.
// R4: MUST be the FIRST sampler added to the chain.
struct llama_sampler * sampler_gen_phase_init(
    const struct llama_vocab *     vocab,
    const llama_gen_phase_tokens & tokens,
    const std::string &            tool_call_grammar_str);

// Reset phase to TEXT and free inner grammar.
// R3: Guaranteed cleanup — called on slot release/interruption.
void sampler_gen_phase_reset(struct llama_sampler * smpl);
