# Generation Phase State Machine for DeepSeek V4 Pro

**Date:** 2026-05-30
**Status:** Specified
**Model:** DeepSeek V4 Pro (AOCC-optimized fork, claude_llama.cpp)

## Objective

Implement a strict Generation State Machine that isolates Tool Calling from Text/Reasoning, preventing prompt injection and accidental tool invocation. GBNF grammar is ONLY active during the `TOOL_INVOCATION` phase, with logit bias masking of trigger tokens in all other phases.

## Architecture

Three-layer design with phase ownership at the sampling layer:

```
server-context.cpp (server_slot)
  └─ stores JSON schema per request
  └─ converts schema → GBNF on demand
  └─ hard-resets phase on slot release
  └─ strips pre-filled tool-call grammar from params

common/sampling.cpp (common_sampler)
  └─ owns llama_gen_phase phase (TEXT/REASONING/TOOL_INVOCATION)
  └─ in apply(): masks <|call|> token if phase != TOOL_INVOCATION
  └─ in accept(): detects phase transitions via token IDs
  └─ dynamically creates/destroys grammar sampler on transition

DeepSeek V4 Pro Vocab
  └─ token IDs resolved at load time via vocab scan
```

## Components

### 1. `llama_gen_phase_tokens` — Resolved Token IDs

Defined in `common/sampling.h` (or a new `common/gen_phase.h`):

```cpp
struct llama_gen_phase_tokens {
    llama_token id_thought    = LLAMA_TOKEN_NULL;  // <|thought|>
    llama_token id_call       = LLAMA_TOKEN_NULL;  // <|call|> or <tool_call>
    llama_token id_end        = LLAMA_TOKEN_NULL;  // <|im_end|> or </s>
    llama_token id_json_start = LLAMA_TOKEN_NULL;  // '{'
};
```

Resolved once at model load time via `resolve_model_signatures()` which probes the vocabulary. Defaults to `LLAMA_TOKEN_NULL` (= no-op) if markers not found.

### 2. `llama_gen_phase` — Phase Enum

```cpp
enum class llama_gen_phase {
    TEXT,             // Free text generation, grammar inactive
    REASONING,        // Inside <|thought|>...</|thought|>, grammar inactive
    TOOL_INVOCATION   // After <|call|>, grammar active, JSON constrained
};
```

### 3. Custom `llama_sampler_i` — `sampler_gen_phase`

A new `llama_sampler_i` implementation added to the sampler chain at init time. Implements:

**`apply()`** (called before main chain):
- If `phase == TEXT` or `phase == REASONING`:
  - If `id_call != LLAMA_TOKEN_NULL`, set `candidates->data[id_call].logit = -FLT_MAX`
  - Return immediately (constant-time, single branch)
- If `phase == TOOL_INVOCATION`:
  - If inner grammar sampler exists: apply it first
  - Return

**`accept()`** (called after token is selected):
- If `phase == TEXT`:
  - `token == id_thought` → set phase to `REASONING`
  - `token == id_call` → set phase to `TOOL_INVOCATION`, create grammar
- If `phase == REASONING`:
  - `token == id_call` → set phase to `TOOL_INVOCATION`, create grammar
  - `token == id_end` → set phase to `TEXT`
- If `phase == TOOL_INVOCATION`:
  - **Primary exit:** Check `llama_grammar_is_finished(inner_grmr)`. If true → JSON is valid and complete → set phase to `TEXT`, destroy grammar
  - **Fallback exit:** JSON brace-depth tracking: increment counter on `{`, decrement on `}`. When depth reaches 0 after decrement and primary check is unavailable → JSON block complete → set phase to `TEXT`, destroy grammar
  - `token == id_end` → set phase to `TEXT` (last resort), destroy grammar

**Grammar creation** (on entering `TOOL_INVOCATION`):
- Convert stored `json_schema` string → GBNF grammar via `json_schema_to_grammar()`
- Call `llama_sampler_init_grammar()` to create grammar sampler
- Store as inner child sampler

**Grammar destruction** (on exiting `TOOL_INVOCATION`):
- Call grammar sampler's `reset()` then `free()`
- Set inner grammar pointer to `nullptr`

### 3a. Sampler Chain Position Guarantee

`sampler_gen_phase` MUST be the **first** sampler added to `llama_sampler_chain`. This guarantees the forbidden `<|call|>` token is masked with `-FLT_MAX` before any other sampler (temperature, top_k, min_p, etc.) can transform the probability distribution. If masking happened after temperature scaling, residual probability mass could leak through numerical imprecision.

Insertion order in `common_sampler_init()`:
```
chain.add(sampler_gen_phase)   // 1st: mask forbidden tokens
chain.add(logit_bias)          // 2nd: user logit biases
chain.add(grammar/dry/...)     // 3rd+: all other samplers
```

### 4. `resolve_model_signatures()` — Vocab Scan

Called once during model load in `server.cpp`. Probes the vocabulary in order:

1. Try specific marker tokens: `<|thought|>`, `<|call|>`
2. Fallback markers: `<tool_call>`, `<|tool_call|>`
3. Always resolve `{` for JSON start detection
4. Always resolve `id_end` from EOS token

Supports `--model-type [deepseek|kimi|auto]` CLI flag. Default `auto`.

### 5. Server Integration Changes

**`server-context.cpp` — `launch_slot_with_task()`:**
- Drop pre-filled tool-call grammar from `params.sampling.grammar` when tools are present
- Pass `json_schema` string through to `common_sampler_init()` via a new field

**`server_context_impl` — GBNF compilation cache:**
- Add `std::unordered_map<std::string, std::string> schema_to_gbnf_cache` to `server_context_impl`
- Before converting a JSON schema to GBNF, check the cache: `if (cache.count(json_schema)) { return cache[json_schema]; }`
- Cache is per-server-instance, persists across requests
- Motivation: MCP agents repeatedly call the same tools (e.g. `read_file`, `google_search`), producing identical JSON schemas. Caching avoids re-compiling the same schema → GBNF conversion on every request.

**`server-context.cpp` — `update_slots()`:**
- No changes needed at this layer (all phase logic is in the sampler)

**`server-context.cpp` — slot release / reset:**
- Ensure `common_sampler_reset()` is called, which triggers `sampler_gen_phase.reset()` → phase = TEXT, grammar freed

### 6. CLI Integration (`--model-type` flag)

Added to `common_params` and `common_params_sampling`:
```
--model-type [deepseek|kimi|auto]     Model architecture for phase tokens
```

## Phase Transition Diagram

```
TEXT ───────────────────────────────────────────┐
  │ accept(id_thought) → REASONING              │
  │ accept(id_call) → TOOL_INVOCATION           │
  ▼                                             │
REASONING                                       │
  │ accept(id_call) → TOOL_INVOCATION           │
  │ accept(id_end) → TEXT                       │
  ▼                                             │
TOOL_INVOCATION ◄───────────────────────────────┘
  │ llama_grammar_is_finished() → TEXT (primary)
  │ brace_depth == 0 → TEXT (fallback)
  │ accept(id_end) → TEXT (last resort)
  │ slot release / n_predict → TEXT (hard reset)
```

## Grammar Lifecycle

- **Created:** Lazily, on first transition to `TOOL_INVOCATION`
- **Active:** Only during `TOOL_INVOCATION` phase
- **Destroyed:** On transition back to `TEXT` or `REASONING`, or on slot reset
- **Source:** JSON schema provided per-request, converted to GBNF at activation time

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Model stops mid-JSON (n_predict) | Partial JSON sent as content, phase reset to TEXT on next sample call. `brace_depth` is reset to 0. |
| Timeout / disconnect | `slot.release()` → `sampler.reset()` → phase = TEXT, grammar freed |
| Model emits `<\|call\|>` but no valid JSON | Grammar constrains output; resampling until EOS if model can't produce valid JSON |
| Markers not found in vocab | `id_call == LLAMA_TOKEN_NULL` → phase always TEXT, sampler is identity pass-through |
| Multiple `<\|call\|>` emitted (tool chaining) | Each tool call completes a JSON block → TEXT → next call triggers TOOL_INVOCATION again |

## Testing (Red Stage)

Four test cases in `tests/isolation_test.py` using `pytest` + `requests`:

1. **test_instruction_injection_rejected** — Ensure prompt injection does not trigger tool call grammar
2. **test_grammar_inactive_during_reasoning** — Verify grammar inactive inside `<|thought|>` blocks
3. **test_tool_token_masked_in_text** — Verify `<|call|>` logit is -inf in TEXT phase
4. **test_mid_json_reset_on_n_predict** — Verify phase resets cleanly after mid-JSON interruption

## AOCC Optimization Notes

- Phase comparison is a single `int32_t` register comparison (`token == id_call`)
- Logit masking is a single array write: `candidates->data[idx].logit = -FLT_MAX`
- Grammar activation is a single null-pointer check: `if (grmr) { ... }`
- No allocations in the hot path (grammar creation/destruction happens on phase transitions, not per-token)
- The `apply()` fast path for non-TOOL_INVOCATION phases is: 1 comparison + 1 branch + 1 array write = ~3 cycles
- `sampler_gen_phase` being **first in chain** means downstream samplers never perform entropy work on the masked `<|call|>` token — it is removed from the candidate pool before any distribution transform
- GBNF cache in server_context eliminates repeated `json_schema → GBNF` compilation for MCP-style repeated tool calls
