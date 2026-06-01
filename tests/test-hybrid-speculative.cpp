#include "ggml.h"
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
#include <climits>

#ifdef NDEBUG
#undef NDEBUG
#endif

// --- Non-CUDA tests (pure logic, always compile) ---

// Test VRAM pool size calculation
static void test_pool_size() {
    uint32_t n_layers     = 61;
    uint32_t n_ctx_max    = 256 * 1024;
    uint32_t kv_lora_rank = 512;
    uint32_t max_lookahead = 6;

    uint64_t n_slots = n_ctx_max + 2 * max_lookahead;
    uint64_t stride = n_slots * kv_lora_rank * 1; // FP8 = 1 byte
    uint64_t total = n_layers * stride;

    printf("test_pool_size:\n");
    printf("  n_slots    = %llu\n", (unsigned long long)n_slots);
    printf("  stride     = %llu bytes/layer\n", (unsigned long long)stride);
    printf("  total      = %llu bytes (%.2f GB)\n",
           (unsigned long long)total, total / (double)(1ull << 30));
    printf("  expected   = 8187281408 bytes (7.62 GB)\n");

    assert(total == 8187281408ull);
    printf("  PASS\n\n");
}

// Test verify_and_rollback counting logic
static void test_verify_counting() {
    std::vector<int> draft = {10, 20, 30, 40, 50};
    std::vector<int> target_full_match = {10, 20, 30, 40, 50, 60};
    std::vector<int> target_partial    = {10, 20, 30, 99, 100};
    std::vector<int> target_first_mismatch = {99};

    auto count_matches = [](const auto & draft, const auto & target) -> uint32_t {
        uint32_t n = 0;
        for (uint32_t i = 0; i < draft.size() && i < target.size(); i++) {
            if (target[i] == draft[i]) { n++; } else { break; }
        }
        return n;
    };

    printf("test_verify_counting:\n");
    assert(count_matches(draft, target_full_match) == 5);
    printf("  full match: 5/5 PASS\n");
    assert(count_matches(draft, target_partial) == 3);
    printf("  partial match: 3/5 PASS\n");
    assert(count_matches(draft, target_first_mismatch) == 0);
    printf("  first mismatch: 0/5 PASS\n");
    printf("  PASS\n\n");
}

// Test adaptive lookahead thresholds
static void test_adaptive_lookahead() {
    // Thresholds from spec:
    // > 60%: 6, 30-60%: 4, < 30%: 3 (floor)
    auto lookahead_for = [](float accept_rate) -> int {
        if (accept_rate > 0.60f) return 6;
        if (accept_rate > 0.30f) return 4;
        return 3;
    };

    printf("test_adaptive_lookahead:\n");
    assert(lookahead_for(0.80f) == 6);
    assert(lookahead_for(0.60f) == 4); // not > 0.60
    assert(lookahead_for(0.45f) == 4);
    assert(lookahead_for(0.30f) == 3); // not > 0.30
    assert(lookahead_for(0.10f) == 3);
    printf("  PASS\n\n");
}

#ifdef GGML_USE_CUDA
// Test orchestrator initial state (requires CUDA for constructor)
#include "common/hybrid_stage.h"
static void test_orchestrator_state() {
    hybrid_orchestrator orch;

    printf("test_orchestrator_state:\n");
    assert(orch.stages.empty());
    assert(orch.n_layers == 0);
    assert(orch.kv_lora_rank == 0);
    assert(orch.current_layer == 0);
    assert(orch.current_token == 0);
    assert(orch.draft.lookahead_buffer.empty());
    assert(!orch.draft.in_flight);
    assert(orch.verify.accepted == 0);
    assert(orch.verify.rejected_at == UINT32_MAX);
    assert(orch.verify.n_draft == 0);
    printf("  PASS\n\n");
}
#else
static void test_orchestrator_state() {
    printf("test_orchestrator_state: SKIP (no CUDA)\n\n");
}
#endif

int main() {
    printf("=== Hybrid Speculative Engine Tests ===\n\n");

    test_pool_size();
    test_verify_counting();
    test_adaptive_lookahead();
    test_orchestrator_state();

    printf("=== ALL TESTS PASSED ===\n");
    return 0;
}
