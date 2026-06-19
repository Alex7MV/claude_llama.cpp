#pragma once

// Graph scanning, cascade forcing, and deep-copy caching for MoE Phase 2.
// Operates on phase2_hijack structs via reference parameters.
// All functions are in namespace moe.

#ifdef GGML_USE_CUDA

#include <utility>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"

struct phase2_hijack;
struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace moe {

std::pair<int,int> match_hijack_name(phase2_hijack & h2, const char * name);

int  scan_and_hijack(phase2_hijack & h2, ggml_cgraph * gf);

bool scan_and_update_snapshots(phase2_hijack & h2, ggml_cgraph * gf);

void restore_all(phase2_hijack & h2);

void copy_data_to_static(phase2_hijack & h2, void * cuda_stream);

void cascade_force_moe_consumers(
    phase2_hijack & h2,
    ggml_cgraph * gf,
    ggml_backend_sched_t sched,
    ggml_backend_t gpu_backend);

struct phase2_graph_cache {
    bool valid = false;
    ggml_cgraph * persistent_gf = nullptr;
    struct ggml_context * persistent_ctx = nullptr;
    std::vector<ggml_tensor *> persistent_tensors;

    void capture() { valid = true; }
    void release();
};

void deep_copy_phase2_graph(
    phase2_graph_cache & cache,
    ggml_cgraph * src_gf);

} // namespace moe

#endif // GGML_USE_CUDA