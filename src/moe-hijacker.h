#pragma once

// Graph scanning and cascade forcing for MoE Phase 2.
// Operates on phase2_hijack structs via reference parameters.
// All functions are in namespace moe.

#ifdef GGML_USE_CUDA

#include <utility>

#include "ggml-backend.h"

struct phase2_hijack;
struct ggml_cgraph;

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

    void capture() { valid = true; }
    void release() { valid = false; }
};

} // namespace moe

#endif // GGML_USE_CUDA