#pragma once

// Graph scanning, cascade forcing, and persistent capture for MoE Phase 2.
// namespace moe

#ifdef GGML_USE_CUDA

#include <utility>

#include "ggml-backend.h"

struct ggml_context;
struct phase2_hijack;
struct ggml_cgraph;
struct ggml_tensor;

namespace moe {

std::pair<int,int> match_hijack_name(phase2_hijack & h2, const char * name);
int  scan_and_hijack(phase2_hijack & h2, ggml_cgraph * gf);
bool scan_and_update_snapshots(phase2_hijack & h2, ggml_cgraph * gf);
void restore_all(phase2_hijack & h2);
void copy_data_to_static(phase2_hijack & h2, void * cuda_stream);
void cascade_force_moe_consumers(phase2_hijack & h2, ggml_cgraph * gf, ggml_backend_sched_t sched, ggml_backend_t gpu_backend);

// Persistent cache for the Phase 2 MoE compute graph.
// Uses ggml_new_graph_custom() so the cgraph's internal hash set and use-counts
// are valid, while the actual tensor objects are shared with the source graph.
struct phase2_graph_cache {
    bool valid = false;
    struct ggml_context * ctx = nullptr;
    ggml_cgraph * persistent_gf = nullptr;

    void release();

    // Shallow capture shares the same tensor objects, so remapping is a no-op.
    ggml_tensor * remap(ggml_tensor * orig) const { return orig; }
};

// Captures a shallow copy of src_gf: only the graph structure is duplicated.
// Tensor objects are shared, preserving their backend-specific state.
void capture_phase2_graph(phase2_graph_cache & cache, ggml_cgraph * src_gf);

} // namespace moe

#endif
