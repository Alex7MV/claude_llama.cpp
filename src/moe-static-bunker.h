#pragma once

// Static memory hijacking infrastructure for MoE Phase 2.
// Contains inert data structures (POD) — no graph-walking logic.
// Graph scanning / cascade forcing lives in moe-hijacker.h.

#ifdef GGML_USE_CUDA

#include <cstddef>

#define H2_ALIGN         256
#define H2_ALIGN_UP(x)   (((x) + (H2_ALIGN - 1)) & ~(H2_ALIGN - 1))
#define H2_N_LAYERS_MAX  128

struct phase2_hijack {
    static constexpr const char * TENSOR_NAMES[7] = {
        "ffn_moe_gate_up",
        "ffn_moe_gate",
        "ffn_moe_up",
        "ffn_moe_swiglu",
        "ffn_moe_down",
        "ffn_moe_weighted",
        "ffn_moe_out",
    };
    enum TypeId : int {
        T_GATE_UP   = 0,
        T_GATE      = 1,
        T_UP        = 2,
        T_SWIGLU    = 3,
        T_DOWN      = 4,
        T_WEIGHTED  = 5,
        T_OUT       = 6,
        N_TYPES     = 7,
    };

    struct slot {
        void * addr;
        size_t size;
        int    parent;
        ptrdiff_t offset;
        void * orig_data;
    };

    void * buffer = nullptr;
    size_t buffer_size = 0;
    int    n_layers = 0;

    slot * slots = nullptr;
    int    n_slots = 0;
    int    slot_scan_count = 0;

    bool   captured = false;
    void * cuda_graph_exec = nullptr;
    void * stream = nullptr;

    void init(int n_moe_layers);
    void destroy();
    void allocate_slots(int max_il);

    int slot_idx(int il, int type_id) const { return il * N_TYPES + type_id; }
};

struct phase2_inject {
    void * host_moe_ids[H2_N_LAYERS_MAX];
    float * host_moe_w[H2_N_LAYERS_MAX];

    void init();
    void destroy();
    void fill_layer(int il, const int * ids, const float * w);
    void inject_all(void * stream);
};

struct phase2_guard {
    void * phase1_done_event = nullptr;

    void init();
    void destroy();
    void record(void * stream);
    void wait(void * stream);
};

#endif // GGML_USE_CUDA