#pragma once
#include <cstdint>
#include "ggml.h"
#include "ggml-backend.h"

struct hybrid_vram_pool {
    hybrid_vram_pool() = default;
    ~hybrid_vram_pool();

    hybrid_vram_pool(const hybrid_vram_pool &) = delete;
    hybrid_vram_pool & operator=(const hybrid_vram_pool &) = delete;

    bool init(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t kv_lora_rank,
        uint32_t max_lookahead,
        ggml_backend_dev_t device,
        ggml_type   cache_type_k = GGML_TYPE_F16,
        uint32_t    key_dim      = 0);

    void free_all();

    float * slot_ptr(uint32_t layer, uint32_t seq_pos) const;
    uint64_t total_bytes() const { return m_total_bytes; }
    ggml_backend_buffer_t backend_buffer() const { return m_dev_buffer; }

private:
    float *  m_device_ptr    = nullptr;
    uint64_t m_total_bytes   = 0;
    uint32_t m_n_layers      = 0;
    uint32_t m_n_ctx_max     = 0;
    uint32_t m_kv_lora_rank  = 0;
    uint32_t m_per_slot_bytes = 0;
    uint32_t m_stride         = 0;

    ggml_backend_buffer_t m_dev_buffer = nullptr;
    ggml_backend_dev_t    m_device     = nullptr;
};
