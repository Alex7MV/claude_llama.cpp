#include "hybrid_vram_pool.h"
#include <cstdio>

static constexpr uint32_t FP8_BYTES = 1;

hybrid_vram_pool::~hybrid_vram_pool() { free_all(); }

bool hybrid_vram_pool::init(
    uint32_t n_layers,
    uint32_t n_ctx_max,
    uint32_t kv_lora_rank,
    uint32_t max_lookahead,
    ggml_backend_t backend)
{
    m_n_layers     = n_layers;
    m_n_ctx_max    = n_ctx_max;
    m_kv_lora_rank = kv_lora_rank;
    m_backend      = backend;

    uint32_t n_slots = n_ctx_max + 2 * max_lookahead;
    m_stride = static_cast<uint64_t>(n_slots) * kv_lora_rank * FP8_BYTES;
    m_total_bytes = static_cast<uint64_t>(m_n_layers) * m_stride;

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        fprintf(stderr, "hybrid_vram_pool: backend has no device\n");
        return false;
    }

    m_dev_buffer = ggml_backend_buft_alloc_buffer(
        ggml_backend_dev_buffer_type(dev), m_total_bytes);
    if (!m_dev_buffer) {
        fprintf(stderr, "hybrid_vram_pool: ggml_backend_buft_alloc_buffer(%llu) failed\n",
                (unsigned long long)m_total_bytes);
        return false;
    }

    m_device_ptr = reinterpret_cast<float *>(ggml_backend_buffer_get_base(m_dev_buffer));
    if (!m_device_ptr) {
        ggml_backend_buffer_free(m_dev_buffer);
        m_dev_buffer = nullptr;
        fprintf(stderr, "hybrid_vram_pool: buffer has nil base\n");
        return false;
    }

    ggml_backend_buffer_clear(m_dev_buffer, 0);

    fprintf(stderr, "hybrid_vram_pool: allocated %.2f GB VRAM (%u layers, %u ctx, rank=%u)\n",
            m_total_bytes / (double)(1ull << 30), n_layers, n_ctx_max, kv_lora_rank);
    return true;
}

void hybrid_vram_pool::free_all() {
    if (m_dev_buffer) {
        ggml_backend_buffer_free(m_dev_buffer);
        m_dev_buffer = nullptr;
    }
    m_device_ptr = nullptr;
    m_total_bytes = 0;
}

float * hybrid_vram_pool::slot_ptr(uint32_t layer, uint32_t seq_pos) const {
    return reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(m_device_ptr)
        + static_cast<uint64_t>(layer) * m_stride
        + static_cast<uint64_t>(seq_pos) * m_kv_lora_rank * FP8_BYTES);
}
