#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Blackwell (SM 100+) Pipeline Primitives
// ---------------------------------------------------------------------------
// TMA descriptors, mbarrier orchestration, and WGMMA early-start hooks.
// Compiled only when __CUDA_ARCH__ >= 1000.
//
// Reference: CUDA 13.x PTX ISA, SM 100/101/120/121 mbarrier & WGMMA.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared-memory mbarrier layout
// ---------------------------------------------------------------------------
// Each mbarrier is 8 bytes (uint64_t), 8-byte aligned.
// For WGMMA warp-group sync we typically need 1 mbarrier per tile-in-flight.

#define GGML_BLACKWELL_MBARRIER_ALIGN 8

// ---------------------------------------------------------------------------
// Host-side: tile geometry for 600GB Kimi-K2.6 model
// ---------------------------------------------------------------------------
// KV-cache prefill tiles: we decompose the giant KV tensor into chunks
// that fit in SMEM and keep WGMMA busy.
//
// For 256K+ context, head_dim typically 128 or 256 (BF16).
// A "page" is one token's K or V vector.
// A "tile" is a contiguous group of pages processed together.
//
// Tile dimensions (conservative for 128 KB SMEM budget per SM):
//   TILE_M = 64 tokens
//   TILE_N = 128 head_dim (BF16 = 256 bytes per token)
//   TILE_K = 64 (chunk of sequence length for attention reduction)
//
// KV tile bytes (K or V, not both):
//   64 tokens * 128 head_dim * 2 bytes (BF16) = 16,384 bytes
//
// Attention Q@K^T tile:
//   Q tile: 64 * 128 * 2 = 16,384 bytes
//   K tile: 64 * 128 * 2 = 16,384 bytes
//   Accumulator: 64 * 64 * 4 (FP32) = 16,384 bytes
//
// Total SMEM per tile ~ 48 KB (well under 128 KB).
// ---------------------------------------------------------------------------

struct ggml_blackwell_tile_shape {
    int32_t tile_m;   // tokens per tile
    int32_t tile_n;   // head_dim or chunk size
    int32_t tile_k;   // reduction dimension chunk
    int32_t type_size; // 2 for BF16, 1 for Q8_0, etc.
};

static inline constexpr ggml_blackwell_tile_shape ggml_blackwell_default_kv_tile() {
    return { 64, 128, 64, 2 }; // BF16
}

// ---------------------------------------------------------------------------
// Transaction count computation
// ---------------------------------------------------------------------------
// mbarrier tracks *bytes* expected via TMA bulk async copy.
// For a KV tile, the transaction count is the number of bytes the TMA
// engine will move into SMEM before the mbarrier unblocks WGMMA.
//
// In the "early start" model, WGMMA begins as soon as the *first* tile
// is resident, not the whole layer. We therefore set the mbarrier
// transaction count to ONE tile's worth of bytes, and the WGMMA loop
// iterates with its own per-tile mbarrier wait.
// ---------------------------------------------------------------------------

__host__ __device__ inline uint32_t ggml_blackwell_mbar_tx_bytes_kv(
    const ggml_blackwell_tile_shape & tile) {
    // One K or V tile: tile_m tokens * tile_n features * type_size
    uint64_t bytes = uint64_t(tile.tile_m) * tile.tile_n * tile.type_size;
    assert(bytes <= 0xFFFFFFFFu); // mbarrier tx count is 32-bit
    return static_cast<uint32_t>(bytes);
}

__host__ __device__ inline uint32_t ggml_blackwell_mbar_tx_bytes_qkt(
    const ggml_blackwell_tile_shape & tile) {
    // Q@K^T needs Q tile + K tile loaded before first WGMMA
    uint64_t bytes = 2ull * tile.tile_m * tile.tile_n * tile.type_size;
    assert(bytes <= 0xFFFFFFFFu);
    return static_cast<uint32_t>(bytes);
}

// ---------------------------------------------------------------------------
// Device-side mbarrier primitives (SM 100+)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Device-side mbarrier primitives (SM 100+)
// ---------------------------------------------------------------------------
// Guarded by GGML_CUDA_HAS_WGMMA: the CMake try_compile verifies that
// this CUDA toolkit's ptxas understands the mbarrier / cp.async.bulk
// PTX instructions. Without the guard, toolkits that target sm_100+
// but lack PTX support would fail to compile the inline asm.
// ---------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(GGML_CUDA_HAS_WGMMA)

// Initialize an mbarrier in shared memory.
// @param addr       Shared-memory uint64_t pointer (8-byte aligned)
// @param arrive_count  Number of expected arrivals (usually 1 for TMA+WGMMA)
__device__ __forceinline__ void ggml_blackwell_mbar_init(
    uint64_t * addr,
    uint32_t arrive_count) {
    // PTX: mbarrier.init.shared.b64 [addr], arrive_count;
    asm volatile ("mbarrier.init.shared.b64 [%0], %1;"
                  :: "r"((uint32_t)__cvta_generic_to_shared(addr)),
                     "r"(arrive_count));
}

// Arrive with expected transaction bytes.
// Called by the TMA-producing warp *before* issuing cp.async.bulk.
// @param addr   Shared-memory mbarrier address
// @param tx_bytes  Total bytes this arrival will move via TMA
__device__ __forceinline__ uint64_t ggml_blackwell_mbar_arrive_expect_tx(
    uint64_t * addr,
    uint32_t tx_bytes) {
    uint64_t state = 0;
    // PTX: mbarrier.arrive.expect_tx.shared.b64 state, [addr], tx_bytes;
    asm volatile ("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;"
                  : "=l"(state)
                  : "r"((uint32_t)__cvta_generic_to_shared(addr)),
                    "r"(tx_bytes));
    return state;
}

// Wait for all expected transactions to complete.
// Called by the WGMMA-consuming warp group.
// @param addr   Shared-memory mbarrier address
// @param state  Value returned by arrive_expect_tx (or prior wait)
__device__ __forceinline__ void ggml_blackwell_mbar_wait(
    uint64_t * addr,
    uint64_t state) {
    // PTX: mbarrier.try_wait.shared.b64 [addr], state;
    // Loop until completion (handle PENDING status).
    for (;;) {
        uint32_t waitComplete;
        asm volatile ("mbarrier.try_wait.shared.b64 %0, [%1], %2;"
                      : "=r"(waitComplete)
                      : "r"((uint32_t)__cvta_generic_to_shared(addr)),
                        "l"(state));
        if (waitComplete) break;
    }
}

// ---------------------------------------------------------------------------
// TMA descriptor cache warm-up (device)
// ---------------------------------------------------------------------------
// Before using a TMA descriptor in a cp.async.bulk instruction, the
// descriptor must be loaded into the TMA descriptor cache.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ggml_blackwell_tma_desc_cache_prefetch(
    const void * desc_ptr) {
    // PTX: cp.async.bulk.prefetch.L2 [desc_ptr], 16;
    asm volatile ("cp.async.bulk.prefetch.L2 [%0], 16;"
                  :: "l"(desc_ptr));
}

// ---------------------------------------------------------------------------
// Early-Start WGMMA orchestration kernel (stub)
// ---------------------------------------------------------------------------
// This device function is the core of the zero-bubble pipeline.
// It runs inside the GPU compute kernel for a single pipeline slot.
//
// Flow per tile:
//   1. Init mbarrier (once per kernel launch, per tile slot)
//   2. TMA warp: arrive_expect_tx(mbar, tile_bytes)
//   3. TMA warp: cp.async.bulk.tensor for K tile + V tile
//   4. TMA warp: cp.async.bulk.commit_group
//   5. WGMMA warp: mbar_wait(mbar, state)  <-- early start!
//   6. WGMMA warp: wgmma.mma_sync using SMEM pointers
//   7. WGMMA warp: wgmma.commit_group + wait_group
//   8. Advance to next tile
//
// The key insight: step 5 unblocks as soon as the TMA engine signals
// the tile is resident, NOT when the whole layer is copied.
// ---------------------------------------------------------------------------

// Shared memory layout per tile slot:
//   [0]                     mbarrier (8 bytes)
//   [8..tile_bytes+8)       K tile buffer
//   [next..next+tile_bytes) V tile buffer
struct ggml_blackwell_smem_tile_slot {
    uint64_t mbar;
    // K and V buffers follow immediately; offsets computed at runtime
};

__device__ inline void * ggml_blackwell_smem_k_ptr(void * smem_base, size_t offs) {
    return reinterpret_cast<char *>(smem_base) + offs;
}

__device__ inline void * ggml_blackwell_smem_v_ptr(void * smem_base, size_t offs, uint32_t tile_bytes) {
    return reinterpret_cast<char *>(smem_base) + offs + tile_bytes;
}

// Per-tile early-start orchestration.
// @param smem_slot   Shared memory base for this tile's mbarrier + buffers
// @param desc_k      TMA descriptor for K source in pinned host memory
// @param desc_v      TMA descriptor for V source in pinned host memory
// @param tile_bytes  Bytes per K or V tile (mbarrier transaction count)
// @param lane_id     Thread lane within warp
// @param is_tma_warp True if this thread is in the TMA-producing warp
// @param is_wgmma_warp True if this thread is in the WGMMA-consuming warp
__device__ __forceinline__ void ggml_blackwell_tile_early_start(
    ggml_blackwell_smem_tile_slot * smem_slot,
    const void * desc_k,
    const void * desc_v,
    uint32_t tile_bytes,
    int lane_id,
    bool is_tma_warp,
    bool is_wgmma_warp) {

    uint64_t * mbar = &smem_slot->mbar;
    void * smem_k = reinterpret_cast<char *>(smem_slot) + sizeof(uint64_t);
    void * smem_v = reinterpret_cast<char *>(smem_k) + tile_bytes;

    // --- Phase 1: Init mbarrier (leader thread only) ---
    if (lane_id == 0) {
        ggml_blackwell_mbar_init(mbar, 1); // 1 arrival (TMA warp)
    }
    __syncwarp();

    // --- Phase 2: TMA warp arrives with expected tx count ---
    uint64_t mbar_state = 0;
    if (is_tma_warp && lane_id == 0) {
        // Total bytes = K tile + V tile
        mbar_state = ggml_blackwell_mbar_arrive_expect_tx(mbar, tile_bytes * 2);
    }
    __syncwarp();

    // --- Phase 3: Issue TMA bulk copies (TMA warp) ---
    if (is_tma_warp) {
        // Prefetch descriptors into cache
        ggml_blackwell_tma_desc_cache_prefetch(desc_k);
        ggml_blackwell_tma_desc_cache_prefetch(desc_v);

        // Issue cp.async.bulk.tensor for K and V
        // These are Blackwell PTX instructions; inline assembly required.
        // We use the uniform variant for contiguous tiles.
        asm volatile ("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], [%2], %3;"
                      :: "r"((uint32_t)__cvta_generic_to_shared(smem_k)),
                         "l"(desc_k),
                         "r"((uint32_t)__cvta_generic_to_shared(mbar)),
                         "r"(tile_bytes)
                      : "memory");

        asm volatile ("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], [%2], %3;"
                      :: "r"((uint32_t)__cvta_generic_to_shared(smem_v)),
                         "l"(desc_v),
                         "r"((uint32_t)__cvta_generic_to_shared(mbar)),
                         "r"(tile_bytes)
                      : "memory");

        // Commit TMA group
        asm volatile ("cp.async.bulk.commit_group;" ::: "memory");
    }

    // --- Phase 4: WGMMA warp waits on mbarrier (EARLY START) ---
    if (is_wgmma_warp) {
        // Wait until both K and V tiles are resident in SMEM
        ggml_blackwell_mbar_wait(mbar, mbar_state);

        // At this point the WGMMA can start computing using smem_k / smem_v
        // even if other tiles in the same layer are still being transferred.
        // Actual WGMMA invocation is model-specific and lives in the caller.
    }
}

#endif // __CUDA_ARCH__ >= 1000 && defined(GGML_CUDA_HAS_WGMMA)

// ---------------------------------------------------------------------------
// Host-side: mbarrier SMEM budget calculation
// ---------------------------------------------------------------------------
// Call at kernel launch time to decide dynamic shared memory size.
// ---------------------------------------------------------------------------

__host__ inline size_t ggml_blackwell_smem_bytes_per_tile_slot(
    const ggml_blackwell_tile_shape & tile) {

    uint32_t tile_bytes = ggml_blackwell_mbar_tx_bytes_kv(tile);
    // mbarrier (8 bytes) + K buffer + V buffer
    size_t smem = sizeof(uint64_t) + 2ull * tile_bytes;
    // Round up to 16-byte alignment for TMA alignment requirements
    smem = (smem + 15) & ~15;
    return smem;
}

// Total SMEM for a kernel that processes `num_tiles_inflight` concurrently.
__host__ inline size_t ggml_blackwell_smem_total(
    const ggml_blackwell_tile_shape & tile,
    int num_tiles_inflight) {
    return num_tiles_inflight * ggml_blackwell_smem_bytes_per_tile_slot(tile);
}

// ---------------------------------------------------------------------------
// Host-side: sanity checks for Blackwell TMA pipeline init
// ---------------------------------------------------------------------------

struct ggml_blackwell_pipeline_caps {
    bool tma_supported;
    bool wgmma_supported;
    int  smem_total;       // Bytes of shared memory per SM
    int  smem_per_block;   // Max static + dynamic SMEM per block
};

__host__ inline ggml_blackwell_pipeline_caps ggml_blackwell_probe_caps(int device_id) {
    ggml_blackwell_pipeline_caps caps = {};
    int cc_major = 0, cc_minor = 0;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device_id);

    int cc = cc_major * 10 + cc_minor;
    caps.tma_supported     = (cc >= 90);   // TMA introduced in Hopper
    caps.wgmma_supported   = (cc >= 100);  // WGMMA in Blackwell

    cudaDeviceGetAttribute(&caps.smem_total,     cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id);
    cudaDeviceGetAttribute(&caps.smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
    return caps;
}

// Validate that the chosen tile shape + inflight count fits in SMEM.
__host__ inline bool ggml_blackwell_validate_smem_config(
    const ggml_blackwell_tile_shape & tile,
    int num_tiles_inflight,
    const ggml_blackwell_pipeline_caps & caps) {

    if (!caps.wgmma_supported) return false;

    size_t smem_needed = ggml_blackwell_smem_total(tile, num_tiles_inflight);
    if (smem_needed > static_cast<size_t>(caps.smem_per_block)) {
        // Fall back to smaller tile or fewer inflight tiles
        return false;
    }
    return true;
}
