#pragma once
#include "ggml-cpu.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GGML_NUMA_MAX_CPUS
#define GGML_NUMA_MAX_CPUS 512
#endif
#ifndef GGML_NUMA_MAX_NODES
#define GGML_NUMA_MAX_NODES 8
#endif

/**
 * A pair of CCDs from opposite NUMA nodes for dual threadpool affinity.
 * Each CCD contributes its primary + SMT threads to the shared pool.
 */
struct ggml_cpu_ccd_pair {
    int ccd_indices[2];
    uint32_t threads[GGML_NUMA_MAX_CPUS];
    uint32_t thread_count;
};

/**
 * Probe CCD topology and return pairs of CCDs suitable for dual threadpool.
 * Returns number of pairs (0, 1, or 2). Each pair contains two CCDs
 * from opposite NUMA nodes for symmetric access.
 * Requires __linux__ && __x86_64__; returns 0 on other platforms.
 */
GGML_BACKEND_API int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs);

/**
 * Initialize dual threadpool parameters for EPYC CCD pairs.
 * Fills params_out[0] and params_out[1] with CCD-affine CPU masks.
 * Returns 2 on success, 0 on failure (no CCDs detected or count < 2).
 */
GGML_BACKEND_API int ggml_cpu_init_dual_threadpool(
    struct ggml_threadpool_params params_out[],
    int count,
    int threads_per_pair,
    enum ggml_sched_priority prio,
    uint32_t poll);

#ifdef __cplusplus
}
#endif
