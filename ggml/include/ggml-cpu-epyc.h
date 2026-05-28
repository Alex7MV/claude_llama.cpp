#pragma once

#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__linux__) && defined(__x86_64__)

// Maximum CCD pairs we'll detect
#define GGML_EPYC_MAX_CCD_PAIRS 16

// One CCD pair: two CCDs that share a NUMA node or are neighbor-linked
struct ggml_cpu_ccd_pair {
    int ccd_ids[2];        // CCD IDs in this pair (-1 if singleton)
    int numa_node;         // NUMA node affinity for this pair
    int n_threads;         // Total threads assignable to this pair
};

// Probe CCD topology and build pairs.
// @param pairs     Output array (caller allocates GGML_EPYC_MAX_CCD_PAIRS)
// @param max_pairs Size of pairs array
// @return          Number of pairs populated (0 if not EPYC or probing failed)
int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs);

// Initialize dual threadpools from CCD pairs.
// @param params     Output threadpool params array (size >= 2)
// @param max_params Size of params array
// @param n_threads  Total threads to distribute across pairs
// @param prio       Thread priority
// @param poll       Polling flag
// @return           Number of threadpools created (0, 1, or 2)
int ggml_cpu_init_dual_threadpool_epyc(
    struct ggml_threadpool_params params[],
    int max_params,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll);

#else // !(__linux__ && __x86_64__)

static inline int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs) {
    (void)pairs; (void)max_pairs;
    return 0;
}

static inline int ggml_cpu_init_dual_threadpool_epyc(
    struct ggml_threadpool_params params[],
    int max_params,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll) {
    (void)params; (void)max_params; (void)n_threads; (void)prio; (void)poll;
    return 0;
}

#endif // __linux__ && __x86_64__

#ifdef __cplusplus
}
#endif
