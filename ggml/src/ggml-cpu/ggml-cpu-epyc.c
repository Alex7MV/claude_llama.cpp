#include "ggml-cpu-epyc.h"
#include "ggml-cpu-epyc-impl.h"
#include "ggml.h"

#include <string.h>

#if defined(__linux__) && defined(__x86_64__)

// The CCD topology probing and dual threadpool init are kept in ggml-cpu.c
// because they share g_state.ccd which is used by threadpool affinity code.
// This file provides the EPYC-specific entry point wrapper.

int ggml_cpu_init_dual_threadpool_epyc(
    struct ggml_threadpool_params params[],
    int max_params,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll) {

    if (!params || max_params < 2 || n_threads <= 0) {
        return 0;
    }

    int threads_per_pair = n_threads / 2;
    if (threads_per_pair < 1) {
        threads_per_pair = 1;
    }

    // Delegate to the existing implementation in ggml-cpu.c
    return ggml_cpu_init_dual_threadpool(params, max_params, threads_per_pair, prio, poll);
}

#else // !(__linux__ && __x86_64__)

int ggml_cpu_init_dual_threadpool_epyc(
    struct ggml_threadpool_params params[],
    int max_params,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll) {
    (void)params; (void)max_params; (void)n_threads; (void)prio; (void)poll;
    return 0;
}

#endif // __linux__ && __x86_64__
