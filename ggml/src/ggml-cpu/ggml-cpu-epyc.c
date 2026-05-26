#include "ggml-cpu-epyc-impl.h"
#include "ggml-cpu-impl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#if defined(__gnu_linux__)

// Internal helper to populate CCD pairs from the shared topology.
// The topology is probed once by ggml_numa_init() in ggml-cpu.c.
static int ggml_epyc_fill_ccd_pairs(const struct ggml_ccd_topology *topo,
                                     struct ggml_cpu_ccd_pair pairs[], int max_pairs) {
    if (!topo || !pairs || max_pairs < 1 || topo->n_ccds < 2) return 0;

    int pc = 0;
    // Pair 0: first CCD + last CCD (NUMA 0 + NUMA 1)
    pairs[0].ccd_indices[0] = 0;
    pairs[0].ccd_indices[1] = (int)topo->n_ccds - 1;
    pairs[0].thread_count = 0;
    for (uint32_t i = 0; i < topo->total_threads; i++) {
        uint32_t cpu = topo->ccd_threads[i];
        uint32_t ccd = topo->ccd_for_cpu[cpu];
        if (ccd == 0 || ccd == topo->n_ccds - 1) {
            pairs[0].threads[pairs[0].thread_count++] = cpu;
        }
    }
    pc++;

    // Pair 1: second + second-to-last CCD (if >= 4 CCDs)
    if (topo->n_ccds >= 4 && max_pairs >= 2) {
        pairs[1].ccd_indices[0] = 1;
        pairs[1].ccd_indices[1] = (int)topo->n_ccds - 2;
        pairs[1].thread_count = 0;
        for (uint32_t i = 0; i < topo->total_threads; i++) {
            uint32_t cpu = topo->ccd_threads[i];
            uint32_t ccd = topo->ccd_for_cpu[cpu];
            if (ccd == 1 || ccd == topo->n_ccds - 2) {
                pairs[1].threads[pairs[1].thread_count++] = cpu;
            }
        }
        pc++;
    }
    return pc;
}

int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs) {
    const struct ggml_ccd_topology *topo = ggml_cpu_get_ccd_topology();
    return ggml_epyc_fill_ccd_pairs(topo, pairs, max_pairs);
}

int ggml_cpu_init_dual_threadpool(struct ggml_threadpool_params params_out[],
                                   int count, int threads_per_pair,
                                   enum ggml_sched_priority prio, uint32_t poll) {
    if (threads_per_pair < 1) return 0;
    struct ggml_cpu_ccd_pair pairs[GGML_EPYC_MAX_CCD_PAIRS];
    int np = ggml_cpu_probe_ccd_pairs(pairs, GGML_EPYC_MAX_CCD_PAIRS);
    if (np < 2 || count < 2 || !params_out) return 0;

    for (int p = 0; p < 2; p++) {
        memset(&params_out[p], 0, sizeof(struct ggml_threadpool_params));
        int nt = threads_per_pair < (int)pairs[p].thread_count ? threads_per_pair : (int)pairs[p].thread_count;
        params_out[p].n_threads = nt;
        params_out[p].prio = prio;
        params_out[p].poll = poll;
        params_out[p].strict_cpu = true;
        params_out[p].paused = false;
        for (int t = 0; t < nt && t < (int)pairs[p].thread_count; t++) {
            uint32_t cpu = pairs[p].threads[t];
            if (cpu < GGML_MAX_N_THREADS) {
                params_out[p].cpumask[cpu] = true;
            }
        }
    }
    return 2;
}

#else // non-Linux stubs

int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs) {
    (void)pairs; (void)max_pairs;
    return 0;
}

int ggml_cpu_init_dual_threadpool(struct ggml_threadpool_params params_out[],
                                   int count, int threads_per_pair,
                                   enum ggml_sched_priority prio, uint32_t poll) {
    (void)params_out; (void)count; (void)threads_per_pair; (void)prio; (void)poll;
    return 0;
}

#endif
