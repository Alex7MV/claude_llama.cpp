#pragma once

#include <stdint.h>
#include <stdbool.h>

#if defined(__linux__) && defined(__x86_64__)

#define GGML_EPYC_MAX_CPUS 1024

struct ggml_epyc_ccd_topology {
    uint32_t n_ccds;
    uint32_t ccd_threads[GGML_EPYC_MAX_CPUS];
    uint32_t ccd_thread_count[GGML_EPYC_MAX_CPUS];
    uint32_t ccd_for_cpu[GGML_EPYC_MAX_CPUS];
    uint32_t total_threads;
};

// Parse /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
// Returns true if this CPU is an SMT sibling (not the primary).
bool ggml_epyc_cpu_is_smt_sibling(int cpu_id);

// Parse /sys/devices/system/cpu/cpu*/cache/index3/id
// Returns CCD ID or -1 on error.
int ggml_epyc_probe_ccd_id(int cpu_id);

#endif // __linux__ && __x86_64__
