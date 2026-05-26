#pragma once

// GGML CPU EPYC internal header - CCD topology types (private, no API guarantee)

#include "ggml-cpu-epyc.h"
#include <stdint.h>
#include <stdbool.h>

#define GGML_EPYC_MAX_CCD_PAIRS 2

// CCD topology as probed once by ggml_numa_init() in ggml-cpu.c.
// This struct mirrors the layout in ggml-cpu.c.
struct ggml_ccd_topology {
    uint32_t n_ccds;
    uint32_t ccd_threads[GGML_NUMA_MAX_CPUS];
    uint32_t ccd_thread_count[GGML_NUMA_MAX_CPUS];
    uint32_t ccd_for_cpu[GGML_NUMA_MAX_CPUS];
    uint32_t total_threads;
};

// Accessors provided by ggml-cpu.c — topology is probed once at startup.
#ifdef __cplusplus
extern "C" {
#endif
const struct ggml_ccd_topology * ggml_cpu_get_ccd_topology(void);
bool ggml_cpu_thread_apply_affinity(const bool * mask);
#ifdef __cplusplus
}
#endif
