#pragma once

// GGML CPU EPYC internal header - CCD topology types (private, no API guarantee)

#include "ggml-cpu-epyc.h"
#include <stdint.h>

#define GGML_EPYC_MAX_CCDs 16
#define GGML_EPYC_MAX_CCD_PAIRS 8

struct ggml_ccd_group {
    uint32_t ccd_id;
    uint32_t thread_count;
    uint32_t threads[GGML_NUMA_MAX_CPUS];
};

struct ggml_ccd_topology {
    uint32_t n_ccds;
    uint32_t ccd_threads[GGML_NUMA_MAX_CPUS];
    uint32_t ccd_thread_count[GGML_NUMA_MAX_CPUS];
    uint32_t ccd_for_cpu[GGML_NUMA_MAX_CPUS];
    uint32_t total_threads;
    struct ggml_ccd_group groups[GGML_EPYC_MAX_CCDs];
};
