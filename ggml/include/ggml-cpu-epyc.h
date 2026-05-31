#pragma once

// EPYC-specific CPU backend extensions.
// These functions are declared in ggml-cpu.h; implementations live in
// ggml-cpu-epyc.c (Linux x86_64 only).
//
// Include this header when you need the EPYC-specific API explicitly,
// or include ggml-cpu.h which already declares:
//   - struct ggml_cpu_ccd_pair
//   - ggml_cpu_probe_ccd_pairs()
//   - ggml_cpu_init_dual_threadpool()

#include "ggml-cpu.h"

int ggml_cpu_init_dual_threadpool_epyc(
    struct ggml_threadpool_params params[],
    int max_params,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll);
