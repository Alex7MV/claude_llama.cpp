#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pipelined backend scheduler for hybrid CPU+GPU prefill.
// Wraps an existing ggml_backend_sched and executes its splits
// in a depth-3 pipeline: CPU compute -> TMA transfer -> GPU compute.

typedef struct ggml_backend_sched_pipelined * ggml_backend_sched_pipelined_t;

ggml_backend_sched_pipelined_t ggml_backend_sched_pipelined_init(
    ggml_backend_sched_t base,
    int depth,
    int split_size,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll,
    ggml_backend_t gpu_backend);

enum ggml_status ggml_backend_sched_pipelined_compute(
    ggml_backend_sched_pipelined_t sched,
    struct ggml_cgraph * gf);

void ggml_backend_sched_pipelined_synchronize(ggml_backend_sched_pipelined_t sched);

void ggml_backend_sched_pipelined_free(ggml_backend_sched_pipelined_t sched);

/**
 * Compute a single split graph within the pipeline.
 * split_index determines which stage event to wait on and rotate.
 */
enum ggml_status ggml_backend_sched_pipelined_compute_split(
    ggml_backend_sched_pipelined_t sched,
    struct ggml_cgraph * gf,
    int split_index);

/** Get the pipeline depth (number of stage events). */
int ggml_backend_sched_pipelined_get_depth(ggml_backend_sched_pipelined_t sched);

/** Record stage completion event from caller (e.g., after GPU dispatch). */
void ggml_backend_sched_pipelined_record_stage(ggml_backend_sched_pipelined_t sched, int stage);

/** Wait for stage event from caller side. */
void ggml_backend_sched_pipelined_wait_stage(ggml_backend_sched_pipelined_t sched, int stage);

#ifdef __cplusplus
}
#endif
