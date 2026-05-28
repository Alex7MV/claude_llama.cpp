#pragma once

// Internal EPYC helpers.
// The CCD topology state (g_state.ccd) remains in ggml-cpu.c because it is
// shared with threadpool affinity code. This header is reserved for future
// EPYC-specific internals that can be decoupled.
