#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a pinned buffer region
typedef struct ggml_pinned_buffer * ggml_pinned_buffer_t;

// Create a pinned buffer region.
// total_bytes: total size of the mmap'd region
// window_bytes: size of each sliding window (active pinned subset)
// Returns nullptr on failure (falls back to regular heap).
ggml_pinned_buffer_t ggml_pinned_buffer_create(size_t total_bytes, size_t window_bytes);

void ggml_pinned_buffer_free(ggml_pinned_buffer_t buf);

// Allocate from the pinned buffer. Returns nullptr if exhausted.
// alignment: must be power of 2 (typically 128 for TMA)
void * ggml_pinned_buffer_alloc(ggml_pinned_buffer_t buf, size_t bytes, size_t alignment);

// Free an allocation back to the slab.
void ggml_pinned_buffer_free_block(ggml_pinned_buffer_t buf, void * ptr);

// Compact the allocator (shift live blocks to reduce fragmentation).
// Returns number of bytes moved.
size_t ggml_pinned_buffer_compact(ggml_pinned_buffer_t buf);

// Get fragmentation ratio (0.0 = none, 1.0 = fully fragmented).
float ggml_pinned_buffer_fragmentation(ggml_pinned_buffer_t buf);

// Slide the active window. Old window pages are madvise(DONTNEED).
// Returns true if slide succeeded.
bool ggml_pinned_buffer_slide_window(ggml_pinned_buffer_t buf);

#ifdef __cplusplus
}
#endif
