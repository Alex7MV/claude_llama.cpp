#pragma once
#include "ggml-backend.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Slab allocator for pinned (MAP_LOCKED) system RAM.
// Single large mmap, internal block tracking, first-fit + bump.
// 128-byte alignment by default (TMA requirement).

typedef struct ggml_pinned_region * ggml_pinned_region_t;

// Create a pinned memory region. region_size_bytes is the total size.
ggml_pinned_region_t ggml_pinned_region_new(size_t region_size_bytes);

// Release the region (munmap + free metadata).
void ggml_pinned_region_free(ggml_pinned_region_t region);

// Allocate a block from the region. Returns nullptr on OOM.
void * ggml_pinned_region_alloc(ggml_pinned_region_t region, size_t size, size_t alignment);

// Free a previously allocated block.
void ggml_pinned_region_free_block(ggml_pinned_region_t region, void * ptr);

// Compact: shift live blocks to the front, merge free space.
bool ggml_pinned_region_compact(ggml_pinned_region_t region);

// Return fragmentation ratio (0.0 = none, 1.0 = worst case).
double ggml_pinned_region_fragmentation(ggml_pinned_region_t region);

// Return peak usage since region creation.
size_t ggml_pinned_region_high_watermark(ggml_pinned_region_t region);

#ifdef __cplusplus
}
#endif
