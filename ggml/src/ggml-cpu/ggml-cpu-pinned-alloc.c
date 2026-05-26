#include "ggml-cpu-pinned-alloc.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <stdio.h>

#define GGML_PINNED_MIN_REGION (1ULL << 30)  // 1GB minimum
#define GGML_PINNED_ALIGNMENT 128
#define GGML_PINNED_MAX_BLOCKS 65536

struct pinned_block {
    size_t offset;
    size_t size;
    bool used;
};

struct ggml_pinned_region {
    void * base;
    size_t total_size;
    size_t allocated;
    size_t high_water;
    uint32_t block_count;
    struct pinned_block blocks[GGML_PINNED_MAX_BLOCKS];
};

ggml_pinned_region_t ggml_pinned_region_new(size_t region_size) {
    if (region_size < GGML_PINNED_MIN_REGION) region_size = GGML_PINNED_MIN_REGION;

    void * base = mmap(NULL, region_size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
    if (base == MAP_FAILED) return NULL;

    struct ggml_pinned_region * region =
        (struct ggml_pinned_region*)calloc(1, sizeof(*region));
    if (!region) {
        munmap(base, region_size);
        return NULL;
    }
    region->base       = base;
    region->total_size = region_size;
    return region;
}

void ggml_pinned_region_free(ggml_pinned_region_t region) {
    if (!region) return;
    if (region->base && region->base != MAP_FAILED) {
        munmap(region->base, region->total_size);
    }
    free(region);
}

void * ggml_pinned_region_alloc(ggml_pinned_region_t region, size_t size, size_t align) {
    if (!region || size == 0) return NULL;
    if (align < GGML_PINNED_ALIGNMENT) align = GGML_PINNED_ALIGNMENT;

    size = (size + align - 1) & ~(align - 1);

    // First-fit search in existing free blocks
    for (uint32_t i = 0; i < region->block_count; i++) {
        if (!region->blocks[i].used && region->blocks[i].size >= size) {
            region->blocks[i].size = size;
            region->blocks[i].used = true;
            region->allocated += size;
            if (region->allocated > region->high_water) region->high_water = region->allocated;
            return (char*)region->base + region->blocks[i].offset;
        }
    }

    // New allocation at end (bump pointer)
    size_t end_offset = region->allocated;
    end_offset = (end_offset + align - 1) & ~(align - 1);

    if (end_offset + size > region->total_size) {
        // Try compacting and retry
        if (!ggml_pinned_region_compact(region)) return NULL;
        for (uint32_t i = 0; i < region->block_count; i++) {
            if (!region->blocks[i].used && region->blocks[i].size >= size) {
                region->blocks[i].used = true;
                region->allocated += size;
                return (char*)region->base + region->blocks[i].offset;
            }
        }
        return NULL;
    }

    if (region->block_count >= GGML_PINNED_MAX_BLOCKS) return NULL;

    uint32_t idx = region->block_count++;
    region->blocks[idx].offset = end_offset;
    region->blocks[idx].size   = size;
    region->blocks[idx].used   = true;
    region->allocated += size;
    if (region->allocated > region->high_water) region->high_water = region->allocated;

    return (char*)region->base + end_offset;
}

void ggml_pinned_region_free_block(ggml_pinned_region_t region, void * ptr) {
    if (!region || !ptr) return;
    char * p = (char*)ptr;
    for (uint32_t i = 0; i < region->block_count; i++) {
        if (region->blocks[i].used &&
                (char*)region->base + region->blocks[i].offset == p) {
            region->allocated -= region->blocks[i].size;
            region->blocks[i].used = false;
            return;
        }
    }
}

bool ggml_pinned_region_compact(ggml_pinned_region_t region) {
    if (!region) return false;
    size_t new_offset = 0;
    for (uint32_t i = 0; i < region->block_count; i++) {
        if (region->blocks[i].used) {
            size_t aligned = (new_offset + GGML_PINNED_ALIGNMENT - 1) &
                             ~(GGML_PINNED_ALIGNMENT - 1);
            if (aligned != region->blocks[i].offset) {
                void * src = (char*)region->base + region->blocks[i].offset;
                void * dst = (char*)region->base + aligned;
                memmove(dst, src, region->blocks[i].size);
                region->blocks[i].offset = aligned;
            }
            new_offset = aligned + region->blocks[i].size;
        }
    }
    region->allocated = new_offset;
    return true;
}

double ggml_pinned_region_fragmentation(ggml_pinned_region_t region) {
    if (!region || region->block_count == 0) return 0.0;
    size_t used = 0, free_space = 0;
    uint32_t free_blocks = 0;
    for (uint32_t i = 0; i < region->block_count; i++) {
        if (region->blocks[i].used) used += region->blocks[i].size;
        else { free_space += region->blocks[i].size; free_blocks++; }
    }
    if (free_blocks < 2) return 0.0;
    return 1.0 - (double)free_space / (region->total_size - used);
}

size_t ggml_pinned_region_high_watermark(ggml_pinned_region_t region) {
    return region ? region->high_water : 0;
}
