#include "ggml-pinned-buffer.h"
#include "ggml.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <algorithm>

struct ggml_pinned_buffer {
    char * base;
    size_t total_bytes;
    size_t window_bytes;
    size_t active_window;

    struct block {
        size_t offset;
        size_t size;
        bool   live;
    };
    std::vector<block> blocks;
    size_t allocated_bytes;
    size_t high_watermark;
};

ggml_pinned_buffer_t ggml_pinned_buffer_create(size_t total_bytes, size_t window_bytes) {
    if (total_bytes == 0 || window_bytes == 0 || window_bytes > total_bytes) {
        return nullptr;
    }

    void * ptr = mmap(nullptr, total_bytes,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED,
        -1, 0);
    if (ptr == MAP_FAILED) {
        GGML_LOG_WARN("pinned-buffer: mmap(MAP_LOCKED) failed (%zu bytes), falling back to regular heap\n", total_bytes);
        return nullptr;
    }

    auto * buf = new ggml_pinned_buffer();
    buf->base = (char *)ptr;
    buf->total_bytes = total_bytes;
    buf->window_bytes = window_bytes;
    buf->active_window = 0;
    buf->allocated_bytes = 0;
    buf->high_watermark = 0;

    // Initial free block
    buf->blocks.push_back({0, window_bytes, false});

    GGML_LOG_INFO("pinned-buffer: created %zu bytes (%zu window) at %p\n",
        total_bytes, window_bytes, ptr);
    return buf;
}

void ggml_pinned_buffer_free(ggml_pinned_buffer_t buf) {
    if (!buf) return;
    if (buf->base) {
        munmap(buf->base, buf->total_bytes);
    }
    delete buf;
}

void * ggml_pinned_buffer_alloc(ggml_pinned_buffer_t buf, size_t bytes, size_t alignment) {
    if (!buf || bytes == 0) return nullptr;

    size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);

    // First-fit search in current window
    for (auto & blk : buf->blocks) {
        if (!blk.live && blk.size >= aligned_bytes) {
            blk.live = true;
            if (blk.size > aligned_bytes) {
                // Split: create new free block after this one
                size_t remaining = blk.size - aligned_bytes;
                buf->blocks.push_back({blk.offset + aligned_bytes, remaining, false});
                blk.size = aligned_bytes;
            }
            buf->allocated_bytes += aligned_bytes;
            if (buf->allocated_bytes > buf->high_watermark) {
                buf->high_watermark = buf->allocated_bytes;
            }
            return buf->base + blk.offset;
        }
    }

    // Exhausted — try compact
    float frag = ggml_pinned_buffer_fragmentation(buf);
    if (frag > 0.25f) {
        ggml_pinned_buffer_compact(buf);
        // Retry allocation after compact
        for (auto & blk : buf->blocks) {
            if (!blk.live && blk.size >= aligned_bytes) {
                blk.live = true;
                if (blk.size > aligned_bytes) {
                    size_t remaining = blk.size - aligned_bytes;
                    buf->blocks.push_back({blk.offset + aligned_bytes, remaining, false});
                    blk.size = aligned_bytes;
                }
                buf->allocated_bytes += aligned_bytes;
                if (buf->allocated_bytes > buf->high_watermark) {
                    buf->high_watermark = buf->allocated_bytes;
                }
                return buf->base + blk.offset;
            }
        }
    }

    // Still exhausted — caller falls back to heap
    return nullptr;
}

void ggml_pinned_buffer_free_block(ggml_pinned_buffer_t buf, void * ptr) {
    if (!buf || !ptr) return;
    size_t offset = (char *)ptr - buf->base;
    for (auto & blk : buf->blocks) {
        if (blk.offset == offset && blk.live) {
            blk.live = false;
            buf->allocated_bytes -= blk.size;
            // Simple coalesce: merge with adjacent free blocks
            for (auto it = buf->blocks.begin(); it != buf->blocks.end(); ) {
                if (!it->live) {
                    auto next = it + 1;
                    while (next != buf->blocks.end() && !next->live && next->offset == it->offset + it->size) {
                        it->size += next->size;
                        next = buf->blocks.erase(next);
                    }
                }
                ++it;
            }
            return;
        }
    }
}

float ggml_pinned_buffer_fragmentation(ggml_pinned_buffer_t buf) {
    if (!buf || buf->window_bytes == 0) return 0.0f;
    size_t free_bytes = 0;
    size_t largest_free = 0;
    for (const auto & blk : buf->blocks) {
        if (!blk.live) {
            free_bytes += blk.size;
            if (blk.size > largest_free) largest_free = blk.size;
        }
    }
    if (free_bytes == 0) return 0.0f;
    // Fragmentation = 1 - (largest_free / total_free)
    return 1.0f - ((float)largest_free / (float)free_bytes);
}

size_t ggml_pinned_buffer_compact(ggml_pinned_buffer_t buf) {
    if (!buf || buf->blocks.empty()) return 0;

    // Sort blocks by offset
    std::sort(buf->blocks.begin(), buf->blocks.end(),
        [](const auto & a, const auto & b) { return a.offset < b.offset; });

    size_t moved = 0;
    size_t write_offset = 0;

    for (auto & blk : buf->blocks) {
        if (blk.live) {
            if (blk.offset != write_offset) {
                memmove(buf->base + write_offset, buf->base + blk.offset, blk.size);
                blk.offset = write_offset;
                moved += blk.size;
            }
            write_offset += blk.size;
        }
    }

    // Consolidate all trailing space into one free block
    buf->blocks.erase(
        std::remove_if(buf->blocks.begin(), buf->blocks.end(),
            [](const auto & blk) { return !blk.live; }),
        buf->blocks.end());

    if (write_offset < buf->window_bytes) {
        buf->blocks.push_back({write_offset, buf->window_bytes - write_offset, false});
    }

    return moved;
}

bool ggml_pinned_buffer_slide_window(ggml_pinned_buffer_t buf) {
    if (!buf) return false;
    size_t next_window = buf->active_window + buf->window_bytes;
    if (next_window + buf->window_bytes > buf->total_bytes) {
        next_window = 0; // Wrap around
    }

    // madvise old window to release physical pages
    if (madvise(buf->base + buf->active_window, buf->window_bytes, MADV_DONTNEED) != 0) {
        GGML_LOG_WARN("pinned-buffer: madvise(DONTNEED) failed\n");
    }

    buf->active_window = next_window;
    buf->blocks.clear();
    buf->blocks.push_back({next_window, buf->window_bytes, false});
    buf->allocated_bytes = 0;

    GGML_LOG_INFO("pinned-buffer: slid window to offset %zu\n", next_window);
    return true;
}
