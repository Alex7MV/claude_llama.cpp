// tools/regression/cuda_bench.cu
// Standalone CUDA H2D bandwidth benchmark.
// Compile: nvcc -O3 -o cuda_bench cuda_bench.cu
// Run:     ./cuda_bench [--p2p]

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Bandwidth test sizes: 1 MB, 10 MB, 100 MB, 1 GB
static const size_t test_sizes[] = {
    1UL << 20,
    10UL << 20,
    100UL << 20,
    1UL << 30
};
static const int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

// Returns H2D bandwidth in GB/s for the largest tested size
static double bench_h2d(size_t max_size, int num_devices) {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    // Use at most half of free memory
    size_t alloc_size = max_size;
    if (alloc_size > free_mem / 2) {
        alloc_size = free_mem / 2;
    }
    // Round down to nearest test_size
    for (int i = num_sizes - 1; i >= 0; i--) {
        if (test_sizes[i] <= alloc_size) {
            alloc_size = test_sizes[i];
            break;
        }
    }
    if (alloc_size < (1UL << 20)) {
        alloc_size = 1UL << 20;  // minimum 1 MB
    }

    // Allocate pinned host memory
    void *host_buf = NULL;
    CUDA_CHECK(cudaHostAlloc(&host_buf, alloc_size, cudaHostAllocDefault));
    memset(host_buf, 0xAB, alloc_size);

    // Allocate device memory
    void *dev_buf = NULL;
    CUDA_CHECK(cudaMalloc(&dev_buf, alloc_size));

    // Warmup
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemcpy(dev_buf, host_buf, alloc_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double best_sec = 1e100;
    const int num_trials = 10;
    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(dev_buf, host_buf, alloc_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double sec = ms / 1000.0;
        if (sec > 0 && sec < best_sec) best_sec = sec;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dev_buf));
    CUDA_CHECK(cudaFreeHost(host_buf));

    // Bandwidth in GB/s: (bytes / sec) / 1e9
    return (double)alloc_size / best_sec / 1e9;
}

int main(int argc, char **argv) {
    int do_p2p = (argc > 1 && strcmp(argv[1], "--p2p") == 0);

    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // H2D benchmark
    double h2d_gbs = bench_h2d(1UL << 30, num_devices);

    // P2P (reserved for future)
    int p2p_enabled = 0;
    double p2p_gbs = 0.0;
    if (do_p2p && num_devices > 1) {
        for (int i = 0; i < num_devices && !p2p_enabled; i++) {
            for (int j = 0; j < num_devices; j++) {
                if (i == j) continue;
                int can = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
                if (can) { p2p_enabled = 1; break; }
            }
        }
    }

    printf("{\n");
    printf("  \"gpu_name\": \"%s\",\n", prop.name);
    printf("  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"num_devices\": %d,\n", num_devices);
    printf("  \"h2d_bandwidth_gbs\": %.1f,\n", h2d_gbs);
    printf("  \"h2d_max_size_bytes\": %zu,\n", (size_t)(1UL << 30));
    printf("  \"p2p_enabled\": %s,\n", p2p_enabled ? "true" : "false");
    printf("  \"p2p_bandwidth_gbs\": %s\n", p2p_enabled ? "0.0" : "null");
    printf("}\n");

    return 0;
}
