#ifdef GGML_USE_CUDA

#include "moe-static-bunker.h"

#include <cstdio>
#include <cstring>

// CUDA runtime API forward declarations
extern "C" {
    int  cudaMalloc(void ** devPtr, size_t size);
    int  cudaFree(void * devPtr);
    int  cudaHostAlloc(void ** pHost, size_t size, unsigned int flags);
    int  cudaFreeHost(void * ptr);
    int  cudaEventCreate(void ** event);
    int  cudaEventDestroy(void * event);
    int  cudaEventRecord(void * event, void * stream);
    int  cudaStreamWaitEvent(void * stream, void * event, unsigned int flags);
    int  cudaStreamCreate(void ** stream);
    int  cudaStreamDestroy(void * stream);
    int  cudaStreamSynchronize(void * stream);
    int  cudaGraphExecDestroy(void * graphExec);
    int  cudaGraphDestroy(void * graph);
    const char * cudaGetErrorString(int);
}
constexpr int cudaHostAllocDefault   = 0;

// --- phase2_hijack ---

void phase2_hijack::init(int n_moe_layers) {
    n_layers = n_moe_layers;
    n_slots = n_layers * N_TYPES;
    slots = new slot[n_slots]();
    if (!slots) { fprintf(stderr, "phase2_hijack: OOM for slots\n"); return; }
    cudaStream_t s;
    cudaStreamCreate(&s);
    stream = s;
    captured = false;
    buffer = nullptr;
    buffer_size = 0;
}

void phase2_hijack::destroy() {
    if (cuda_graph_exec) { cudaGraphExecDestroy((void*)cuda_graph_exec); cuda_graph_exec = nullptr; }
    if (buffer)           { cudaFree(buffer); buffer = nullptr; }
    if (stream)           { cudaStreamSynchronize((void*)stream); cudaStreamDestroy((void*)stream); stream = nullptr; }
    delete[] slots; slots = nullptr;
    n_slots = 0; n_layers = 0;
}

void phase2_hijack::allocate_slots(int /*max_il*/) {
    size_t total = 0;
    for (int i = 0; i < n_slots; i++) {
        if (slots[i].size > 0 && slots[i].parent == -1) {
            total = H2_ALIGN_UP(total);
            total += slots[i].size;
        }
    }
    if (total == 0) {
        fprintf(stderr, "phase2_hijack: no slots to allocate\n");
        return;
    }
    int e = cudaMalloc(&buffer, total);
    if (e != 0) {
        fprintf(stderr, "phase2_hijack: cudaMalloc(%zu) failed: %s\n", total, cudaGetErrorString(e));
        buffer = nullptr;
        return;
    }
    buffer_size = total;
    size_t offset = 0;
    for (int i = 0; i < n_slots; i++) {
        if (slots[i].size > 0 && slots[i].parent == -1) {
            offset = H2_ALIGN_UP(offset);
            slots[i].addr = (char*)buffer + offset;
            offset += slots[i].size;
        }
    }
    fprintf(stderr, "phase2_hijack: allocated %zu byte buffer (%d types x %d layers)\n",
            total, N_TYPES, n_layers);
}

// --- phase2_inject ---

void phase2_inject::init() {
    for (int i = 0; i < H2_N_LAYERS_MAX; i++) {
        cudaHostAlloc((void**)&host_moe_ids[i],  sizeof(int)   * 32, cudaHostAllocDefault);
        cudaHostAlloc((void**)&host_moe_w[i],    sizeof(float) * 32, cudaHostAllocDefault);
    }
}

void phase2_inject::destroy() {
    for (int i = 0; i < H2_N_LAYERS_MAX; i++) {
        if (host_moe_ids[i])  { cudaFreeHost(host_moe_ids[i]);  host_moe_ids[i]  = nullptr; }
        if (host_moe_w[i])    { cudaFreeHost(host_moe_w[i]);    host_moe_w[i]    = nullptr; }
    }
}

void phase2_inject::fill_layer(int il, const int * ids, const float * w) {
    if (il >= H2_N_LAYERS_MAX) return;
    memcpy(host_moe_ids[il], ids, sizeof(int) * 32);
    memcpy(host_moe_w[il],   w,   sizeof(float) * 32);
}

void phase2_inject::inject_all(void * stream) {
    (void)stream;
}

// --- phase2_guard ---

void phase2_guard::init() {
    cudaEventCreate(&phase1_done_event);
}

void phase2_guard::destroy() {
    if (phase1_done_event) { cudaEventDestroy(phase1_done_event); phase1_done_event = nullptr; }
}

void phase2_guard::record(void * stream) {
    cudaEventRecord(phase1_done_event, stream);
}

void phase2_guard::wait(void * stream) {
    cudaStreamWaitEvent((void*)stream, (void*)phase1_done_event, 0);
}

#endif // GGML_USE_CUDA