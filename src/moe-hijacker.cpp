#ifdef GGML_USE_CUDA

#include "moe-hijacker.h"
#include "moe-static-bunker.h"
#include "ggml.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

extern "C" {
    int cudaMemcpyAsync(void * dst, const void * src, size_t count, int kind, void * stream);
}
constexpr int cudaMemcpyDefault = 4;

namespace moe {

std::pair<int,int> match_hijack_name(phase2_hijack & h2, const char * name) {
    if (!name || !name[0]) return {-1, -1};
    const char * dash = strrchr(name, '-');
    if (!dash) return {-1, -1};
    int il = 0;
    for (const char * p = dash + 1; *p; p++) {
        if (*p < '0' || *p > '9') return {-1, -1};
        il = il * 10 + (*p - '0');
    }
    size_t prefix_len = (size_t)(dash - name);
    for (int t = 0; t < h2.N_TYPES; t++) {
        size_t pat_len = strlen(h2.TENSOR_NAMES[t]);
        if (prefix_len == pat_len && memcmp(name, h2.TENSOR_NAMES[t], prefix_len) == 0) {
            return {t, il};
        }
    }
    return {-1, -1};
}

int scan_and_hijack(phase2_hijack & h2, ggml_cgraph * gf) {
    h2.slot_scan_count = 0;
    int total_nodes = ggml_graph_n_nodes(gf);
    int total_leafs = ggml_graph_n_leafs(gf);
    int total_t = total_leafs + total_nodes;
    int max_il = 0;
    struct Match { ggml_tensor * t; int type_id; int il; void * orig; };
    std::vector<Match> matches, views;
    for (int i = 0; i < total_t; i++) {
        ggml_tensor * t = (i < total_leafs) ? ggml_graph_leaf(gf, i) : ggml_graph_node(gf, i - total_leafs);
        auto [type_id, il] = match_hijack_name(h2, t->name);
        if (type_id < 0 || type_id >= h2.N_TYPES || il < 0 || il >= h2.n_layers) continue;
        if (il > max_il) max_il = il;
        size_t sz = ggml_type_size(t->type);
        for (int d = 0; d < 4; d++) sz *= t->ne[d] > 0 ? t->ne[d] : 1;
        int s_idx = h2.slot_idx(il, type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots) continue;
        if (h2.slots[s_idx].size == 0) {
            h2.slots[s_idx].size = H2_ALIGN_UP(sz);
            h2.slots[s_idx].addr = nullptr; h2.slots[s_idx].parent = -1; h2.slots[s_idx].offset = 0;
        }
        if (t->view_src) {
            ptrdiff_t off = (char*)t->data - (char*)t->view_src->data;
            h2.slots[s_idx].offset = off; h2.slots[s_idx].parent = -2;
            views.push_back({t, type_id, il, t->data});
        } else {
            h2.slots[s_idx].parent = -1;
            matches.push_back({t, type_id, il, t->data});
        }
    }
    if (matches.empty() && views.empty()) { fprintf(stderr, "phase2_hijack: no tensors matched!\n"); return 0; }
    if (!h2.buffer) { h2.allocate_slots(max_il + 1); if (!h2.buffer) return 0; }
    for (auto & m : matches) {
        int s_idx = h2.slot_idx(m.il, m.type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots || !h2.slots[s_idx].addr) { continue; }
        h2.slots[s_idx].orig_data = m.orig; m.t->data = h2.slots[s_idx].addr; h2.slot_scan_count++;
    }
    for (auto & m : views) {
        if (!m.t->view_src) { continue; }
        auto [pid, pl] = match_hijack_name(h2, m.t->view_src->name);
        if (pid < 0 || pid >= h2.N_TYPES) { continue; }
        int p_idx = h2.slot_idx(pl, pid);
        if (p_idx < 0 || p_idx >= h2.n_slots || !h2.slots[p_idx].addr) { continue; }
        int s_idx = h2.slot_idx(m.il, m.type_id);
        void * view_addr = (char*)h2.slots[p_idx].addr + h2.slots[s_idx].offset;
        if (s_idx >= 0 && s_idx < h2.n_slots) { h2.slots[s_idx].addr = view_addr; h2.slots[s_idx].orig_data = m.orig; }
        m.t->data = view_addr; h2.slot_scan_count++;
    }
    fprintf(stderr, "phase2_hijack: %d/%d matched (%d layers)\n", h2.slot_scan_count, (int)(matches.size()+views.size()), max_il+1);
    return h2.slot_scan_count;
}

bool scan_and_update_snapshots(phase2_hijack & h2, ggml_cgraph * gf) {
    int total_nodes = ggml_graph_n_nodes(gf), total_leafs = ggml_graph_n_leafs(gf), found = 0;
    for (int i = 0; i < total_leafs + total_nodes; i++) {
        ggml_tensor * t = (i < total_leafs) ? ggml_graph_leaf(gf, i) : ggml_graph_node(gf, i - total_leafs);
        auto [type_id, il] = match_hijack_name(h2, t->name);
        if (type_id < 0 || type_id >= h2.N_TYPES || il < 0 || il >= h2.n_layers) continue;
        int s_idx = h2.slot_idx(il, type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots || !h2.slots[s_idx].addr) continue;
        if (t->data != h2.slots[s_idx].addr) { h2.slots[s_idx].orig_data = t->data; t->data = h2.slots[s_idx].addr; }
        found++;
    }
    return found > 0;
}

void restore_all(phase2_hijack & h2) { (void)h2; asm volatile("" ::: "memory"); }

void copy_data_to_static(phase2_hijack & h2, void * cuda_stream) {
    for (int i = 0; i < h2.n_slots; i++)
        if (h2.slots[i].addr && h2.slots[i].orig_data && h2.slots[i].size > 0)
            cudaMemcpyAsync(h2.slots[i].addr, h2.slots[i].orig_data, h2.slots[i].size, cudaMemcpyDefault, (void*)cuda_stream);
}

void cascade_force_moe_consumers(phase2_hijack & h2, ggml_cgraph * gf, ggml_backend_sched_t sched, ggml_backend_t gpu_backend) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op != 37 && t->op != 2) continue;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) {
            auto [sid, sil] = match_hijack_name(h2, t->src[s]->name);
            if (sid >= 0) { ggml_backend_sched_set_tensor_backend(sched, t, gpu_backend); break; }
        }
    }
}

// ---- Persistent graph capture ----

void phase2_graph_cache::release() {
    valid = false;
    if (persistent_gf) { persistent_gf = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
}

// Shallow capture: duplicate only the cgraph structure using a private ggml context.
// The tensor objects are shared with the source graph, preserving framework-owned
// backend-specific state (extra, buffer, view offsets).
void capture_phase2_graph(phase2_graph_cache & cache, ggml_cgraph * src_gf) {
    cache.release();

    int n_nodes = ggml_graph_n_nodes(src_gf);
    int n_leafs = ggml_graph_n_leafs(src_gf);
    int gsize   = ggml_graph_size(src_gf);

    // Create a minimal context that owns only the cgraph and its hash sets.
    size_t mem_size = ggml_graph_overhead_custom(gsize, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    cache.ctx = ggml_init(params);
    if (!cache.ctx) {
        fprintf(stderr, "capture_phase2_graph: ggml_init failed\n");
        return;
    }

    cache.persistent_gf = ggml_new_graph_custom(cache.ctx, gsize, false);
    if (!cache.persistent_gf) {
        fprintf(stderr, "capture_phase2_graph: ggml_new_graph_custom failed\n");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return;
    }

    cache.persistent_gf->order = src_gf->order;
    cache.persistent_gf->uid   = src_gf->uid;

    for (int i = 0; i < n_leafs; i++) {
        cache.persistent_gf->leafs[cache.persistent_gf->n_leafs++] = ggml_graph_leaf(src_gf, i);
    }
    for (int i = 0; i < n_nodes; i++) {
        cache.persistent_gf->nodes[cache.persistent_gf->n_nodes++] = ggml_graph_node(src_gf, i);
    }

    cache.valid = true;
    fprintf(stderr, "capture_phase2_graph: %d nodes + %d leafs (shallow, size=%d)\n", n_nodes, n_leafs, gsize);
}

} // namespace moe
#endif
