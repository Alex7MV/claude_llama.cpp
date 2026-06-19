#ifdef GGML_USE_CUDA

#include "moe-hijacker.h"
#include "moe-static-bunker.h"

#include "ggml.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <unordered_map>
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
        if (prefix_len == pat_len && memcmp(name, h2.TENSOR_NAMES[t], pat_len) == 0) {
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
        ggml_tensor * t;
        if (i < total_leafs) { t = ggml_graph_leaf(gf, i); } else { t = ggml_graph_node(gf, i - total_leafs); }
        auto [type_id, il] = match_hijack_name(h2, t->name);
        if (type_id < 0 || type_id >= h2.N_TYPES) continue;
        if (il < 0 || il >= h2.n_layers) continue;
        if (il > max_il) max_il = il;

        size_t sz = ggml_type_size(t->type);
        for (int d = 0; d < 4; d++) sz *= t->ne[d] > 0 ? t->ne[d] : 1;

        int s_idx = h2.slot_idx(il, type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots) continue;

        if (h2.slots[s_idx].size == 0) {
            h2.slots[s_idx].size = H2_ALIGN_UP(sz);
            h2.slots[s_idx].addr = nullptr;
            h2.slots[s_idx].parent = -1;
            h2.slots[s_idx].offset = 0;
        }

        if (t->view_src) {
            ptrdiff_t off = (char*)t->data - (char*)t->view_src->data;
            h2.slots[s_idx].offset = off;
            h2.slots[s_idx].parent = -2;
            Match m = {t, type_id, il, t->data};
            views.push_back(m);
        } else {
            h2.slots[s_idx].parent = -1;
            Match m = {t, type_id, il, t->data};
            matches.push_back(m);
        }
    }

    int total_matched = (int)(matches.size() + views.size());
    if (total_matched == 0) {
        fprintf(stderr, "phase2_hijack: no tensors matched by name!\n");
        return 0;
    }

    if (!h2.buffer) {
        h2.allocate_slots(max_il + 1);
        if (!h2.buffer) return 0;
    }

    int skipped = 0;

    for (auto & m : matches) {
        int s_idx = h2.slot_idx(m.il, m.type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots || !h2.slots[s_idx].addr) { skipped++; continue; }
        h2.slots[s_idx].orig_data = m.orig;
        m.t->data = h2.slots[s_idx].addr;
        h2.slot_scan_count++;
    }

    for (auto & m : views) {
        if (!m.t->view_src) { skipped++; continue; }
        auto [pid, pl] = match_hijack_name(h2, m.t->view_src->name);
        if (pid < 0 || pid >= h2.N_TYPES) { skipped++; continue; }
        int p_idx = h2.slot_idx(pl, pid);
        if (p_idx < 0 || p_idx >= h2.n_slots || !h2.slots[p_idx].addr) { skipped++; continue; }
        int s_idx = h2.slot_idx(m.il, m.type_id);
        void * view_addr = (char*)h2.slots[p_idx].addr + h2.slots[s_idx].offset;
        if (s_idx >= 0 && s_idx < h2.n_slots) {
            h2.slots[s_idx].addr = view_addr;
            h2.slots[s_idx].orig_data = m.orig;
        }
        m.t->data = view_addr;
        h2.slot_scan_count++;
    }

    fprintf(stderr, "phase2_hijack: scan_and_hijack -> %d/%d matched, %d skipped (%d layers)\n",
            h2.slot_scan_count, (int)(matches.size() + views.size()), skipped, max_il + 1);
    return h2.slot_scan_count;
}

bool scan_and_update_snapshots(phase2_hijack & h2, ggml_cgraph * gf) {
    int total_nodes = ggml_graph_n_nodes(gf);
    int total_leafs = ggml_graph_n_leafs(gf);
    int total_t = total_leafs + total_nodes;
    int found = 0;

    for (int i = 0; i < total_t; i++) {
        ggml_tensor * t;
        if (i < total_leafs) { t = ggml_graph_leaf(gf, i); } else { t = ggml_graph_node(gf, i - total_leafs); }
        auto [type_id, il] = match_hijack_name(h2, t->name);
        if (type_id < 0 || type_id >= h2.N_TYPES) continue;
        if (il < 0 || il >= h2.n_layers) continue;

        int s_idx = h2.slot_idx(il, type_id);
        if (s_idx < 0 || s_idx >= h2.n_slots) continue;

        void * target = h2.slots[s_idx].addr;
        if (!target) continue;

        if (t->data != target) {
            h2.slots[s_idx].orig_data = t->data;
            t->data = target;
        }
        found++;
    }

    return found > 0;
}

void restore_all(phase2_hijack & h2) {
    (void)h2;
    asm volatile("" ::: "memory");
}

void copy_data_to_static(phase2_hijack & h2, void * cuda_stream) {
    for (int i = 0; i < h2.n_slots; i++) {
        if (h2.slots[i].addr && h2.slots[i].orig_data && h2.slots[i].size > 0) {
            cudaMemcpyAsync(h2.slots[i].addr, h2.slots[i].orig_data,
                h2.slots[i].size, cudaMemcpyDefault, (void*)cuda_stream);
        }
    }
}

void cascade_force_moe_consumers(
    phase2_hijack & h2,
    ggml_cgraph * gf,
    ggml_backend_sched_t sched,
    ggml_backend_t gpu_backend)
{
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op != 37 && t->op != 2) continue;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) {
            auto [sid, sil] = match_hijack_name(h2, t->src[s]->name);
            if (sid >= 0) {
                ggml_backend_sched_set_tensor_backend(sched, t, gpu_backend);
                break;
            }
        }
    }
}

// ---- Phase 2 Graph Deep-Copy ----

void phase2_graph_cache::release() {
    if (persistent_gf) { delete[] (char *)persistent_gf; persistent_gf = nullptr; }
    if (persistent_ctx) { ggml_free(persistent_ctx); persistent_ctx = nullptr; }
    persistent_tensors.clear();
    valid = false;
}

void deep_copy_phase2_graph(
    phase2_graph_cache & cache,
    ggml_cgraph * src_gf)
{
    cache.release();

    int n_nodes = ggml_graph_n_nodes(src_gf);
    int n_leafs = ggml_graph_n_leafs(src_gf);
    int total_t = n_nodes + n_leafs;

    const size_t tensor_bytes = (size_t)total_t * 1024;
    struct ggml_init_params params = { tensor_bytes, nullptr, true };
    cache.persistent_ctx = ggml_init(params);
    if (!cache.persistent_ctx) {
        fprintf(stderr, "deep_copy_phase2_graph: ggml_init(%zu) failed\n", tensor_bytes);
        return;
    }

    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;

    auto dup_tensor = [&](const ggml_tensor * src) -> ggml_tensor * {
        auto * dst = ggml_dup_tensor(cache.persistent_ctx, src);
        if (!dst) return nullptr;
        strncpy(dst->name, src->name, GGML_MAX_NAME - 1);
        dst->name[GGML_MAX_NAME - 1] = '\0';
        dst->op       = src->op;
        dst->data     = src->data;
        dst->flags    = src->flags;
        dst->view_offs = src->view_offs;
        dst->extra    = src->extra;
        memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
        for (int d = 0; d < GGML_MAX_DIMS; d++) dst->nb[d] = src->nb[d];
        map[src] = dst;
        cache.persistent_tensors.push_back(dst);
        return dst;
    };

    for (int i = 0; i < n_leafs; i++) dup_tensor(ggml_graph_leaf(src_gf, i));
    for (int i = 0; i < n_nodes; i++) dup_tensor(ggml_graph_node(src_gf, i));

    for (auto * dst : cache.persistent_tensors) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (dst->src[s]) {
                auto it = map.find(dst->src[s]);
                if (it != map.end()) dst->src[s] = it->second;
            }
        }
        if (dst->view_src) {
            auto it = map.find(dst->view_src);
            if (it != map.end()) dst->view_src = it->second;
        }
    }

    size_t graph_bytes = sizeof(ggml_cgraph) + (size_t)total_t * sizeof(ggml_tensor *) * 2;
    char * buf = new char[graph_bytes]();
    cache.persistent_gf = (ggml_cgraph *)buf;
    cache.persistent_gf->size    = total_t;
    cache.persistent_gf->n_nodes = 0;
    cache.persistent_gf->n_leafs = 0;
    cache.persistent_gf->nodes   = (ggml_tensor **)(buf + sizeof(ggml_cgraph));
    cache.persistent_gf->leafs   = cache.persistent_gf->nodes + total_t;
    cache.persistent_gf->grads     = nullptr;
    cache.persistent_gf->grad_accs = nullptr;
    cache.persistent_gf->use_counts = nullptr;
    cache.persistent_gf->order      = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;
    cache.persistent_gf->uid        = 0;

    for (int i = 0; i < n_leafs; i++) {
        auto it = map.find(ggml_graph_leaf(src_gf, i));
        cache.persistent_gf->leafs[cache.persistent_gf->n_leafs++] = it->second;
    }
    for (int i = 0; i < n_nodes; i++) {
        auto it = map.find(ggml_graph_node(src_gf, i));
        cache.persistent_gf->nodes[cache.persistent_gf->n_nodes++] = it->second;
    }

    cache.valid = true;
    fprintf(stderr, "deep_copy_phase2_graph: copied %d nodes + %d leafs (%zu KB ctx, %zu B graph)\n",
            n_nodes, n_leafs, tensor_bytes / 1024, graph_bytes);
}

} // namespace moe

#endif // GGML_USE_CUDA