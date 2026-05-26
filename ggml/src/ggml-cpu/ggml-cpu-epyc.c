#include "ggml-cpu-epyc-impl.h"
#include "ggml-cpu-impl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#if defined(__gnu_linux__)

#define GGML_EPYC_SYSFS_PATH "/sys/devices/system/cpu"

// Internal CCD topology (cached after first probe)
static struct ggml_ccd_topology g_epyc_topo;
static bool g_epyc_probed = false;

static int ggml_epyc_read_sysfs(const char *path, char *buf, size_t buf_size) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, (int)buf_size, f)) { fclose(f); return -1; }
    fclose(f);
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
    return 0;
}

static void ggml_epyc_probe_topology(void) {
    if (g_epyc_probed) return;
    memset(&g_epyc_topo, 0, sizeof(g_epyc_topo));
    char path[512];
    char buf[512];
    int cpu = 0;

    // Determine CCD file: Linux >= 6.5 uses "die_id", older kernels use "core_defaults".
    // Probe cpu0 first to pick the correct file name.
    const char *ccd_file = NULL;
    snprintf(path, sizeof(path), "%s/cpu0/topology/die_id", GGML_EPYC_SYSFS_PATH);
    if (ggml_epyc_read_sysfs(path, buf, sizeof(buf)) == 0) {
        ccd_file = "die_id";
    } else {
        snprintf(path, sizeof(path), "%s/cpu0/topology/core_defaults", GGML_EPYC_SYSFS_PATH);
        if (ggml_epyc_read_sysfs(path, buf, sizeof(buf)) == 0) {
            ccd_file = "core_defaults";
        }
    }

    if (!ccd_file) {
        fprintf(stderr, "WARN: CCD: core_defaults not available, CCD affinity disabled\n");
        g_epyc_probed = true;
        return;
    }

    while (1) {
        int n = snprintf(path, sizeof(path), "%s/cpu%u/topology/%s", GGML_EPYC_SYSFS_PATH, cpu, ccd_file);
        if (n < 0 || (size_t)n >= sizeof(path)) break;
        if (ggml_epyc_read_sysfs(path, buf, sizeof(buf)) != 0) break;
        uint32_t ccd_id = 0;
        if (sscanf(buf, "%u", &ccd_id) != 1) break;
        g_epyc_topo.ccd_for_cpu[cpu] = ccd_id;
        g_epyc_topo.ccd_threads[g_epyc_topo.total_threads++] = cpu;
        if (ccd_id >= g_epyc_topo.n_ccds) g_epyc_topo.n_ccds = ccd_id + 1;
        cpu++;
    }

    // Identify SMT siblings
    bool is_sibling[GGML_NUMA_MAX_CPUS] = {0};
    for (int c = 0; (uint32_t)c < g_epyc_topo.total_threads; c++) {
        int n = snprintf(path, sizeof(path), "%s/cpu%u/topology/thread_siblings_list", GGML_EPYC_SYSFS_PATH, c);
        if (n < 0 || (size_t)n >= sizeof(path)) continue;
        if (ggml_epyc_read_sysfs(path, buf, sizeof(buf)) != 0) continue;
        char *saveptr = NULL;
        char *token = strtok_r(buf, ",", &saveptr);
        int first = -1;
        while (token) {
            int sibling = atoi(token);
            if (first < 0) first = sibling;
            else if (sibling != c) is_sibling[sibling] = true;
            token = strtok_r(NULL, ",", &saveptr);
        }
    }

    // Build ordered: primaries first, then siblings
    uint32_t ordered[GGML_NUMA_MAX_CPUS];
    uint32_t oc = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (uint32_t c = 0; c < g_epyc_topo.total_threads; c++) {
            if (is_sibling[c] && pass == 0) continue;
            if (!is_sibling[c] && pass == 1) continue;
            ordered[oc++] = g_epyc_topo.ccd_threads[c];
        }
    }
    memcpy(g_epyc_topo.ccd_threads, ordered, oc * sizeof(uint32_t));

    // Per-CCD thread counts
    for (uint32_t i = 0; i < oc; i++) {
        uint32_t ccd = g_epyc_topo.ccd_for_cpu[g_epyc_topo.ccd_threads[i]];
        g_epyc_topo.ccd_thread_count[ccd]++;
    }
    g_epyc_probed = true;
}

int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs) {
    ggml_epyc_probe_topology();
    if (!pairs || max_pairs < 1 || g_epyc_topo.n_ccds < 2) return 0;

    int pc = 0;
    // Pair 0: first CCD + last CCD (NUMA 0 + NUMA 1)
    pairs[0].ccd_indices[0] = 0;
    pairs[0].ccd_indices[1] = (int)g_epyc_topo.n_ccds - 1;
    pairs[0].thread_count = 0;
    for (uint32_t i = 0; i < g_epyc_topo.total_threads; i++) {
        uint32_t cpu = g_epyc_topo.ccd_threads[i];
        uint32_t ccd = g_epyc_topo.ccd_for_cpu[cpu];
        if (ccd == 0 || ccd == g_epyc_topo.n_ccds - 1) {
            pairs[0].threads[pairs[0].thread_count++] = cpu;
        }
    }
    pc++;

    // Pair 1: second + second-to-last CCD (if >= 4 CCDs)
    if (g_epyc_topo.n_ccds >= 4 && max_pairs >= 2) {
        pairs[1].ccd_indices[0] = 1;
        pairs[1].ccd_indices[1] = (int)g_epyc_topo.n_ccds - 2;
        pairs[1].thread_count = 0;
        for (uint32_t i = 0; i < g_epyc_topo.total_threads; i++) {
            uint32_t cpu = g_epyc_topo.ccd_threads[i];
            uint32_t ccd = g_epyc_topo.ccd_for_cpu[cpu];
            if (ccd == 1 || ccd == g_epyc_topo.n_ccds - 2) {
                pairs[1].threads[pairs[1].thread_count++] = cpu;
            }
        }
        pc++;
    }
    return pc;
}

int ggml_cpu_init_dual_threadpool(struct ggml_threadpool_params params_out[],
                                   int count, int threads_per_pair,
                                   enum ggml_sched_priority prio, uint32_t poll) {
    if (threads_per_pair < 1) return 0;
    struct ggml_cpu_ccd_pair pairs[GGML_EPYC_MAX_CCD_PAIRS];
    int np = ggml_cpu_probe_ccd_pairs(pairs, GGML_EPYC_MAX_CCD_PAIRS);
    if (np < 2 || count < 2 || !params_out) return 0;

    for (int p = 0; p < 2; p++) {
        memset(&params_out[p], 0, sizeof(struct ggml_threadpool_params));
        int nt = threads_per_pair < (int)pairs[p].thread_count ? threads_per_pair : (int)pairs[p].thread_count;
        params_out[p].n_threads = nt;
        params_out[p].prio = prio;
        params_out[p].poll = poll;
        params_out[p].strict_cpu = true;
        params_out[p].paused = false;
        for (int t = 0; t < nt && t < (int)pairs[p].thread_count; t++) {
            uint32_t cpu = pairs[p].threads[t];
            if (cpu < GGML_MAX_N_THREADS) {
                params_out[p].cpumask[cpu] = true;
            }
        }
    }
    return 2;
}

#else // non-Linux stubs

int ggml_cpu_probe_ccd_pairs(struct ggml_cpu_ccd_pair pairs[], int max_pairs) {
    (void)pairs; (void)max_pairs;
    return 0;
}

int ggml_cpu_init_dual_threadpool(struct ggml_threadpool_params params_out[],
                                   int count, int threads_per_pair,
                                   enum ggml_sched_priority prio, uint32_t poll) {
    (void)params_out; (void)count; (void)threads_per_pair; (void)prio; (void)poll;
    return 0;
}

#endif
