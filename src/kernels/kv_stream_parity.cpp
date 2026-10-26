// src/kernels/kv_stream_parity.cpp - KV streaming (kv_stream.hpp) against a fully resident pool (GPU, no model).
//
// Decodes a synthetic sequence twice: into a fully resident pool with the identity page table, and into a streamed
// layer (host copy + a small VRAM slot pool that must evict constantly). Every few cells a batch of 1-8 queries
// with selections shaped like the indexer's (whole 4-cell blocks, a partial threshold block, the tail block, half of
// them recent) is attended in both; `kv_stream_resolve` runs before the streamed attention. Checks:
//   1. the attention outputs are BITWISE equal (the streamed readers see exactly the resident values);
//   2. the residency map is consistent after every call (slot_block and page_table invert each other);
//   3. no call overflowed, and the hit/miss counters add up;
//   4. a ring (the MTP drafter's layout) restored from the host copy reads the same values as the resident pool.
// INT8, FP16, Q4_0 (PR #21) and hybrid K8V4 pools; K8V4's appends are the engine's folded calls (layer.cpp), its host
// copy written through kv_hybrid_k_half / kv_hybrid_v_half.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
template <typename T> T* halloc(size_t n) {   // pinned, mapped; returns the device pointer
    void* h = nullptr;
    void* d = nullptr;
    ck(cudaHostAlloc(&h, n * sizeof(T) + 64, cudaHostAllocMapped), "hostalloc");
    std::memset(h, 0, n * sizeof(T) + 64);
    ck(cudaHostGetDevicePointer(&d, h, 0), "devptr");
    return (T*) d;
}

struct Pools {   // one K/V pool set of `pages` pages
    k::KvHostPools p;   // reused as a plain pointer bundle
    void alloc(int64_t pages, const k::QsaShapes& s, int fmt, bool host) {
        const size_t rows = (size_t) pages * s.n_head_kv * s.page_size;
        if (fmt == k::kKvHybrid) {
            const size_t b = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
            p.k_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.k_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
            p.v_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
        } else if (fmt == k::kKvQ4) {
            const size_t b = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
            p.k_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
            p.v_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
        } else if (fmt == k::kKvInt8) {
            p.k_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.v_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.k_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
            p.v_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
        } else {
            p.k_pool = host ? halloc<uint16_t>(rows * s.head_dim) : dalloc<uint16_t>(rows * s.head_dim);
            p.v_pool = host ? halloc<uint16_t>(rows * s.head_dim) : dalloc<uint16_t>(rows * s.head_dim);
        }
    }
    k::QsaAttnPools attn(const int32_t* table) const {
        k::QsaAttnPools a;
        a.k_pool = p.k_pool; a.v_pool = p.v_pool; a.k_q = p.k_q; a.v_q = p.v_q; a.k_scale = p.k_scale;
        a.v_scale = p.v_scale; a.k_q4 = p.k_q4; a.v_q4 = p.v_q4; a.page_table = table;
        return a;
    }
};

void append(const Pools& pl, const int32_t* table, const int32_t* step, const float* kc, const float* vc,
            const k::QsaShapes& s, int fmt, const k::KvHostPools* host) {
    if (fmt == k::kKvHybrid) {   // exactly as qsa_layer: each half folded onto its own pool, host copy by halves
        const k::KvHostPools hk = host ? k::kv_hybrid_k_half(*host) : k::KvHostPools{},
                             hv = host ? k::kv_hybrid_v_half(*host) : k::KvHostPools{};
        k::kv_append_q8_step(pl.p.k_q, pl.p.k_q, pl.p.k_scale, pl.p.k_scale, table, step, kc, kc, s, nullptr,
                             host ? &hk : nullptr);
        k::kv_append_q4_step(pl.p.v_q4, pl.p.v_q4, table, step, vc, vc, s, nullptr, host ? &hv : nullptr);
    } else if (fmt == k::kKvQ4) k::kv_append_q4_step(pl.p.k_q4, pl.p.v_q4, table, step, kc, vc, s, nullptr, host);
    else if (fmt == k::kKvInt8) k::kv_append_q8_step(pl.p.k_q, pl.p.v_q, pl.p.k_scale, pl.p.v_scale, table, step, kc, vc, s, nullptr, host);
    else k::kv_append_step(pl.p.k_pool, pl.p.v_pool, table, step, kc, vc, s, nullptr, host);
}

// A selection like qsa_block_topk's: ascending cells, whole blocks, the tail block's cells, and one partial block.
std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t c = 0; c < n_kv; ++c) ids.push_back((int32_t) c);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1), recent(std::max<int64_t>(0, n_bid - 2048), n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(rng() % 2 ? recent(rng) : any(rng));
    int64_t partial_block = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks) {
        const int64_t take = b == partial_block ? part : 4;
        for (int64_t i = 0; i < take; ++i) ids.push_back((int32_t) (b * 4 + i));
    }
    for (int64_t c = n_bid * 4; c < n_kv; ++c) ids.push_back((int32_t) c);
    std::sort(ids.begin(), ids.end());
    return ids;
}

bool run(int fmt) {
    const char* name = fmt == k::kKvHybrid ? "k8v4" : fmt == k::kKvQ4 ? "q4_0" : fmt == k::kKvInt8 ? "int8" : "fp16";
    k::QsaShapes s = k::qsa_real_shapes();
    const int64_t N = 40000, n_blocks = (N + 3) / 4, n_slots = 8 * 516 + 700;   // must evict: slots < blocks
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s), NQ = 8;
    const int64_t H = s.n_head_kv, D = s.head_dim, NH = s.n_head;
    std::mt19937 rng(7 + 4 * fmt);
    std::normal_distribution<float> nd(0.f, 1.f);

    Pools ref, slots, host;
    ref.alloc(n_blocks, s, fmt, false);
    slots.alloc(n_slots, s, fmt, false);
    host.alloc(n_blocks, s, fmt, true);
    int32_t* ident = dalloc<int32_t>(n_blocks);
    {
        std::vector<int32_t> t(n_blocks);
        for (int64_t i = 0; i < n_blocks; ++i) t[i] = (int32_t) i;
        ck(cudaMemcpy(ident, t.data(), n_blocks * 4, cudaMemcpyHostToDevice), "ident");
    }
    k::KvStreamMap m;
    m.page_table = dalloc<int32_t>(n_blocks);
    m.slot_block = dalloc<int32_t>(n_slots); m.slot_stamp = dalloc<int32_t>(n_slots); m.slot_ref = dalloc<int32_t>(n_slots);
    m.miss_block = dalloc<int32_t>(n_slots); m.miss_slot = dalloc<int32_t>(n_slots);
    m.ctl = dalloc<int32_t>(k::kKvCtlInts);
    m.n_blocks = n_blocks; m.n_slots = n_slots;
    k::kv_stream_reset(m, nullptr);

    float* kc = dalloc<float>(H * D);
    float* vc = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    int32_t* steps = dalloc<int32_t>(NQ * k::kStepCount);
    int32_t* ids = dalloc<int32_t>(NQ * cap);
    float* q = dalloc<float>(NQ * NH * D);
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>(NQ * scr);
    float* out_ref = dalloc<float>(NQ * NH * D);
    float* out_str = dalloc<float>(NQ * NH * D);
    std::vector<float> hk(H * D), hv(H * D), hq(NQ * NH * D), a((size_t) NQ * NH * D), b2((size_t) NQ * NH * D);
    std::vector<int32_t> pt(n_blocks), sb(n_slots);

    int batches = 0, bad = 0;
    for (int64_t pos = 0; pos < N; ++pos) {
        for (auto& x : hk) x = nd(rng) * ((pos % 17 == 0) ? 30.f : 1.f);
        for (auto& x : hv) x = nd(rng);
        const int32_t st[4] = {(int32_t) pos, (int32_t) (pos + 1), (int32_t) ((pos + 1) / 4), 0};
        ck(cudaMemcpy(kc, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vc, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
        ck(cudaMemcpy(step, st, sizeof(st), cudaMemcpyHostToDevice), "step");
        append(ref, ident, step, kc, vc, s, fmt, nullptr);
        append(slots, m.page_table, step, kc, vc, s, fmt, &host.p);
        if (pos % 131 != 130 && pos != N - 1) continue;
        // a batch: n_q queries at the last n_q positions (as a verify window), each with its own selection
        const int n_q = 1 + (int) (rng() % NQ);
        std::vector<int32_t> hids((size_t) NQ * cap, 0), hst(NQ * 4, 0);
        for (int t = 0; t < n_q; ++t) {
            const int64_t p = std::max<int64_t>(0, pos - (n_q - 1 - t)), n_kv = p + 1;
            const int64_t width = std::min<int64_t>(n_kv, cap);
            const std::vector<int32_t> sel = selection(n_kv, width, rng);
            if ((int64_t) sel.size() != width) { std::fprintf(stderr, "selection size %zu != %lld\n", sel.size(), (long long) width); return false; }
            std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) t * cap);
            hst[t * 4 + 0] = (int32_t) p; hst[t * 4 + 1] = (int32_t) n_kv; hst[t * 4 + 2] = (int32_t) (n_kv / 4);
            hst[t * 4 + 3] = (int32_t) width;
        }
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, n_q, nullptr);
        k::kv_stream_resolve(m, slots.attn(m.page_table), host.p, fmt, ids, steps, n_q, cap, s, nullptr);
        k::qsa_decode_attn_batch(q, slots.attn(m.page_table), ids, steps, cap, s, scratch, out_str, n_q, nullptr);
        ck(cudaDeviceSynchronize(), "batch");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "b");
        if (std::memcmp(a.data(), b2.data(), (size_t) n_q * NH * D * 4) != 0) {
            if (bad++ < 5) std::fprintf(stderr, "  %s pos %lld n_q %d: streamed attention differs\n", name, (long long) pos, n_q);
        }
        // the map inverts itself
        ck(cudaMemcpy(pt.data(), m.page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "pt");
        ck(cudaMemcpy(sb.data(), m.slot_block, n_slots * 4, cudaMemcpyDeviceToHost), "sb");
        int64_t resident = 0;
        for (int64_t bk = 0; bk < n_blocks; ++bk) {
            if (pt[bk] < -1 || pt[bk] >= n_slots || (pt[bk] >= 0 && sb[pt[bk]] != bk)) {
                if (bad++ < 5) std::fprintf(stderr, "  map broken at block %lld (table %d)\n", (long long) bk, pt[bk]);
                break;
            }
            resident += pt[bk] >= 0;
        }
        for (int64_t sl = 0; sl < n_slots; ++sl)
            if (sb[sl] >= 0 && pt[sb[sl]] != sl) { if (bad++ < 5) std::fprintf(stderr, "  slot %lld not in the table\n", (long long) sl); break; }
        ++batches;
    }
    const k::KvStreamCounters c = k::kv_stream_counters(m);
    std::printf("  %s: %d batches, %llu block lookups, %llu misses (%.1f%% hit), overflow %d, %d failures\n",
                name, batches, (unsigned long long) c.lookups, (unsigned long long) c.misses,
                c.lookups ? 100.0 * (double) (c.lookups - c.misses) / (double) c.lookups : 0.0, (int) c.overflow, bad);
    if (c.overflow || c.calls != (uint64_t) batches || c.misses == 0 || c.misses >= c.lookups) ++bad;

    // 4. a ring over the same host copy: blocks [b1 - R, b1) restored, read through `b % R`
    {
        const int64_t R = 1500, b1 = n_blocks, b0 = b1 - R;
        Pools ring;
        ring.alloc(R, s, fmt, false);
        int32_t* rt = dalloc<int32_t>(n_blocks);
        k::kv_ring_table(rt, n_blocks, R, nullptr);
        k::kv_ring_restore(ring.attn(rt), host.p, fmt, b0, b1, R, s, nullptr);
        const int64_t width = cap;
        std::vector<int32_t> hids((size_t) cap), hst = {(int32_t) (N - 1), (int32_t) N, (int32_t) (N / 4), (int32_t) width};
        for (int64_t i = 0; i < width; ++i) hids[i] = (int32_t) (N - width + i);   // the window's last cells
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), 16, cudaMemcpyHostToDevice), "steps");
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, 1, nullptr);
        k::qsa_decode_attn_batch(q, ring.attn(rt), ids, steps, cap, s, scratch, out_str, 1, nullptr);
        ck(cudaDeviceSynchronize(), "ring");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "b");
        const bool ok = std::memcmp(a.data(), b2.data(), (size_t) NH * D * 4) == 0;
        std::printf("  %s ring restore: %s\n", name, ok ? "identical" : "DIFFERS");
        if (!ok) ++bad;
    }
    return bad == 0;
}

// The main session's graph must follow the device descriptor on EVERY replay. Deliberately leave the
// captured CPU bundle pointing at A, then replay A/B/A/B/B/A with different contents and append positions.
// This exercises the actual FP16 append and resolve kernels, and checks the inactive owner's whole pool.
bool run_host_borrow() {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t blocks = 2048, slots_n = 1024, cap = 8;
    const size_t values = (size_t) blocks * s.n_head_kv * s.page_size * s.head_dim;
    const size_t slot_values = (size_t) slots_n * s.n_head_kv * s.page_size * s.head_dim;
    const size_t page_values = (size_t) s.n_head_kv * s.page_size * s.head_dim;
    std::vector<void*> device_allocations, host_allocations;
    auto device = [&](size_t bytes) {
        void* p = nullptr;
        ck(cudaMalloc(&p, bytes), "borrow malloc");
        device_allocations.push_back(p);
        ck(cudaMemset(p, 0, bytes), "borrow memset");
        return p;
    };
    auto mapped = [&](size_t bytes) {
        void* h = nullptr;
        void* d = nullptr;
        ck(cudaHostAlloc(&h, bytes, cudaHostAllocMapped), "borrow hostalloc");
        host_allocations.push_back(h);   // free the HOST allocation, never its device alias
        ck(cudaHostGetDevicePointer(&d, h, 0), "borrow mapped pointer");
        return static_cast<uint16_t*>(d);
    };
    k::KvHostPools owners[2];
    std::vector<uint16_t> expected[2][2];
    for (int owner = 0; owner < 2; ++owner) {
        owners[owner].k_pool = mapped(values * sizeof(uint16_t));
        owners[owner].v_pool = mapped(values * sizeof(uint16_t));
        for (int role = 0; role < 2; ++role) {
            auto& e = expected[owner][role];
            e.resize(values);
            for (size_t i = 0; i < values; ++i)
                e[i] = (uint16_t) ((owner ? 0xb800 : 0x3400) + role * 0x100 + i % 127);
            ck(cudaMemcpy(role ? owners[owner].v_pool : owners[owner].k_pool, e.data(),
                          values * sizeof(uint16_t), cudaMemcpyDefault), "borrow poison owner");
        }
    }
    k::KvHostPools* descriptor = static_cast<k::KvHostPools*>(device(sizeof(k::KvHostPools)));
    k::KvHostPools view = owners[0];
    view.descriptor = descriptor;
    Pools slots, ref;
    slots.p.k_pool = static_cast<uint16_t*>(device(slot_values * sizeof(uint16_t)));
    slots.p.v_pool = static_cast<uint16_t*>(device(slot_values * sizeof(uint16_t)));
    ref.p.k_pool = static_cast<uint16_t*>(device(values * sizeof(uint16_t)));
    ref.p.v_pool = static_cast<uint16_t*>(device(values * sizeof(uint16_t)));
    auto ints = [&](int64_t n) { return static_cast<int32_t*>(device((size_t) n * sizeof(int32_t))); };
    k::KvStreamMap m;
    m.n_blocks = blocks; m.n_slots = slots_n;
    m.page_table = ints(blocks); m.slot_block = ints(slots_n); m.slot_stamp = ints(slots_n);
    m.slot_ref = ints(slots_n); m.miss_block = ints(slots_n); m.miss_slot = ints(slots_n);
    m.ctl = ints(k::kKvCtlInts);
    int32_t* ident = ints(blocks);
    int32_t* step = ints(k::kStepCount);
    int32_t* ids = ints(cap);
    std::vector<int32_t> table((size_t) blocks);
    for (int64_t b = 0; b < blocks; ++b) table[(size_t) b] = (int32_t) b;
    ck(cudaMemcpy(ident, table.data(), table.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "borrow identity");
    float* kc = static_cast<float*>(device((size_t) s.n_head_kv * s.head_dim * sizeof(float)));
    float* vc = static_cast<float*>(device((size_t) s.n_head_kv * s.head_dim * sizeof(float)));
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "borrow stream");
    ck(cudaDeviceSynchronize(), "borrow before capture");
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "borrow capture begin");
    k::kv_stream_reset(m, stream);
    k::kv_append_step(slots.p.k_pool, slots.p.v_pool, m.page_table, step, kc, vc, s, stream, &view);
    k::kv_stream_resolve(m, slots.attn(m.page_table), view, k::kKvF16, ids, step, 1, cap, s, stream);
    ck(cudaStreamEndCapture(stream, &graph), "borrow capture end");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "borrow instantiate");

    int bad = 0, turn = 0;
    std::vector<float> hkc((size_t) s.n_head_kv * s.head_dim), hvc(hkc.size());
    std::vector<uint16_t> actual(values), page(page_values);
    for (const int owner : {0, 1, 0, 1, 1, 0}) {
        const int block = 100 + 37 * turn;
        const int pos = block * (int) s.page_size + (int) s.page_size - 1;
        const int32_t st[k::kStepCount] = {pos, pos + 1, (pos + 1) / 4, (int32_t) cap};
        int32_t selected[8];
        for (int i = 0; i < 4; ++i) { selected[i] = i; selected[i + 4] = block * 4 + i; }
        for (size_t i = 0; i < hkc.size(); ++i) {
            hkc[i] = (float) (turn + 1) + (float) (i % 31) / 32.f;
            hvc[i] = -hkc[i];
        }
        ck(cudaMemcpy(step, st, sizeof st, cudaMemcpyHostToDevice), "borrow step");
        ck(cudaMemcpy(ids, selected, sizeof selected, cudaMemcpyHostToDevice), "borrow selection");
        ck(cudaMemcpy(kc, hkc.data(), hkc.size() * sizeof(float), cudaMemcpyHostToDevice), "borrow K input");
        ck(cudaMemcpy(vc, hvc.data(), hvc.size() * sizeof(float), cudaMemcpyHostToDevice), "borrow V input");
        ck(cudaMemcpy(ref.p.k_pool, expected[owner][0].data(), values * sizeof(uint16_t), cudaMemcpyHostToDevice), "borrow reference K");
        ck(cudaMemcpy(ref.p.v_pool, expected[owner][1].data(), values * sizeof(uint16_t), cudaMemcpyHostToDevice), "borrow reference V");
        // Exact original append kernel is the arithmetic reference; no hand-written FP16 conversion oracle.
        k::kv_append_step(ref.p.k_pool, ref.p.v_pool, ident, step, kc, vc, s, stream, nullptr);
        ck(cudaStreamSynchronize(stream), "borrow reference append");
        ck(cudaMemcpy(expected[owner][0].data(), ref.p.k_pool, values * sizeof(uint16_t), cudaMemcpyDeviceToHost), "borrow expected K");
        ck(cudaMemcpy(expected[owner][1].data(), ref.p.v_pool, values * sizeof(uint16_t), cudaMemcpyDeviceToHost), "borrow expected V");
        ck(cudaMemcpy(descriptor, &owners[owner], sizeof(k::KvHostPools), cudaMemcpyHostToDevice), "borrow switch descriptor");
        ck(cudaGraphLaunch(exec, stream), "borrow graph replay");
        ck(cudaStreamSynchronize(stream), "borrow graph complete");
        for (int check_owner = 0; check_owner < 2; ++check_owner)
            for (int role = 0; role < 2; ++role) {
                ck(cudaMemcpy(actual.data(), role ? owners[check_owner].v_pool : owners[check_owner].k_pool,
                              values * sizeof(uint16_t), cudaMemcpyDefault), "borrow check owner");
                if (actual != expected[check_owner][role]) ++bad;
            }
        ck(cudaMemcpy(table.data(), m.page_table, table.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "borrow table");
        for (const int b : {0, block}) {
            const int sl = table[(size_t) b];
            if (sl < 0 || sl >= slots_n) { ++bad; continue; }
            for (int role = 0; role < 2; ++role) {
                const uint16_t* source = (role ? slots.p.v_pool : slots.p.k_pool) + (size_t) sl * page_values;
                ck(cudaMemcpy(page.data(), source, page_values * sizeof(uint16_t), cudaMemcpyDeviceToHost), "borrow resolved page");
                if (!std::equal(page.begin(), page.end(), expected[owner][role].begin() + (size_t) b * page_values)) ++bad;
            }
        }
        const auto counters = k::kv_stream_counters(m);
        if (counters.overflow || counters.calls != 1 || counters.misses != 2) ++bad;
        ++turn;
    }
    ck(cudaGraphExecDestroy(exec), "borrow destroy executable");
    ck(cudaGraphDestroy(graph), "borrow destroy graph");
    ck(cudaStreamDestroy(stream), "borrow destroy stream");
    for (void* p : device_allocations) ck(cudaFree(p), "borrow free device");
    for (void* p : host_allocations) ck(cudaFreeHost(p), "borrow free owner");
    std::printf("  fp16 borrowed host graph: %d owner replays, %d byte/ownership failures\n", turn, bad);
    return bad == 0;
}
}  // namespace

int main(int argc, char** argv) {
    const bool borrow_only = argc == 2 && std::strcmp(argv[1], "--host-borrow-only") == 0;
    if (argc != 1 && !borrow_only) {
        std::fprintf(stderr, "usage: kv_stream_parity [--host-borrow-only]\n");
        return 2;
    }
    std::printf("kv_stream_parity: streamed vs resident KV, bitwise\n");
    if (!run_host_borrow()) ++g_fail;
    if (!borrow_only) {
        const bool a = run(k::kKvInt8), b = run(k::kKvF16), c = run(k::kKvQ4), d = run(k::kKvHybrid);
        if (!a || !b || !c || !d) ++g_fail;
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
