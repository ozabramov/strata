// src/kernels/cpu/expert.cpp - P2.S3: the Q2_0 expert kernel, promoted from bench/micro/cpu_s2.cpp.
//
// The body is the validated kernel, moved rather than rewritten: it carries the P0.T2 parity result
// (rel 1.461e-06 against the ggml formula) and the three performance findings recorded in
// include/strata/kernels/cpu/expert.hpp.  Read that header first; it says why each piece is shaped this way.
#include "strata/kernels/cpu/expert.hpp"

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cmath>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// h2f, the Q8_0 oracle quantizer, the 64-code unpack (VPMULTISHIFTQB here), the AVX-512 activation quantizer and the
// GGUF-layout Q2_0 rows: shared with q2_avx512bw.cpp, which compiles them without VBMI.
#define STRATA_Q2_UNPACK_VBMI 1
#include "q2_avx512_kernels.inl"

/// 2-bit unpack via VPMULTISHIFTQB.
///
/// ARGUMENT ORDER IS (control, data) AND WAS DETERMINED EMPIRICALLY (`bench/micro/probe_multishift.cpp`).
/// The Intel guide states (a=data, b=control); using that produced 43 of 64 wrong codes.  `plan.md`'s sketch
/// passes the shift pattern first, which is correct - so a "fix" that swapped it to match the guide was the
/// bug, not the fix.  Within each 64-bit lane, output byte i is `(u16 >> 2i) & 0xFF`, so AND 3 yields the
/// eight 2-bit codes in order.
///
/// The result is the 32 codes of ONE ACTIVATION CHUNK (4 qword lanes), which is the unit the Q8_1 scale is
/// defined on - not the 64-weight block.
inline __m256i unpack_q2_0(const uint8_t* codes) {
    const __m128i packed = _mm_loadl_epi64((const __m128i*) codes);   // 8 bytes -> 4 u16 in the low half
    const __m256i lanes = _mm256_cvtepu16_epi64(packed);              // 4 qwords, 8 codes each
    const __m256i ctrl = _mm256_set1_epi64x((long long) 0x0E0C0A0806040200ULL);
    return _mm256_and_si256(_mm256_multishift_epi64_epi8(ctrl, lanes), _mm256_set1_epi8(3));
}

/// 32 codes against 32 int8 activations. Codes 0 contribute nothing to `sum(c*xhat)`, so the zero lanes the
/// narrow load leaves behind are harmless.
inline __m256i block_dot(const uint8_t* codes, const int8_t* xq) {
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), unpack_q2_0(codes),
                               _mm256_load_si256((const __m256i*) xq));
}

inline float hsum_ps(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    return _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(s, s), s));
}

// ================================ plan v0.3 P6: the row dot in 512-bit lanes ================================
//
// The 256-bit row dot below spent ~12 cycles per 64-weight block: two unpacks, two VNNI dots, two converts, two
// scalar scale products broadcast into two FMAs, an fp16 conversion and a scalar correction - compute-bound at
// ~39 GB/s while the machine streams 52.  Here a block is ONE unpack (16 code bytes -> 64 codes), ONE VNNI dot
// over 64 activations, ONE convert and ONE FMA whose per-chunk scales come from a permute of a vector of scale
// products built for eight blocks at a time; the weight-independent correction is one FMA per eight blocks.
// The sum order differs from the 256-bit kernel (last-bit differences); the verify window and plain decode both
// use this kernel, so speculative decode still reproduces plain decode exactly.
static const bool kZmm = std::getenv("STRATA_CPU_YMM") == nullptr;


// For blocks [b0, b0+nb) (nb <= 8): p[2i+h] = d_{b0+i} * xscale[2(b0+i)+h], dd[2i+h] = d_{b0+i}.
inline void scales8(const uint8_t* wscales, const ActQ& a, int b0, int nb, __m512& p, __m512& dd) {
    const __mmask8 m8 = nb >= 8 ? (__mmask8) 0xFF : (__mmask8) ((1u << nb) - 1u);
    const __m128i h = _mm_maskz_loadu_epi16(m8, wscales + 2 * b0);
    const __m512 d8 = _mm512_castps256_ps512(_mm256_cvtph_ps(h));
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    dd = _mm512_permutexvar_ps(dup, d8);
    const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
    dd = _mm512_maskz_mov_ps(m16, dd);
    p = _mm512_mul_ps(dd, _mm512_maskz_loadu_ps(m16, a.scale + 2 * b0));
}

inline float row_dot_z(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    __m512 acc = _mm512_setzero_ps(), corr = _mm512_setzero_ps();
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        __m512 p, dd;
        scales8(scales, a, b0, nb, p, dd);
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        corr = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a.hx + 2 * b0), corr);
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), unpack64_q2_0(codes + b * 16),
                                                    _mm512_load_si512((const void*) (a.q + b * QK)));
            const __m512 sv = _mm512_permutexvar_ps(_mm512_add_epi32(base, _mm512_set1_epi32(2 * i)), p);
            acc = _mm512_fmadd_ps(sv, _mm512_cvtepi32_ps(dot), acc);
        }
    }
    return _mm512_reduce_add_ps(acc) - _mm512_reduce_add_ps(corr);
}

template<int NT>
inline void row_dot_multi_z(const uint8_t* codes, const uint8_t* scales, const ActQ* const* a, int nblocks,
                            float* res) {
    __m512 acc[NT], corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm512_setzero_ps(); corr[t] = _mm512_setzero_ps(); }
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        __m512 p[NT];
        for (int t = 0; t < NT; ++t) {
            __m512 dd;
            scales8(scales, *a[t], b0, nb, p[t], dd);
            corr[t] = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->hx + 2 * b0), corr[t]);
        }
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i w = unpack64_q2_0(codes + b * 16);
            const __m512i idx = _mm512_add_epi32(base, _mm512_set1_epi32(2 * i));
            for (int t = 0; t < NT; ++t) {
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w,
                                                        _mm512_load_si512((const void*) (a[t]->q + b * QK)));
                acc[t] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx, p[t]), _mm512_cvtepi32_ps(dot), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]) - _mm512_reduce_add_ps(corr[t]);
}

/// One weight row against a quantized activation.
///
///     sum_j (c_j - 1) d_w xhat_j d_x  ==  d_w * ( d_x*sum(c*xhat) - d_x*sum(xhat) )
///
/// split per 32-chunk because `d_x` changes at the chunk boundary while `d_w` does not.
///
/// The accumulator stays a FLOAT VECTOR for the whole row and is reduced once at the end.  Reducing per
/// 64-weight block (two 8-lane horizontal sums: a store plus 16 scalar adds) cost far more than the dot
/// products themselves - it made the kernel compute-bound at 21 GB/s instead of DRAM-bound at 32.
inline float row_dot(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    if (kZmm) return row_dot_z(codes, scales, a, nblocks);
    __m256 acc = _mm256_setzero_ps();
    // The -sum(xhat) term is a SCALAR: broadcasting it into all 8 lanes and letting hsum_ps add them would
    // apply it eight times per block.
    float corr = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f(scales + 2 * b);
        const __m256 slo = _mm256_cvtepi32_ps(block_dot(codes + b * 16, a.q + b * QK));
        const __m256 shi = _mm256_cvtepi32_ps(block_dot(codes + b * 16 + 8, a.q + b * QK + QKA));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * a.scale[2 * b]), slo, acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * a.scale[2 * b + 1]), shi, acc);
        corr += d * (a.hx[2 * b] + a.hx[2 * b + 1]);
    }
    return hsum_ps(acc) - corr;
}

// VNNI's integer partial sums are exact. Remove the Q2_0 code offset BEFORE
// conversion to float, then follow the pinned generic CPU dot's two-chunk and
// per-64-block reduction. Its x86 type trait selects this generic reduction even
// in the AVX-512 oracle build. The legacy eight-lane FP32 reduction above has a
// different rounding order and remains the default.
inline int chunk_dot_oracle(const uint8_t* codes, const int8_t* q, int sum) {
    const __m256i lanes = block_dot(codes, q);
    const __m128i halves = _mm_add_epi32(_mm256_castsi256_si128(lanes), _mm256_extracti128_si256(lanes, 1));
    const __m128i pairs = _mm_hadd_epi32(halves, halves);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(pairs, pairs)) - sum;
}

inline float row_dot_oracle(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    float result = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        float block = 0.0f;
        for (int half = 0; half < 2; ++half) {
            const int chunk = 2 * b + half;
            const int dot = chunk_dot_oracle(codes + b * 16 + half * 8, a.q + chunk * QKA, a.sum[chunk]);
            block += a.scale[chunk] * float(dot);
        }
        result += h2f(scales + 2 * b) * block;
    }
    return result;
}

/// `row_dot` for NT activations at once: the codes of each block are unpacked once, and each token's
/// accumulator sees exactly the operations `row_dot` would apply to it, in the same order.
template<int NT>
inline void row_dot_multi(const uint8_t* codes, const uint8_t* scales, const ActQ* const* a, int nblocks,
                          float* res) {
    if (kZmm) { row_dot_multi_z<NT>(codes, scales, a, nblocks, res); return; }
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f(scales + 2 * b);
        const __m256i lo = unpack_q2_0(codes + b * 16);
        const __m256i hi = unpack_q2_0(codes + b * 16 + 8);
        for (int t = 0; t < NT; ++t) {
            const __m256 slo = _mm256_cvtepi32_ps(
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), lo, _mm256_load_si256((const __m256i*) (a[t]->q + b * QK))));
            const __m256 shi = _mm256_cvtepi32_ps(
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), hi,
                                    _mm256_load_si256((const __m256i*) (a[t]->q + b * QK + QKA))));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b]), slo, acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b + 1]), shi, acc[t]);
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_ps(acc[t]) - corr[t];
}

template<int NT>
void expert_multi(const uint8_t* blob, const ActQ* const* a1, float* const* out, ExpertScratchMulti& ws) {
    float g[NT], u[NT];
    for (int r = 0; r < FF; ++r) {
        row_dot_multi<NT>(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                          blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU, g);
        row_dot_multi<NT>(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                          blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU, u);
        for (int t = 0; t < NT; ++t) ws.ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
    const ActQ* a2[NT];
    for (int t = 0; t < NT; ++t) {
        act_quant_q8_1(ws.ff[t], FF, ws.a2[t]);
        a2[t] = &ws.a2[t];
    }
    float o[NT];
    for (int r = 0; r < H; ++r) {
        row_dot_multi<NT>(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2, a2,
                          SC_D, o);
        for (int t = 0; t < NT; ++t) out[t][r] = o[t];
    }
}

void expert_oracle_q8_0(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    for (int r = 0; r < FF; ++r) {
        const float g = row_dot_oracle(blob + O_GU_CODES + size_t(2 * r) * ROW_GU,
                                      blob + O_GU_SCALES + size_t(2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot_oracle(blob + O_GU_CODES + size_t(2 * r + 1) * ROW_GU,
                                      blob + O_GU_SCALES + size_t(2 * r + 1) * SC_GU * 2, a1, SC_GU);
        ws.ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
    quantize_oracle_q8_0(ws.ff, FF, ws.a2);
    for (int r = 0; r < H; ++r)
        out[r] = row_dot_oracle(blob + O_D_CODES + size_t(r) * ROW_D,
                               blob + O_D_SCALES + size_t(r) * SC_D * 2, ws.a2, SC_D);
}

}  // namespace

// CpuFeatures::reason, cpu_features, cpu_require_expert_support and the oracle flag live in expert_layout.cpp: this
// file is compiled for AVX-512, and the code that finds out whether the CPU has AVX-512 must not be in it (#795).
void act_quant_q8_1(const float* x, int n, ActQ& a) {
    act_quant_q8_1_512(x, n, a);   // q2_avx512_kernels.inl, shared with the Cascade Lake build (q2_avx512bw.cpp)
}

void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& ws) {
    act_quant_q8_1(x, H, ws.a1);
    s2_expert_vnni_q(blob, ws.a1, out, ws);
}

void s2_expert_vnni_q(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    if (expert_oracle_q8_0_enabled()) {
        expert_oracle_q8_0(blob, a1, out, ws);
        return;
    }
    for (int r = 0; r < FF; ++r) {
        const float g = row_dot(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                                blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                                blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU);
        // SiLU on the GATE, multiplied by up - the same reading `docs/semantics.md` records for the shared
        // expert, and the one that is wrong the other way round in a way that still produces a number.
        ws.ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
    act_quant_q8_1(ws.ff, FF, ws.a2);
    for (int r = 0; r < H; ++r)
        out[r] = row_dot(blob + O_D_CODES + (size_t) r * ROW_D,
                         blob + O_D_SCALES + (size_t) r * SC_D * 2, ws.a2, SC_D);
}


void s2_expert_gu_rows(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float g = row_dot(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                                blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                                blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}

void s2_expert_down_rows(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        out[r] = row_dot(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2, a2, SC_D);
}

namespace {
template<int NT>
void gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, float* const* ff, int r0, int r1) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi<NT>(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                          blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU, g);
        row_dot_multi<NT>(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                          blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}
template<int NT>
void down_rows_multi(const uint8_t* blob, const ActQ* const* a2, float* const* out, int r0, int r1) {
    float o[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi<NT>(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2, a2,
                          SC_D, o);
        for (int t = 0; t < NT; ++t) out[t][r] = o[t];
    }
}
}  // namespace

void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0,
                             int r1) {
    switch (n_tokens) {
        case 1: gu_rows_multi<1>(blob, a1, ff, r0, r1); break;
        case 2: gu_rows_multi<2>(blob, a1, ff, r0, r1); break;
        case 3: gu_rows_multi<3>(blob, a1, ff, r0, r1); break;
        case 4: gu_rows_multi<4>(blob, a1, ff, r0, r1); break;
        case 5: gu_rows_multi<5>(blob, a1, ff, r0, r1); break;
        case 6: gu_rows_multi<6>(blob, a1, ff, r0, r1); break;
        case 7: gu_rows_multi<7>(blob, a1, ff, r0, r1); break;
        default: gu_rows_multi<8>(blob, a1, ff, r0, r1); break;
    }
}

void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out, int r0,
                               int r1) {
    switch (n_tokens) {
        case 1: down_rows_multi<1>(blob, a2, out, r0, r1); break;
        case 2: down_rows_multi<2>(blob, a2, out, r0, r1); break;
        case 3: down_rows_multi<3>(blob, a2, out, r0, r1); break;
        case 4: down_rows_multi<4>(blob, a2, out, r0, r1); break;
        case 5: down_rows_multi<5>(blob, a2, out, r0, r1); break;
        case 6: down_rows_multi<6>(blob, a2, out, r0, r1); break;
        case 7: down_rows_multi<7>(blob, a2, out, r0, r1); break;
        default: down_rows_multi<8>(blob, a2, out, r0, r1); break;
    }
}

// plan v0.3 P6: Q2_0 rows in the GGUF block layout (q2g_rows_nt: q2_avx512_kernels.inl explains them)
void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) {
    q2g_rows_nt(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void s2_expert_vnni_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                          ExpertScratchMulti& ws) {
    if (expert_oracle_q8_0_enabled() || n_tokens < 1 || n_tokens > MAXT) {
        for (int t = 0; t < n_tokens; ++t) s2_expert_vnni_q(blob, *a1[t], out[t], ws.single);
        return;
    }
    switch (n_tokens) {
        case 1: expert_multi<1>(blob, a1, out, ws); break;
        case 2: expert_multi<2>(blob, a1, out, ws); break;
        case 3: expert_multi<3>(blob, a1, out, ws); break;
        case 4: expert_multi<4>(blob, a1, out, ws); break;
        case 5: expert_multi<5>(blob, a1, out, ws); break;
        case 6: expert_multi<6>(blob, a1, out, ws); break;
        case 7: expert_multi<7>(blob, a1, out, ws); break;
        default: expert_multi<8>(blob, a1, out, ws); break;
    }
}

void s2_expert_scalar(const uint8_t* blob, const float* x_in, float* out, bool quant_acts) {
    // BOTH STAGES, which is what `quant_acts` promises and what `bench/micro/cpu_s2.cpp` does NOT do.
    //
    // The original's comment says the oracle "consume[s] the SAME INT8 activation values the VNNI path uses,
    // at both stages", but its code quantizes only the INTERMEDIATE - the gate/up projections still see raw
    // f32 `x`.  Feeding it an already-quantized input (which is what the engine does) makes the two coincide
    // and hides the discrepancy entirely; feeding it a raw f32 input makes the comparison one between two
    // DIFFERENT computations, and the gap measured here was 1.19e-02 - three times P2.S3's tolerance, and
    // read as a kernel bug when it was a fixture bug.
    float xq[H];
    const float* x = x_in;
    if (quant_acts) {
        ActQ a1;
        act_quant_q8_1(x_in, H, a1);
        for (int i = 0; i < H; ++i) xq[i] = a1.scale[i / QKA] * (float) a1.q[i];
        x = xq;
    }
    float ff[FF];
    for (int r = 0; r < FF; ++r) {
        const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU;
        const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2;
        const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU;
        const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2;
        float sg = 0.f, su = 0.f;
        for (int b = 0; b < SC_GU; ++b) {
            const float dg = h2f(gs + 2 * b), du = h2f(us + 2 * b);
            for (int j = 0; j < QK; ++j) {
                const int o = b * QK + j;
                sg += (float) (((gc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * dg * x[o];
                su += (float) (((uc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * du * x[o];
            }
        }
        ff[r] = (sg / (1.f + std::exp(-sg))) * su;
    }
    if (quant_acts) {
        // Replace the exact intermediate with the INT8 values the VNNI path actually sees, so the two differ
        // only by FP32 evaluation order.
        ActQ a2;
        act_quant_q8_1(ff, FF, a2);
        for (int i = 0; i < FF; ++i) ff[i] = a2.scale[i / QKA] * (float) a2.q[i];
    }
    for (int r = 0; r < H; ++r) {
        const uint8_t* dc = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* ds = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        float acc = 0.f;
        for (int b = 0; b < SC_D; ++b) {
            const float d = h2f(ds + 2 * b);
            for (int j = 0; j < QK; ++j)
                acc += (float) (((dc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * d * ff[b * QK + j];
        }
        out[r] = acc;
    }
}

}  // namespace strata::kernels::cpu
