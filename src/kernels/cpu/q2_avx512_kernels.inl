// src/kernels/cpu/q2_avx512_kernels.inl - the 512-bit Q2_0 kernels that expert.cpp (compiled with VBMI) and
// q2_avx512bw.cpp (compiled without it, for Cascade Lake) both build.  Included INSIDE an anonymous namespace in
// strata::kernels::cpu, after <immintrin.h>, <cstring> and expert.hpp; the includer defines STRATA_Q2_UNPACK_VBMI
// to 1 (VPMULTISHIFTQB) or 0 (AVX-512BW shifts and byte blends).  Everything but the unpack is the same source, and
// both unpacks give the same 64 codes, so the two builds compute the same bits.
#if !defined(STRATA_Q2_UNPACK_VBMI)
#error "define STRATA_Q2_UNPACK_VBMI (1: VBMI, 0: AVX-512BW only) before including q2_avx512_kernels.inl"
#endif

/// fp16 -> fp32, written out rather than using `_cvtsh_ss`, because the F16C intrinsic's behaviour on
/// subnormals is the one place the two can differ and the scales in this artifact are small.
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// The x86 Q8_0 quantization and generic Q2_0 dot sequences are adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// ggml/src/ggml-cpu/arch/x86/quants.c: quantize_row_q8_0 and
// ggml/src/ggml-cpu/quants.c: ggml_vec_dot_q2_0_q8_0_generic.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
void quantize_oracle_q8_0(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    for (int chunk = 0; chunk < a.nchunks; ++chunk) {
        const float* xb = x + chunk * QKA;
        __m256 v0 = _mm256_loadu_ps(xb);
        __m256 v1 = _mm256_loadu_ps(xb + 8);
        __m256 v2 = _mm256_loadu_ps(xb + 16);
        __m256 v3 = _mm256_loadu_ps(xb + 24);
        const __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 maximum = _mm256_andnot_ps(sign, v0);
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign, v1));
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign, v2));
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign, v3));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maximum, 1), _mm256_castps256_ps128(maximum));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float amax = _mm_cvtss_f32(max4);
        const float d = amax / 127.0f;
        const uint16_t half_d = (uint16_t) _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(d), 0));
        const __m256 inverse = _mm256_set1_ps(amax != 0.0f ? 127.0f / amax : 0.0f);
        v0 = _mm256_round_ps(_mm256_mul_ps(v0, inverse), _MM_FROUND_TO_NEAREST_INT);
        v1 = _mm256_round_ps(_mm256_mul_ps(v1, inverse), _MM_FROUND_TO_NEAREST_INT);
        v2 = _mm256_round_ps(_mm256_mul_ps(v2, inverse), _MM_FROUND_TO_NEAREST_INT);
        v3 = _mm256_round_ps(_mm256_mul_ps(v3, inverse), _MM_FROUND_TO_NEAREST_INT);
        __m256i i0 = _mm256_packs_epi32(_mm256_cvtps_epi32(v0), _mm256_cvtps_epi32(v1));
        const __m256i i2 = _mm256_packs_epi32(_mm256_cvtps_epi32(v2), _mm256_cvtps_epi32(v3));
        i0 = _mm256_packs_epi16(i0, i2);
        i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        int8_t* q = a.q + chunk * QKA;
        _mm256_storeu_si256((__m256i*) q, i0);
        int32_t sum = 0;
        for (int j = 0; j < QKA; ++j) sum += q[j];
        a.scale[chunk] = h2f((const uint8_t*) &half_d);
        a.sum[chunk] = sum;
        a.hx[chunk] = a.scale[chunk] * float(sum);
    }
}

// 16 code bytes (code j in byte j/4, bits 2*(j%4)) -> the 64 codes of one Q2_0 block, one per byte, in order.
#if STRATA_Q2_UNPACK_VBMI
inline __m512i unpack64_q2_0(const uint8_t* codes) {
    const __m128i packed = _mm_loadu_si128((const __m128i*) codes);          // 8 u16 = 64 codes
    const __m512i lanes = _mm512_cvtepu16_epi64(packed);                    // 8 qwords, 8 codes each
    const __m512i ctrl = _mm512_set1_epi64((long long) 0x0E0C0A0806040200ULL);
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, lanes), _mm512_set1_epi8(3));
}
#else
// Without VBMI (Skylake-SP, Cascade Lake): output byte p takes code byte p/4 (one in-lane shuffle of the 16 bytes
// broadcast to the four lanes), so every 16-bit word w holds B | B << 8 with B = code byte w/2.  The word's low
// byte needs B >> 0 (even w) or B >> 4 (odd w), its high byte two bits more: shift the odd words by 4, then take
// the high bytes from that shifted by 2.  Six single-uop instructions, no variable shifts.
inline __m512i unpack64_q2_0(const uint8_t* codes) {
    const __m512i bytes = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) codes));
    const __m512i ctrl = _mm512_set_epi64(0x0F0F0F0F0E0E0E0ELL, 0x0D0D0D0D0C0C0C0CLL, 0x0B0B0B0B0A0A0A0ALL,
                                          0x0909090908080808LL, 0x0707070706060606LL, 0x0505050504040404LL,
                                          0x0303030302020202LL, 0x0101010100000000LL);
    const __m512i x = _mm512_shuffle_epi8(bytes, ctrl);
    const __m512i lo = _mm512_mask_mov_epi16(x, (__mmask32) 0xAAAAAAAAu, _mm512_srli_epi16(x, 4));
    const __m512i both = _mm512_mask_mov_epi8(lo, (__mmask64) 0xAAAAAAAAAAAAAAAAull, _mm512_srli_epi16(lo, 2));
    return _mm512_and_si512(both, _mm512_set1_epi8(3));
}
#endif

/// The activation quantizer (`act_quant_q8_1`): the Q8_0 oracle where it is on, else AVX-512 with the scalar rule's
/// operations per element - max of |x|, s = amax/127, one multiply by 1/s, +-0.5 away from zero (`std::lround`'s
/// rule; `_mm512_roundscale_ps` would round half to even), truncation, clamp to +-127 - so the result is bitwise
/// the scalar rule's and the AVX-2 quantizer's.  The scalar loop took ~22 us per 2560 values.
inline void act_quant_q8_1_512(const float* x, int n, ActQ& a) {
    if (expert_oracle_q8_0_enabled()) {
        quantize_oracle_q8_0(x, n, a);
        return;
    }
    a.nchunks = n / QKA;
    const __m512 half = _mm512_set1_ps(0.5f), mhalf = _mm512_set1_ps(-0.5f), zero = _mm512_setzero_ps();
    const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127);
    const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        const __m512 x0 = _mm512_loadu_ps(xb), x1 = _mm512_loadu_ps(xb + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_and_ps(x0, absmask), _mm512_and_ps(x1, absmask)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m512 vinv = _mm512_set1_ps(inv);
        const __m512 t0 = _mm512_mul_ps(x0, vinv), t1 = _mm512_mul_ps(x1, vinv);
        const __m512 r0 = _mm512_add_ps(t0, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t0, zero, _CMP_GE_OQ), mhalf, half));
        const __m512 r1 = _mm512_add_ps(t1, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t1, zero, _CMP_GE_OQ), mhalf, half));
        __m512i v0 = _mm512_cvttps_epi32(r0), v1 = _mm512_cvttps_epi32(r1);
        v0 = _mm512_min_epi32(_mm512_max_epi32(v0, lo), hi);
        v1 = _mm512_min_epi32(_mm512_max_epi32(v1, lo), hi);
        _mm_storeu_si128((__m128i*) (a.q + k * QKA), _mm512_cvtepi32_epi8(v0));
        _mm_storeu_si128((__m128i*) (a.q + k * QKA + 16), _mm512_cvtepi32_epi8(v1));
        const int32_t sum = _mm512_reduce_add_epi32(_mm512_add_epi32(v0, v1));
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

// ================================ plan v0.3 P6: Q2_0 rows in the GGUF block layout ================================
//
// The IQ model files keep their down projections in Q2_0 GGUF blocks (fp16 d, then 16 code bytes: 18 bytes per
// 64 weights, interleaved), which ggml-cpu computes with a scalar loop on x86.  This is expert.cpp's
// `row_dot_multi_z` with the block stride of the GGUF layout: the same unpack, the same VNNI dot, the same correction.
template<int NT>
inline void q2g_row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m512 acc[NT], corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm512_setzero_ps(); corr[t] = _mm512_setzero_ps(); }
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        alignas(16) uint16_t sc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < nb; ++i) std::memcpy(&sc[i], row + (size_t) (b0 + i) * 18, 2);
        const __m512 d8 = _mm512_castps256_ps512(_mm256_cvtph_ps(_mm_load_si128((const __m128i*) sc)));
        const __m512 dd = _mm512_maskz_mov_ps(m16, _mm512_permutexvar_ps(dup, d8));
        __m512 p[NT];
        for (int t = 0; t < NT; ++t) {
            p[t] = _mm512_mul_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->scale + 2 * b0));
            corr[t] = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->hx + 2 * b0), corr[t]);
        }
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i w = unpack64_q2_0(row + (size_t) b * 18 + 2);
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
template<int NT>
void q2g_rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        q2g_row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

inline void q2g_rows_nt(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                        float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: q2g_rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: q2g_rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: q2g_rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: q2g_rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 5: q2g_rows<5>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 6: q2g_rows<6>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 7: q2g_rows<7>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default: q2g_rows<8>(w, row_bytes, nblocks, a, out, r0, r1); break;
    }
}
