// src/kernels/cpu/q2_avx512bw.cpp - the 512-bit Q2_0 rows and activation quantizer for AVX-512 CPUs without VBMI:
// Cascade Lake (Xeon Scalable 2nd gen, Core i9-10900X) has AVX-512 F/BW/VL/DQ and VNNI but not VPMULTISHIFTQB, so the
// engine ran its Q2_0 experts on the AVX-2 kernels there.  The kernels are expert.cpp's (q2_avx512_kernels.inl)
// with the 2-bit unpack done in AVX-512BW, and this file is compiled for F/BW/VL/DQ/VNNI only, so nothing in it can
// fault on those CPUs.  Taken where q2_avx512bw_enabled() (STRATA_CPU_512BW=1, expert_layout.cpp).
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <immintrin.h>

#include <cstring>

namespace strata::kernels::cpu {
namespace {
#define STRATA_Q2_UNPACK_VBMI 0
#include "q2_avx512_kernels.inl"
}  // namespace

void q2_0_gguf_rows_multi_avx512bw(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                   float* const* out, int r0, int r1) {
    q2g_rows_nt(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_q8_1_avx512bw(const float* x, int n, ActQ& a) { act_quant_q8_1_512(x, n, a); }

}  // namespace strata::kernels::cpu
