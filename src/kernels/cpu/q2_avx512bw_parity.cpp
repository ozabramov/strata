// src/kernels/cpu/q2_avx512bw_parity.cpp - the 512-bit Q2_0 rows for AVX-512 CPUs without VBMI (Skylake-SP's successor
// Cascade Lake: Xeon Scalable 2nd gen, Core i9-10900X), q2_avx512bw.cpp, opt-in with STRATA_CPU_512BW=1.
//
//   1. the activation quantizer is bit for bit the AVX-2 one (both are the scalar rule) and the AVX-512 one;
//   2. on a CPU that also has VBMI (Ice Lake and newer, Zen 4): the rows are bit for bit the VBMI kernel's - the
//      two share every operation but the 2-bit unpack;
//   3. on any CPU that runs them: the rows agree with the AVX-2 kernel to a few ulps of the row's magnitude (the
//      integer part is exact, only the float summation order differs), and a token's rows are the same bits alone,
//      in a window of any width and cut into any row ranges (the pool splits rows, a verify window changes the width);
//   4. the dispatch: q2_avx512bw_enabled() is on only with STRATA_CPU_512BW=1, on a CPU with AVX-512 F/BW/VL/DQ/VNNI
//      and without the full AVX-512 set (where the VBMI kernel runs).
// Synthetic data, CPU only; skipped (pass) on a CPU without AVX-512 F/BW/VL/DQ/VNNI.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {

int check_dispatch() {
    const char* sw = std::getenv("STRATA_CPU_512BW");
    const bool asked = sw != nullptr && sw[0] == '1';
    const bool want = asked && !c::cpu_avx512_ok() && c::cpu_q2_avx512bw_ok();
    if (c::q2_avx512bw_enabled() != want) {
        std::printf("FAIL: q2_avx512bw_enabled() = %d, expected %d (STRATA_CPU_512BW %s, full AVX-512 %d, F/BW/VL/DQ/VNNI %d)\n",
                    (int) c::q2_avx512bw_enabled(), (int) want, asked ? "=1" : "unset", (int) c::cpu_avx512_ok(),
                    (int) c::cpu_q2_avx512bw_ok());
        return 1;
    }
    return 0;
}

bool same_actq(const c::ActQ& x, const c::ActQ& y, int n) {
    if (x.nchunks != y.nchunks || std::memcmp(x.q, y.q, (size_t) n) != 0) return false;
    const size_t k = (size_t) x.nchunks;
    return std::memcmp(x.scale, y.scale, k * 4) == 0 && std::memcmp(x.sum, y.sum, k * 4) == 0 &&
           std::memcmp(x.hx, y.hx, k * 4) == 0;
}

}  // namespace

int main() {
    int fail = check_dispatch();
    if (!c::cpu_q2_avx512bw_ok()) {
        std::printf("q2_avx512bw_parity: no AVX-512 F/BW/VL/DQ/VNNI on this CPU, kernels skipped (%s)\n",
                    fail ? "FAIL" : "PASS");
        return fail ? 1 : 0;
    }
    const bool vbmi = c::cpu_avx512_ok();
    std::mt19937 rng(1032);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const int n : {64, 128, 192, 512, 2560}) {   // 1, 2, 3, 8 and 40 blocks: the 8-block groups and their tails
        const int nblocks = n / 64;
        const size_t row_bytes = (size_t) nblocks * 18;
        const int rows = 37;   // not a multiple of 4 or 8
        std::vector<uint8_t> w((size_t) rows * row_bytes);
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < nblocks; ++b) {
                uint8_t* blk = &w[(size_t) r * row_bytes + (size_t) b * 18];
                const uint16_t d = (uint16_t) (0x1C00 + rng() % 0x0800);   // fp16 ~0.004-0.016
                std::memcpy(blk, &d, 2);
                for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) rng();
            }
        const int NT = 8;
        static c::ActQ acts[8], ref_q;
        std::vector<std::vector<float>> xs(NT, std::vector<float>((size_t) n));
        for (int t = 0; t < NT; ++t) {
            for (float& v : xs[(size_t) t]) v = nd(rng) * (t % 3 == 0 ? 8.f : 1.f);
            if (t == 5) xs[(size_t) t].assign((size_t) n, 0.f);   // an all-zero activation: scale 0
            c::act_quant_q8_1_avx512bw(xs[(size_t) t].data(), n, acts[t]);
            c::act_quant_q8_1_avx2(xs[(size_t) t].data(), n, ref_q);
            if (!same_actq(acts[t], ref_q, n)) { std::printf("FAIL n=%d t=%d: quantizer differs from AVX-2\n", n, t); ++fail; }
            if (vbmi) {
                c::act_quant_q8_1(xs[(size_t) t].data(), n, ref_q);
                if (!same_actq(acts[t], ref_q, n)) { std::printf("FAIL n=%d t=%d: quantizer differs from AVX-512\n", n, t); ++fail; }
            }
        }
        const c::ActQ* ap[8];
        for (int t = 0; t < NT; ++t) ap[t] = &acts[t];
        std::vector<std::vector<float>> win(NT, std::vector<float>((size_t) rows));
        std::vector<std::vector<float>> alone = win, cut = win, avx2 = win, full = win;
        float* wo[8]; float* ao[8]; float* co[8]; float* vo[8]; float* fo[8];
        for (int t = 0; t < NT; ++t) {
            wo[t] = win[(size_t) t].data(); ao[t] = alone[(size_t) t].data(); co[t] = cut[(size_t) t].data();
            vo[t] = avx2[(size_t) t].data(); fo[t] = full[(size_t) t].data();
        }
        c::q2_0_gguf_rows_multi_avx512bw(w.data(), row_bytes, nblocks, ap, NT, wo, 0, rows);
        for (int t = 0; t < NT; ++t) c::q2_0_gguf_rows_multi_avx512bw(w.data(), row_bytes, nblocks, ap + t, 1, ao + t, 0, rows);
        for (int r0 = 0; r0 < rows; r0 += 5)
            c::q2_0_gguf_rows_multi_avx512bw(w.data(), row_bytes, nblocks, ap, 3, co, r0, r0 + 5 < rows ? r0 + 5 : rows);
        c::q2_0_gguf_rows_multi_avx2_legacy(w.data(), row_bytes, nblocks, ap, NT, vo, 0, rows);
        if (vbmi) c::q2_0_gguf_rows_multi(w.data(), row_bytes, nblocks, ap, NT, fo, 0, rows);
        for (int t = 0; t < NT; ++t)
            for (int r = 0; r < rows; ++r) {
                const float a = win[(size_t) t][(size_t) r], b = avx2[(size_t) t][(size_t) r];
                if (std::memcmp(&a, &alone[(size_t) t][(size_t) r], 4) != 0) {
                    std::printf("FAIL n=%d t=%d r=%d: a token alone differs from the window\n", n, t, r); ++fail;
                }
                if (t < 3 && std::memcmp(&a, &cut[(size_t) t][(size_t) r], 4) != 0) {
                    std::printf("FAIL n=%d t=%d r=%d: a row range differs from the whole\n", n, t, r); ++fail;
                }
                if (vbmi && std::memcmp(&a, &full[(size_t) t][(size_t) r], 4) != 0) {
                    std::printf("FAIL n=%d t=%d r=%d: %.9g differs from the VBMI kernel's %.9g\n", n, t, r, a,
                                full[(size_t) t][(size_t) r]); ++fail;
                }
                const float tol = 1e-4f * (std::fabs(b) + 1.f);   // a few ulps of the row's magnitude
                if (!(std::fabs(a - b) <= tol)) {
                    std::printf("FAIL n=%d t=%d r=%d: %.7g vs AVX-2 %.7g\n", n, t, r, a, b); ++fail;
                }
            }
    }
    std::printf("q2_avx512bw_parity (%s): %s\n", vbmi ? "against the VBMI kernel and AVX-2" : "against AVX-2",
                fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
