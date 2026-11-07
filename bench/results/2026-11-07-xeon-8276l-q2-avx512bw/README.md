# Xeon Platinum 8276L (Cascade Lake): the Q2_0 expert rows on 512-bit VNNI without VBMI

Cascade Lake has AVX-512 F/BW/VL/DQ and VNNI but not VBMI. The AVX-512 Q2_0 rows unpack their 2-bit codes with
VPMULTISHIFTQB (VBMI), so on this CPU the engine ran them on AVX-2. `STRATA_CPU_512BW=1` runs the same 512-bit rows with
an unpack written in AVX-512BW (`src/kernels/cpu/q2_avx512bw.cpp`, `q2_avx512_kernels.inl`). It is opt-in.

## Result

- **One core:** the Q2_0 rows take 42 / 36 / 41% less time per expert at 1 / 2 / 4 tokens (`microbench.txt`).
- **End to end** (Tesla T4, Qwen3.8-Flash-Next Q2_0, 6 interleaved pairs): decode paired medians +4.9 / +4.5 / -0.4 / +5.9%
  (16 of 24 pairs faster), the 6K-token prompt read unchanged (+0.2%). The CPU pool is one part of a decode window on this
  box (the T4 holds 6,489 of the experts), so the row gain shows as a few percent.
- **Same bits as the VBMI kernel:** `q2_avx512bw_parity` compares the rows bit for bit with the VBMI kernel on a CPU that has
  both (i7-11370H, Tiger Lake). On the 8276L it compares them with the AVX-2 rows to a few ulps, and the quantizer bit for bit.

## Hardware and software

| | |
| --- | --- |
| CPU | 2 x Intel Xeon Platinum 8276L (`avx512f avx512bw avx512vl avx512dq avx512_vnni`, no `avx512vbmi`), 192 GB DDR4-2933, HPE DL380 Gen10 |
| VM | Proxmox VE 9.2, Debian 13.7 (kernel 6.12.115), 2 sockets x 26 cores, each vCPU pinned to one physical core, virtual NUMA, 160 GB |
| GPU | Tesla T4 16 GB (sm_75, 70 W), VFIO passthrough, driver 620.70.06, CUDA 13.1.2 |
| Build | GCC 14.2, Release, sm_75. End to end: `main` 9f7d977d plus this change and the sm_75 table (`STRATA_SM75_PERF=1` on in every arm) |
| Model | Qwen3.8-Flash-Next GSQ-RCO Q2_0 native pack: gate, up and down are Q2_0 (`native_experts.txt` type 42) |
| Config | setup's own (`strata-q2_0.json`): `--spec 4 --expert-cache auto --kv int8 --ple-io ram`, 32 pool workers over both sockets |

## Method

`microbench.txt`: `iq_avx2_parity --bench --cpu 2 --pairs iq2_s/q2_0 --nt 1,2,4 --reps 7` pinned to one core, with
`STRATA_CPU_512BW=0` and `=1`; the `engine` line's down column is the Q2_0 rows (the gate/up column there is an i-quant, IQ2_S,
whose kernel this change does not touch).

`09-512bw.*`: `tools/ab_engine.py`, a fresh server per arm per round, the order reversed every round, three short greedy chats
and a 6,345-token prompt. Arms: `avx2` (`STRATA_CPU_512BW=0`), `bw` (`=1`), and `bw_bf16` (`=1` and `STRATA_BF16_TC=1`, a
prompt-path switch measured in the same run).

## Numbers

| Request | `bw` / `avx2` decode, paired median | Range | Pairs faster |
| --- | ---: | ---: | ---: |
| chat0 | 1.049 | 0.973 - 1.209 | 5/6 |
| chat1 | 1.045 | 0.961 - 1.156 | 4/6 |
| chat2 | 0.996 | 0.883 - 1.187 | 2/6 |
| long (6,345-token prompt) | 1.059 | 0.923 - 1.129 | 5/6 |

`bw_bf16` against `bw`: prompt read 447 -> 608 tok/s (1.346, 6 of 6 pairs), decode 0.97 / 0.98 / 0.98 / 0.96 (1-2 of 6 pairs
faster).

Microbench, ms per expert (median of 7), down rows: 1 token 0.129 -> 0.075, 2 tokens 0.188 -> 0.121, 4 tokens 0.308 -> 0.182.

## Limits

- One CPU model, in a VM. The answers are not compared across arms: `=1` sums in the AVX-512 kernel's order, not AVX-2's,
  so the last bits and some greedy answers change (as between an AVX-2 and an Ice Lake machine). Identity is checked against
  the VBMI kernel by the parity test, not end to end.
- The IQ2_S gate/up rows in the microbench were 4-6% slower in the `=1` run though their kernel did not change; a lower AVX-512
  clock on this Xeon is a likely cause, not measured.
- Not built with MSVC or on Windows; HIP and SYCL builds do not compile this file differently (it is CPU-only).
