# Tesla T4 (sm_75): the Turing kernel series as the sm_75 default

The bitwise Turing kernel series (`STRATA_SM75_PERF=1`, `include/strata/core/sm75_series.hpp`) has been opt-in because no one on
the project had a compute capability 7.5 card. These runs are on a Tesla T4. They are the measurements behind the sm_75 table in
`src/core/nv_defaults.cpp`.

## Result

- **Decode +20 to +27%** with the series on (paired medians of four requests over 8 interleaved pairs of fresh engines, setup's
  config; 30 of 32 pairs faster). The 6K-token prompt read +4%.
- **Answers byte-identical** with the series on and off, and with each of its 14 switches dropped in turn (deterministic lane,
  full-answer SHA-256, every request and round).
- `STRATA_SM75_IL` carries most of the gain: dropping it alone costs about as much as turning the whole series off. The other 13
  switches were within the card's round-to-round noise (about +-5%); none was measurably slower, so the table is the series as a
  whole rather than a pruned list.

## Hardware and software

| | |
| --- | --- |
| GPU | NVIDIA Tesla T4, 16 GB (15,360 MiB usable), Turing sm_75, passive, 70 W power limit, VBIOS 90.04.96.00.01 |
| PCIe | Gen3 x16, passed through to the VM (VFIO) |
| Host | HPE ProLiant DL380 Gen10, 2 x Intel Xeon Platinum 8276L (Cascade Lake: AVX-512 F/BW/VL/VNNI, **no VBMI**), 192 GB DDR4-2933 |
| Hypervisor | Proxmox VE 9.2.24, kernel 7.0.14 |
| VM | Debian 13.7, kernel 6.12.115, 2 sockets x 26 cores (CPU type host, virtual NUMA, each vCPU pinned 1:1 to a physical core of its socket, no hyperthreads), 160 GB (80 GB bound per node) |
| Driver / CUDA | NVIDIA open kernel modules 620.70.06, CUDA 13.1.2 (nvcc 13.1.115, GCC 14.2) |
| Strata | `main` at `9f7d977d` (version 0.2.11, engine 0.1.57), built by `./setup.sh --build` for sm_75 (`BUILD.json`); ggml-cpu with `-march=native` |
| Model | Qwen3.8-Flash-Next Q2_0 (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`), MTP draft layer, n-gram table in RAM |
| Config | setup's own, unchanged: `strata-q2_0.json` (`--spec 4`, `--expert-cache auto`: 6,489 experts on the card, `--kv int8`, `--ple-io ram`) |

The engine reports its CPU path as AVX2 (the AVX-512 expert kernels need VBMI), 32 pool workers spread over both sockets, and
the NUMA pool split on, with memory not placed per node for this model's arena.

## Method

All runs use `tools/ab_engine.py`: a fresh server per arm per round, the arms' order reversed every round, one short warm-up
answer, then the same four greedy requests (three short chats with 256 new tokens at most and one 6,345-token prompt). Every
answer's SHA-256 is logged and compared.

| Run | Files | Lane | Arms | Rounds |
| --- | --- | --- | --- | --- |
| 1 | `01-stock-vs-all.*` | setup's config | `stock`, `all` (`STRATA_SM75_PERF=1`) | 2 |
| 2 | `02-identity.*`, `02-gpu.csv` | deterministic | `stock`, `all` | 2 |
| 3 | `03-dropone.*`, `03-gpu.csv` | deterministic | `stock`, `all`, `drop_<NAME>` for each of 14 switches (`tools/sm75_ablation.py`) | 3 |
| 4 | `04-confirm-default.*`, `04-gpu.csv` | setup's config | `stock`, `all` | 8 |

The deterministic lane is the one `docs/DETAILS.md` gives for byte-identical repeats: `STRATA_IQ_MT_MIN=1` and the engine
arguments `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`. In setup's config greedy answers can change between rounds of the
same arm (drafting and the adaptive expert tier), so identity is judged only in the deterministic lane.

## Numbers

Run 4, setup's config, `all` / `stock` decode per pair:

| Request | Paired median | Range | Pairs faster | Medians (stock -> all) |
| --- | ---: | ---: | ---: | ---: |
| chat0 | 1.249 | 1.089 - 1.373 | 8/8 | 39.9 -> 48.6 tok/s |
| chat1 | 1.236 | 1.054 - 1.300 | 8/8 | 29.6 -> 35.3 tok/s |
| chat2 | 1.267 | 0.977 - 1.417 | 7/8 | 27.9 -> 34.9 tok/s |
| long (6,345-token prompt) | 1.201 | 1.000 - 1.414 | 7/8 | 26.6 -> 32.3 tok/s |

The 6K prompt read: 427.8 -> 447.6 tok/s (paired median 1.044).

Run 2, deterministic lane: decode 41.6 -> 48.7, 31.9 -> 36.8, 30.6 -> 35.9, 25.8 -> 29.9 tok/s; all four requests `IDENTICAL`.

Run 3, drop-one (decode median against `all`, mean of the four requests; `03-dropone-summary.txt` per request): `stock` -20.4%,
`drop_SM75_IL` -16.7%, `drop_TC_SM75` -4.2%, `drop_SM75_EXP` -2.9%, `drop_TOPK_SAMPLE` -2.4%, `drop_Q2_WARP` -2.2%,
`drop_GR_DOWN_TG` -1.6%, `drop_GR_UP75` -1.2%, `drop_SCORES_THREAD` -0.4%, `drop_PA_V1_PREFETCH` +0.6%, `drop_SCORES_HALVE` +1.1%,
`drop_ATTN_V3` +1.8%, `drop_QIL` +1.9%, `drop_FUSE_WAIT` +3.0%, `drop_TOPK_SPLIT` +3.8%. Every arm's answers equal `all`'s.

## Limits

- One card, one model (Q2_0), one machine, in a VM. The decode speeds drifted down over the evening (run 1 `stock` chat0 47.5,
  run 4 39.9 tok/s); the interleaving cancels a drift but not a difference between hosts.
- The T4 runs at its 70 W cap: throttle reason `0x4` (software power cap) in most samples of the GPU logs, clocks down to about
  660 MHz under load, at most 80 °C, no thermal throttling. A Turing card with more power (RTX 2080 Ti, Tesla T10) may weigh
  the switches differently.
- The identity check is four requests with full-answer hashes, not the project's 10-prompt and exact_pp 4K/20K gate.
- Three rounds per drop-one arm cannot separate effects under about 5% on this card, so the individual switches other than
  `STRATA_SM75_IL` are not ranked. The answer-changing Turing options (`STRATA_BF16_TC`, `STRATA_PA_V2T`, `STRATA_SELECT_SIMT`,
  `STRATA_PF_FUSED_SM75`) were not run.
