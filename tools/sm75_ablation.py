"""Drop-one ablation of the Turing (sm_75) kernel series on one model config, through tools/ab_engine.py.

    python tools/sm75_ablation.py strata-q2_0.json [--only QIL,TC_SM75] [--extra STRATA_X=1] [--rounds 3] [ab_engine options]

Arms: `stock` (the config as it is), `all` (STRATA_SM75_PERF=1: every bitwise Turing kernel) and one `drop_<NAME>` per
switch (STRATA_SM75_PERF=1 with that switch's own variable at 0, which wins over the series in both directions; =0 turns
off each of them, FUSE_WAIT / GR_DOWN_TG / QIL included).  A switch whose drop arm is slower than `all` pays for itself on
this card; the ones that do not are the candidates to leave out of an sm_75 default table (src/core/nv_defaults.cpp).
The switches are bitwise, so ab_engine's answer report should read IDENTICAL for every request.

Only a compute capability 7.5 card runs the series; on any other card every arm is the stock engine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_engine  # noqa: E402

SERIES = "STRATA_SM75_PERF"
# docs/DETAILS.md "RTX 20 (Turing, sm_75) kernels": the bitwise switches STRATA_SM75_PERF=1 turns on.  The options that
# change answers (STRATA_SELECT_SIMT, STRATA_PA_V2T, STRATA_PF_FUSED_SM75, STRATA_BF16_TC) are not part of it.
SM75_SWITCHES = (
    "STRATA_SM75_IL", "STRATA_SM75_EXP", "STRATA_GR_UP75", "STRATA_GR_DOWN_TG", "STRATA_SCORES_HALVE",
    "STRATA_TOPK_SPLIT", "STRATA_TOPK_SAMPLE", "STRATA_ATTN_V3", "STRATA_Q2_WARP", "STRATA_FUSE_WAIT", "STRATA_QIL",
    "STRATA_TC_SM75", "STRATA_PA_V1_PREFETCH", "STRATA_SCORES_THREAD",
)


def short_name(switch: str) -> str:
    return switch.removeprefix("STRATA_")


def select_switches(names: list[str]) -> list[str]:
    """Switches by short (QIL) or full (STRATA_QIL) name, in SM75_SWITCHES order; none named = all of them."""
    if not names:
        return list(SM75_SWITCHES)
    wanted = {n if n.startswith("STRATA_") else "STRATA_" + n for n in names}
    unknown = sorted(wanted - set(SM75_SWITCHES))
    if unknown:
        raise ValueError("not a Turing series switch: " + ", ".join(unknown))
    return [s for s in SM75_SWITCHES if s in wanted]


def drop_one_arms(switches: list[str], extra: dict[str, str] | None = None) -> list[str]:
    """ab_engine arm specs: stock, all, then drop_<NAME> for each switch; `extra` settings go into every arm."""
    base = dict(extra or {})

    def spec(name: str, env: dict[str, str]) -> str:
        return name + ":" + ",".join(f"{k}={v}" for k, v in env.items())

    arms = [spec("stock", base), spec("all", {**base, SERIES: "1"})]
    arms += [spec("drop_" + short_name(s), {**base, SERIES: "1", s: "0"}) for s in switches]
    return arms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("config")
    ap.add_argument("--only", default="", help="comma-separated switches to drop (default: all of them)")
    ap.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE", help="a setting for every arm")
    ap.add_argument("--dry-run", action="store_true", help="print the arms and stop")
    a, rest = ap.parse_known_args(argv)
    try:
        switches = select_switches([x for x in a.only.split(",") if x])
    except ValueError as e:
        ap.error(str(e))
    extra = dict(x.split("=", 1) for x in a.extra)
    arms = drop_one_arms(switches, extra)
    if a.dry_run:
        print("\n".join(arms))
        return 0
    return ab_engine.main([a.config, *arms, *rest])


if __name__ == "__main__":
    sys.exit(main())
