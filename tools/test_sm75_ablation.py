"""Tests for tools/sm75_ablation.py: the drop-one arm list for the Turing (sm_75) kernel series, without starting a server.

    python -m unittest tools.test_sm75_ablation
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sm75_ablation as SA  # noqa: E402


def source_text() -> str:
    parts = []
    for d in ("src", "include"):
        for p in (ROOT / d).rglob("*"):
            if p.suffix in {".cu", ".cpp", ".hpp", ".cuh", ".h"}:
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


class SwitchList(unittest.TestCase):
    def test_every_switch_is_read_by_the_engine(self):
        # A misspelt name would make its "drop" arm the same as "all" and the ablation would credit it with nothing.
        text = source_text()
        missing = [s for s in SA.SM75_SWITCHES if f'"{s}"' not in text]
        self.assertEqual(missing, [])

    def test_matches_the_documented_series(self):
        # docs/DETAILS.md lists the bitwise Turing switches that STRATA_SM75_PERF=1 turns on.
        details = (ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8")
        section = details.split("### RTX 20 (Turing, sm_75) kernels (opt-in)", 1)[1].split("Options that change answers", 1)[0]
        documented = sorted(set(re.findall(r"`(STRATA_[A-Z0-9_]+)", section)) - {SA.SERIES})
        self.assertEqual(sorted(SA.SM75_SWITCHES), documented)

    def test_no_answer_changing_option_is_in_the_bitwise_list(self):
        for name in ("STRATA_SELECT_SIMT", "STRATA_PA_V2T", "STRATA_PF_FUSED", "STRATA_PF_FUSED_SM75", "STRATA_BF16_TC"):
            self.assertNotIn(name, SA.SM75_SWITCHES)


class DropOneArms(unittest.TestCase):
    def test_stock_all_then_one_arm_per_switch(self):
        arms = SA.drop_one_arms(["STRATA_QIL", "STRATA_TC_SM75"])
        self.assertEqual(arms, [
            "stock:",
            "all:STRATA_SM75_PERF=1",
            "drop_QIL:STRATA_SM75_PERF=1,STRATA_QIL=0",
            "drop_TC_SM75:STRATA_SM75_PERF=1,STRATA_TC_SM75=0",
        ])

    def test_extra_settings_go_into_every_arm(self):
        arms = SA.drop_one_arms(["STRATA_QIL"], extra={"STRATA_X": "2"})
        self.assertEqual(arms, [
            "stock:STRATA_X=2",
            "all:STRATA_X=2,STRATA_SM75_PERF=1",
            "drop_QIL:STRATA_X=2,STRATA_SM75_PERF=1,STRATA_QIL=0",
        ])

    def test_arm_names_never_look_like_command_line_options(self):
        for arm in SA.drop_one_arms(SA.SM75_SWITCHES):
            self.assertFalse(arm.startswith("-"), arm)

    def test_arms_parse_back_with_ab_engine(self):
        import ab_engine as AB
        name, env = AB.parse_arm(SA.drop_one_arms(["STRATA_FUSE_WAIT"])[2])
        self.assertEqual((name, env), ("drop_FUSE_WAIT", {"STRATA_SM75_PERF": "1", "STRATA_FUSE_WAIT": "0"}))

    def test_rejects_an_unknown_switch(self):
        with self.assertRaises(ValueError):
            SA.select_switches(["STRATA_NOT_A_SWITCH"])

    def test_an_unknown_switch_on_the_command_line_is_a_usage_error(self):
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            SA.main(["cfg.json", "--only", "BOGUS", "--dry-run"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("STRATA_BOGUS", err.getvalue())

    def test_selects_by_short_or_full_name(self):
        self.assertEqual(SA.select_switches(["QIL", "STRATA_TC_SM75"]), ["STRATA_QIL", "STRATA_TC_SM75"])
        self.assertEqual(SA.select_switches([]), list(SA.SM75_SWITCHES))


if __name__ == "__main__":
    unittest.main()
