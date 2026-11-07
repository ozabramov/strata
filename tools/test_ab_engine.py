"""Tests for tools/ab_engine.py's arm parsing and its answer-identity report, without starting a server.

    python -m unittest tools.test_ab_engine
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ab_engine as AB  # noqa: E402


class ParseArm(unittest.TestCase):
    def test_name_and_environment(self):
        self.assertEqual(AB.parse_arm("on:STRATA_SM75_PERF=1,STRATA_QIL=0"),
                         ("on", {"STRATA_SM75_PERF": "1", "STRATA_QIL": "0"}))

    def test_an_arm_without_settings_runs_the_config_as_it_is(self):
        self.assertEqual(AB.parse_arm("stock:"), ("stock", {}))
        self.assertEqual(AB.parse_arm("stock"), ("stock", {}))

    def test_a_value_may_hold_an_equals_sign(self):
        self.assertEqual(AB.parse_arm("x:xargs=--a=1"), ("x", {"xargs": "--a=1"}))


class AnswerDigest(unittest.TestCase):
    def test_hashes_the_whole_answer(self):
        # The log kept only the first 80 characters, so two answers that part after them looked the same.
        text = "a" * 80 + "tail"
        self.assertEqual(AB.answer_digest(text), hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertNotEqual(AB.answer_digest(text), AB.answer_digest("a" * 80 + "other"))


class IdentityReport(unittest.TestCase):
    def test_every_arm_and_round_the_same_answer(self):
        self.assertEqual(AB.identity_report({"all": ["h1", "h1"], "drop_QIL": ["h1", "h1"]}), "IDENTICAL")

    def test_names_the_arms_whose_answer_differs_from_the_first_arm(self):
        report = AB.identity_report({"all": ["h1", "h1"], "drop_QIL": ["h2", "h2"], "drop_TC": ["h1", "h1"]})
        self.assertEqual(report, "differs from all: drop_QIL")

    def test_names_an_arm_whose_answer_changes_between_its_own_rounds(self):
        report = AB.identity_report({"all": ["h1", "h1"], "drop_QIL": ["h1", "h3"]})
        self.assertEqual(report, "unstable: drop_QIL")

    def test_reports_both_kinds(self):
        report = AB.identity_report({"all": ["h1"], "a": ["h2"], "b": ["h1", "h4"]})
        self.assertEqual(report, "differs from all: a; unstable: b")

    def test_an_unstable_first_arm_is_compared_by_its_first_answer(self):
        report = AB.identity_report({"all": ["h1", "h2"], "a": ["h1", "h1"]})
        self.assertEqual(report, "unstable: all")

    def test_no_answers_yet(self):
        self.assertEqual(AB.identity_report({"all": [], "a": []}), "no answers")


if __name__ == "__main__":
    unittest.main()
