"""The showcase scenarios replay exactly as recorded, and still show a win.

By default the first cycles of every scenario are replayed (fast); set
FLEET_FULL_SHOWCASE=1 to replay every cycle and re-check the summaries.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "web")]

import showcase_scenarios as showcase  # noqa: E402

DATA = json.loads(showcase.OUT.read_text())
FULL = os.environ.get("FLEET_FULL_SHOWCASE") == "1"
PREFIX = 3


class ShowcaseTests(unittest.TestCase):
    def test_file_matches_the_generator(self):
        specs = {s["id"]: s for s in showcase.SCENARIOS}
        self.assertTrue(DATA["scenarios"])
        for scenario in DATA["scenarios"]:
            spec = specs[scenario["id"]]
            self.assertEqual(scenario["positions"], showcase.positions_for(spec))
            self.assertEqual(scenario["params"], showcase.scenario_params(spec))
            self.assertEqual(scenario["comparison"], spec["comparison"])
            self.assertTrue(showcase.wins(scenario["comparison"], scenario["measured"]), scenario["id"])
            self.assertEqual(showcase.summarize(scenario["comparison"], scenario["trace"]), scenario["measured"])

    def test_replay_reproduces_the_recorded_trace(self):
        for scenario in DATA["scenarios"]:
            with self.subTest(scenario=scenario["id"]):
                cycles = None if FULL else PREFIX
                trace = showcase.replay(scenario["params"], scenario["comparison"], scenario["positions"],
                                        showcase.faults_from_json(scenario["faults"]), cycles)
                self.assertEqual(trace, scenario["trace"][:len(trace)])


if __name__ == "__main__":
    unittest.main()
