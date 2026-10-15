"""Planner session and HTTP validation tests for the web console."""
from __future__ import annotations

import json
import math
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "web")]

from fleet_session import (  # noqa: E402
    CONFIG, PlannerSession, SessionStore, min_spanning_cost, validate_comparison,
    validate_faults, validate_params, validate_positions,
)
import server  # noqa: E402

DEFAULTS = dict(CONFIG["defaults"])


def ring(n: int, radius: float):
    return [[radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)] for i in range(n)]


def params(**over):
    out = dict(DEFAULTS)
    out.update(over)
    return validate_params(out)


class ValidationTests(unittest.TestCase):
    def test_defaults_and_presets_validate(self):
        validate_params(DEFAULTS)
        for preset in CONFIG["presets"]:
            validate_params(preset["params"])

    def test_rejects_bad_numbers(self):
        for key, bad in [("roverCount", 1), ("roverCount", 2.5), ("roverCount", True), ("budget", -1),
                         ("coolingRate", float("nan")), ("communicationRange", float("inf")), ("restarts", 1)]:
            with self.assertRaises(ValueError, msg=f"{key}={bad}"):
                validate_params({**DEFAULTS, key: bad})
        with self.assertRaises(ValueError):
            validate_params("not an object")

    def test_positions_must_match_fleet_and_stay_in_arena(self):
        validate_positions(ring(4, 2.0), 4)
        for bad in (ring(3, 2.0), [[0, 0]] * 3 + [[100, 0]], [[0, 0]] * 3 + [[float("nan"), 0]],
                    [[0, 0]] * 3 + [[True, 0]], [[0, 0, 0]] * 4, "nope"):
            with self.assertRaises(ValueError):
                validate_positions(bad, 4)

    def test_faults_and_comparisons(self):
        self.assertEqual(validate_faults([{"kind": "packet_loss_spike", "rover": 99}], 4), [("packet_loss_spike", None)])
        self.assertEqual(validate_faults([{"kind": "battery_drain", "rover": 2}], 4), [("battery_drain", 2)])
        for bad in ([{"kind": "meteor"}], [{"kind": "battery_drain", "rover": 4}], [{"kind": "battery_drain"}], [{}] * 9):
            with self.assertRaises(ValueError):
                validate_faults(bad, 4)
        self.assertEqual(validate_comparison("assign:random"), "assign:random")
        with self.assertRaises(ValueError):
            validate_comparison("milp")


class SessionTests(unittest.TestCase):
    def test_step_payload_is_consistent(self):
        p = params(roverCount=6)
        session = PlannerSession(p)
        result = session.step(ring(6, 2.4), [], "assign:trust_only")
        self.assertEqual(len(result["rovers"]), 6)
        self.assertEqual(result["cycle"], 1)
        kept = [e for e in result["edges"] if e["active"]]
        self.assertTrue(all(0 <= e["from"] < e["to"] < 6 for e in result["edges"]))
        degree = [0] * 6
        for e in kept:
            degree[e["from"]] += 1
            degree[e["to"]] += 1
        if result["metrics"]["pruneMode"] == "full":
            self.assertLessEqual(max(degree), p["maxDegree"])
            self.assertEqual(result["metrics"]["reachable"], 6)
        self.assertEqual(set(result["assignmentBaselines"]), {"trust_only", "random", "nograph"})
        json.dumps(result, allow_nan=False)

    def test_empty_graph_reports_no_links(self):
        session = PlannerSession(params(roverCount=4, communicationRange=1.5))
        result = session.step(ring(4, 5.0), [], "none")
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["metrics"]["pruneMode"], "none")
        self.assertEqual(result["metrics"]["minSpanningCost"], None)
        self.assertEqual(result["tasks"], [])

    def test_impossible_degree_cap_relaxes_and_says_so(self):
        session = PlannerSession(params(roverCount=6, maxDegree=1, communicationRange=9))
        result = session.step(ring(6, 2.0), [], "none")
        self.assertNotEqual(result["metrics"]["pruneMode"], "full")

    def test_budget_below_spanning_tree_relaxes(self):
        session = PlannerSession(params(roverCount=8, budget=3))
        result = session.step(ring(8, 2.6), [], "none")
        self.assertGreater(result["metrics"]["minSpanningCost"], 3)
        self.assertIn(result["metrics"]["pruneMode"], ("no-budget", "budget-only", "unconstrained", "keep-all-fallback"))

    def test_live_tuning_keeps_session_but_seed_restarts(self):
        store = SessionStore()
        ids = iter(["a", "b", "c"])
        sid, first = store.acquire(None, params(), lambda: next(ids))
        same_id, same = store.acquire(sid, params(budget=90, exploration=0.3), lambda: next(ids))
        self.assertIs(first, same)
        self.assertEqual(same.primary.qubo_budget, 90)
        self.assertEqual(same.followers["random"].exploration, 0.3)
        new_id, fresh = store.acquire(sid, params(seed=5), lambda: next(ids))
        self.assertIsNot(fresh, first)
        self.assertNotEqual(new_id, sid)

    def test_capacity_is_bounded(self):
        store = SessionStore(capacity=2)
        counter = iter(range(10))
        store.acquire(None, params(), lambda: str(next(counter)))
        store.acquire(None, params(), lambda: str(next(counter)))
        with self.assertRaises(OverflowError):
            store.acquire(None, params(), lambda: str(next(counter)))

    def test_idle_sessions_are_evicted_when_full(self):
        store = SessionStore(capacity=2, busy_window=0.0)
        counter = iter("xyz")
        first, _ = store.acquire(None, params(), lambda: next(counter))
        store.acquire(None, params(), lambda: next(counter))
        third, _ = store.acquire(None, params(), lambda: next(counter))
        self.assertNotIn(first, store.sessions)
        self.assertIn(third, store.sessions)

    def test_changing_the_fleet_replaces_the_old_session(self):
        store = SessionStore()
        counter = iter("ab")
        old, _ = store.acquire(None, params(roverCount=6), lambda: next(counter))
        new, _ = store.acquire(old, params(roverCount=12), lambda: next(counter))
        self.assertEqual(list(store.sessions), [new])

    def test_faults_reach_primary_and_baselines(self):
        session = PlannerSession(params(roverCount=5))
        result = session.step(ring(5, 2.0), [("reliability_drop", 2), ("packet_loss_spike", None)], "none")
        kinds = {(f["kind"], f["rover"]) for f in result["faults"]}
        self.assertEqual(kinds, {("reliability_drop", 2), ("packet_loss_spike", None)})
        for follower in session.followers.values():
            self.assertEqual(len(follower.faults), 2)

    def test_followers_share_the_primary_topology(self):
        session = PlannerSession(params(roverCount=6))
        session.step(ring(6, 2.4), [], "none")
        for follower in session.followers.values():
            self.assertEqual(follower.kept, [e for e in session.primary.kept if e in set(follower.edges)])

    def test_bit_sa_runs_on_the_same_q_matrix(self):
        session = PlannerSession(params(roverCount=5))
        result = session.step(ring(5, 2.0), [], "solver:bitsa")
        solver = result["solver"]
        self.assertEqual(solver["baseline"], "bitsa")
        latest = solver["latest"]
        self.assertIsNotNone(result["metrics"]["energy"])
        self.assertAlmostEqual(latest["oursEnergy"], result["metrics"]["energy"])
        tol = 1e-9 * max(1.0, abs(latest["oursEnergy"]))
        self.assertEqual(latest["matched"], latest["feasible"] and latest["energy"] <= latest["oursEnergy"] + tol)
        self.assertEqual(solver["totals"]["runs"], 1)

    def test_min_spanning_cost_is_a_tree_lower_bound(self):
        session = PlannerSession(params(roverCount=6))
        session.step(ring(6, 2.4), [], "none")
        sim = session.primary
        bound = min_spanning_cost(sim.nodes, sim.edges, sim.features, sim.root)
        self.assertIsNotNone(bound)
        self.assertGreaterEqual(bound, len(sim.nodes) - 1)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        quiet = type("QuietHandler", (server.Handler,), {"log_message": lambda *args: None})
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), quiet)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body: bytes, content_type="application/json"):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/step", data=body,
                                         headers={"Content-Type": content_type}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/health", timeout=10) as response:
            body = json.loads(response.read())
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["solvers"]["bitsa"])

    def test_round_trip(self):
        body = {"params": params(roverCount=4), "positions": ring(4, 1.8), "comparison": "assign:random", "faults": []}
        status, result = self.post(json.dumps(body).encode())
        self.assertEqual(status, 200, result)
        self.assertEqual(len(result["rovers"]), 4)
        body["sessionId"] = result["sessionId"]
        status, again = self.post(json.dumps(body).encode())
        self.assertEqual((status, again["sessionId"], again["cycle"]), (200, result["sessionId"], 2))

    def test_rejects_malformed_requests(self):
        self.assertEqual(self.post(b"{not json")[0], 400)
        self.assertEqual(self.post(b"{}", content_type="text/plain")[0], 400)
        self.assertEqual(self.post(json.dumps({"params": DEFAULTS, "positions": []}).encode())[0], 400)
        self.assertEqual(self.post(b"x" * 40000)[0], 400)


if __name__ == "__main__":
    unittest.main()
