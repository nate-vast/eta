"""Paired forecasts are scored against the same explicitly verified endpoint."""
import json
import os
import unittest
from unittest.mock import patch

from test_eta import Base, eta, run


class Paired(Base):
    def forecast(self, job="task", guess=1, *options):
        return json.loads(run(["estimate", "--agent-min", str(guess), "--task-id", job,
                               "--reference", "acceptance tests", "--model", "test", "--agent",
                               "codex", "--json", *options]))

    def close_after(self, p, elapsed):
        row = next(r for r in eta.load_jsonl(eta.pred_path()) if r["id"] == p["id"])
        with patch.object(eta.time, "time", return_value=row["created"] + elapsed):
            run(["done", "--id", p["id"]])

    def test_measured_mode_learns_from_abstentions_and_ignores_baseline(self):
        start = self.now - 1000
        for i, actual in enumerate((20, 30, 50)):
            with patch.object(eta.time, "time", return_value=start + i * 100):
                p = self.forecast(str(i), 10, "--measured")
            self.assertIsNone(p["est_s"])
            self.close_after(p, actual)
        for guess in (.01, 50):
            p = self.forecast("new", guess, "--measured", "--no-record")
            self.assertEqual((p["est_s"], p["method"]), (30, "verified task history"))
        self.assertIn("measured ETA coverage: 0/3", run(["stats"]))

    def test_long_stats_use_actual_time_and_one_check_in_per_task(self):
        with patch.object(eta.time, "time", return_value=self.now - 2000):
            a = self.forecast("one", 20, "--measured", "--batch", "1", "3", "180")
            b = self.forecast("two", 20, "--measured")
            short = self.forecast("short", 20, "--measured")
        self.close_after(a, 360)
        self.close_after(b, 400)
        self.close_after(short, 299)
        repeat = self.forecast("one", 30, "--measured")
        self.close_after(repeat, 500)
        report = run(["stats", "--min-remaining-min", "5"])
        self.assertIn("1/2 verified tasks", report)
        self.assertIn("Agent: mean absolute miss 14m 00s; too long on 1/1", report)
        self.assertIn("1 closer, 0 tied, 0 further", report)

    def test_live_progress_and_fixed_wait(self):
        p = self.forecast("job", 5, "--batch", "20", "100", "30", "--wait-min", "2")
        self.assertEqual((p["est_s"], p["agent_s"]), (240, 300))
        self.assertGreaterEqual(p["low_s"], 120)
        self.assertEqual(p["method"], "measured progress")
        with self.assertRaises(SystemExit):
            self.forecast("job", 1, "--batch", "0", "10", "20")

    def test_report_preserves_subminute_and_minute_precision(self):
        p = self.forecast("display", .75)
        self.assertEqual((p["remaining"], p["high"]), ("45.0s", "1m 21s"))

    def test_batch_with_no_work_left(self):
        self.assertEqual(self.forecast("job", .5, "--batch", "10", "10", "20")["est_s"], 0)

    def test_only_verified_independent_past_tasks(self):
        t = self.now - 1000
        for i, name in enumerate(("one", "one", "two", "three")):
            with patch.object(eta.time, "time", return_value=t + i * 100):
                p = self.forecast(name)
            self.close_after(p, 20)
        self.assertEqual(run(["references", "--model", "test", "--agent", "codex"]).strip(),
                         "acceptance tests")
        self.assertEqual(self.forecast("other")["n_cal"], 3)
        self.assertEqual(self.forecast("one")["n_cal"], 2)
        self.assertEqual(self.forecast("other", 1, "--model", "different")["n_cal"], 0)
        with patch.object(eta.time, "time", return_value=t + 1):
            self.assertEqual(self.forecast("past")["n_cal"], 0)
        with patch.object(os, "getcwd", return_value=self.tmp.name):
            self.assertEqual(self.forecast("other")["n_cal"], 0)
        self.assertEqual(self.forecast("new", 1, "--wait-min", "1")["est_s"], 60)

    def test_keep_an_accurate_agent_and_correct_consistent_bias(self):
        t = self.now - 1000
        for scale, model in [(1, "accurate"), (2, "optimistic")]:
            for i, secs in enumerate((10, 40, 20, 30)):
                with patch.object(eta.time, "time", return_value=t + i * 100):
                    p = self.forecast(str(i), secs / 60, "--model", model)
                self.close_after(p, secs * scale)
        self.assertEqual(self.forecast("new", 1, "--model", "accurate")["method"], "working agent")
        p = self.forecast("new", 1, "--model", "optimistic")
        self.assertEqual((p["method"], p["est_s"]), ("agent bias correction", 120))

    def test_idempotent_exact_closure_and_open_coverage(self):
        a, b = self.forecast("one"), self.forecast("two")
        run(["done", "--id", a["id"], a["id"]])
        run(["done", "--id", a["id"]])
        self.assertEqual(len(eta.confirmed_pairs()), 1)
        self.assertIn("1/2", run(["stats"]))
        with self.assertRaises(SystemExit):
            run(["done", "--id", b["id"], "missing"])
        self.assertEqual(len(eta.confirmed_pairs()), 1)
        run(["done", "--id", a["id"], b["id"]])
        self.assertEqual(len(eta.confirmed_pairs()), 2)

    def test_fixed_wait_in_legacy_call(self):
        with patch.object(eta, "calibration", return_value=dict(n=3, factor=.43, lo=.6, hi=1.8, hits=0)):
            run(["estimate", "--minutes", "1", "--wait-min", "22", "--transcript",
                 self.empty_transcript()])
        p = eta.load_jsonl(eta.pred_path())[0]
        self.assertAlmostEqual(p["est_s"], 1320 + .43 * 60)
        self.assertGreaterEqual(p["low_s"], 1320)

    def empty_transcript(self):
        path = self.path("empty.jsonl")
        with open(path, "w") as f:
            f.write("{}\n")
        return path


if __name__ == "__main__":
    unittest.main()
