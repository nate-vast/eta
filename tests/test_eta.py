"""Tests for eta.py. Run: python3 -m unittest discover -s tests -v"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "skills", "eta", "scripts", "eta.py")
spec = importlib.util.spec_from_file_location("eta", SCRIPT)
eta = importlib.util.module_from_spec(spec)
sys.modules.setdefault("eta", eta)   # baseline.py and waits.py share this copy
spec.loader.exec_module(eta)


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


class Claude:
    """Builds a minimal Claude Code transcript."""

    def __init__(self):
        self.lines, self.n = [], 0

    def prompt(self, ts, text, **extra):
        self.lines.append({"type": "user", "timestamp": iso(ts), "isSidechain": False,
                           "message": {"role": "user", "content": text}, **extra})

    def tool(self, ts, dur, name="Bash", inp=None):
        self.n += 1
        tid = f"toolu_{self.n}"
        self.lines.append({"type": "assistant", "timestamp": iso(ts), "isSidechain": False,
                           "message": {"role": "assistant", "content": [
                               {"type": "tool_use", "id": tid, "name": name,
                                "input": inp or {"command": f"step {self.n}"}}]}})
        self.lines.append({"type": "user", "timestamp": iso(ts + dur), "isSidechain": False,
                           "message": {"role": "user", "content": [
                               {"type": "tool_result", "tool_use_id": tid, "content": "ok"}]}})

    def say(self, ts):
        self.lines.append({"type": "assistant", "timestamp": iso(ts), "isSidechain": False,
                           "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}})

    def write(self, path):
        with open(path, "w") as f:
            for l in self.lines:
                f.write(json.dumps(l) + "\n")
        return path


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        eta.main(argv)
    return buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ETA_HOME"] = os.path.join(self.tmp.name, "home")
        self.now = time.time()

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)


class TestPrompts(unittest.TestCase):
    def test_eta_detection(self):
        self.assertEqual(eta._clean_prompt('<skill name="eta" location="/x/SKILL.md">\nbody\n</skill>\n\nstats'),
                         "/eta stats")
        self.assertTrue(eta.is_eta_prompt(eta._clean_prompt('<skill name="eta" location="/x">b</skill>')))
        for t in ["/eta", "/eta stats", "$eta", "/skill:eta", "eta?", "ETA on this?", "how long left?",
                  "when will it be done", "how much longer"]:
            self.assertTrue(eta.is_eta_prompt(t), t)
        for t in ["build the etag cache layer", "add a beta flag",
                  "write a long essay about how long rivers are " * 4]:
            self.assertFalse(eta.is_eta_prompt(t), t)

    def test_command_and_reminder_cleaning(self):
        self.assertEqual(eta._clean_prompt(
            "<command-name>/eta</command-name><command-args>stats</command-args>"), "/eta stats")
        self.assertEqual(eta._clean_prompt(
            [{"type": "text", "text": "<system-reminder>x</system-reminder>do the thing"}]),
            "do the thing")

    def test_started_parsing(self):
        now = datetime(2026, 11, 23, 15, 0).timestamp()
        self.assertAlmostEqual(eta.parse_started("25m", now), now - 1500)
        self.assertAlmostEqual(eta.parse_started("1h10m", now), now - 4200)
        self.assertAlmostEqual(eta.parse_started("2:30pm", now), now - 1800)
        self.assertAlmostEqual(eta.parse_started("14:30", now), now - 1800)
        # a clock time later than now means yesterday
        self.assertAlmostEqual(eta.parse_started("16:00", now), now - 23 * 3600)

    def test_fmt(self):
        self.assertEqual(eta.fmt_dur(42), "42s")
        self.assertEqual(eta.fmt_dur(14 * 60 + 10), "14 min")
        self.assertEqual(eta.fmt_dur(3 * 3600 + 5 * 60), "3h 05m")


class TestMeasureClaude(Base):
    def build(self):
        t0 = self.now - 1200
        c = Claude()
        c.prompt(t0 - 3600, "old unrelated task")
        c.tool(t0 - 3590, 5)
        c.prompt(t0, "<system-reminder>ctx</system-reminder>refactor the parser and add tests")
        t = t0 + 5
        for i in range(10):              # 10 calls, 20 s apart, 5 s each
            c.tool(t, 5)
            t += 20
        c.tool(t, 300, inp={"command": "npm run build", "description": "full build"})  # long op
        t += 310
        c.prompt(t0 + 30, "queued note", isMeta=True)           # meta: ignored
        c.prompt(t + 5, "<task-notification>bg done</task-notification>")  # system: ignored
        c.lines.append({"type": "assistant", "timestamp": iso(t + 6), "isSidechain": False,
                        "message": {"role": "assistant", "content": [
                            {"type": "tool_use", "id": "todo1", "name": "TodoWrite", "input": {"todos": [
                                {"content": "parser", "status": "completed"},
                                {"content": "tests", "status": "in_progress"},
                                {"content": "docs", "status": "pending"}]}}]}})
        c.prompt(t + 20, "<command-name>/eta</command-name><command-args></command-args>")
        c.tool(t + 22, 1, name="Skill")                          # the skill's own call: excluded
        return c.write(self.path("s.jsonl")), t0, t + 20

    def test_window_pace_and_todos(self):
        path, t0, eta_ts = self.build()
        ev = eta.read_events(path)
        m = eta.measure(ev, now=eta_ts + 5)
        self.assertTrue(m["eta_prompt"])
        self.assertEqual(m["task"], "refactor the parser and add tests")
        self.assertAlmostEqual(m["start"], t0, places=0)
        self.assertEqual(m["calls"], 12)          # 10 + build + TodoWrite; Skill excluded
        self.assertEqual(len(m["long_ops"]), 1)
        self.assertEqual(m["long_ops"][0]["desc"], "full build")
        self.assertLess(m["spc_core"], m["spc"])
        self.assertEqual(m["pace_source"], "task")
        self.assertEqual((m["todos"]["done"], m["todos"]["doing"], m["todos"]["pending"]), (1, 1, 1))
        self.assertIsNotNone(m["todos"]["steps_eta"])
        self.assertEqual([p["text"] for p in m["recent_prompts"]],
                         ["refactor the parser and add tests", "old unrelated task"])

    def test_from_earlier_prompt(self):
        path, t0, eta_ts = self.build()
        m = eta.measure(eta.read_events(path), from_n=2, now=eta_ts + 5)
        self.assertEqual(m["task"], "old unrelated task")
        # the gap before the later prompt (waiting on the user) is not active time
        self.assertLess(m["active"], m["elapsed"] - 3000)

    def test_estimate_cli_logs_prediction(self):
        path, _, _ = self.build()
        out = run(["estimate", "--transcript", path, "--calls", "30", "--note", "tests+docs"])
        self.assertIn("ETA RESULT", out)
        preds = eta.load_jsonl(eta.pred_path())
        self.assertEqual(len(preds), 1)
        p = preds[0]
        self.assertEqual(p["calls"], 30)
        self.assertAlmostEqual(p["raw_s"], 30 * p["spc"], places=3)
        self.assertLess(p["low_s"], p["est_s"])
        self.assertGreater(p["high_s"], p["est_s"])

    def test_wait_uses_core_pace(self):
        path, _, _ = self.build()
        j = json.loads(run(["estimate", "--transcript", path, "--calls", "10", "--wait-min", "5",
                            "--json", "--no-record"]))
        self.assertFalse(j["recorded"])
        self.assertEqual(eta.load_jsonl(eta.pred_path()), [])


class TestOtherHarnesses(Base):
    def test_codex(self):
        t0 = self.now - 600
        L = [{"type": "session_meta", "timestamp": iso(t0 - 5), "payload": {"cwd": "/x"}}]

        def item(ts, it):
            L.append({"type": "event_msg", "timestamp": iso(ts),
                      "payload": {"type": "item_completed", "item": it}})
        item(t0, {"type": "UserMessage", "content": [{"type": "text", "text": "fix the bug"}]})
        for i in range(6):
            item(t0 + 10 + i * 30, {"type": "AgentMessage", "content": []})
            item(t0 + 25 + i * 30, {"type": "CommandExecution", "id": f"e{i}",
                                    "command": ["/bin/zsh", "-lc", f"cmd {i}"]})
        L.append({"type": "response_item", "timestamp": iso(t0 + 200), "payload": {
            "type": "function_call", "name": "update_plan",
            "arguments": json.dumps({"plan": [{"step": "a", "status": "completed"},
                                              {"step": "b", "status": "pending"}]})}})
        item(t0 + 300, {"type": "UserMessage", "content": [{"type": "text", "text": "/eta"}]})
        path = self.path("rollout-2026-11-23T00-00-00-abc.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(l) + "\n" for l in L)
        m = eta.measure(eta.read_events(path), now=t0 + 301)
        self.assertEqual(m["task"], "fix the bug")
        self.assertEqual(m["calls"], 6)
        self.assertEqual(m["todos"]["pending"], 1)
        self.assertEqual(m["recent_prompts"][0]["text"], "fix the bug")

    def test_pi(self):
        t0 = self.now - 600
        L = [{"type": "session", "timestamp": iso(t0 - 5), "cwd": "/y"},
             {"type": "message", "timestamp": iso(t0), "message": {
                 "role": "user", "content": [{"type": "text", "text": "write the script"}]}}]
        for i in range(5):
            L.append({"type": "message", "timestamp": iso(t0 + 10 + i * 20), "message": {
                "role": "assistant", "content": [{"type": "toolCall", "id": f"c{i}", "name": "bash",
                                                  "arguments": {"command": "ls"}}]}})
            L.append({"type": "message", "timestamp": iso(t0 + 15 + i * 20), "message": {
                "role": "toolResult", "toolCallId": f"c{i}", "content": []}})
        path = os.path.join(self.tmp.name, ".pi", "s.jsonl")
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.writelines(json.dumps(l) + "\n" for l in L)
        self.assertEqual(eta.sniff_kind(path), "pi")
        m = eta.measure(eta.read_events(path), now=t0 + 200)
        self.assertEqual((m["task"], m["calls"], m["pace_source"]), ("write the script", 5, "task"))

    def test_self_reported_mode_and_done(self):
        # Any harness with no readable transcript: agent reports its own start + calls.
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            argv = ["--agent", "pi", "--started", "20m", "--calls-done", "40"]
            out = run(["measure"] + argv)
            self.assertIn("self-reported", out)
            j = json.loads(run(["estimate"] + argv + ["--calls", "20", "--json"]))
            p = eta.load_jsonl(eta.pred_path())[0]
            self.assertIsNone(p["transcript"])
            self.assertAlmostEqual(p["spc"], 30, delta=1)          # 1200 s / 40 calls
            self.assertAlmostEqual(p["raw_s"], 600, delta=20)
            out = run(["done", "--agent", "pi"])
            self.assertIn("1 ETA(s) closed", out)
            r = eta.load_jsonl(eta.res_path())[-1]
            self.assertEqual((r["id"], r["status"], r["by"]), (p["id"], "ok", "done"))
        finally:
            os.chdir(cwd)


class TestCalibration(Base):
    def fake_session(self, created, finish, next_prompt=None):
        c = Claude()
        c.prompt(created - 600, "task")
        c.prompt(created - 1, "/eta")
        t = created + 5
        while t < finish:
            c.tool(t, 2)
            t += 20
        c.say(finish)
        if next_prompt:
            c.prompt(next_prompt, "next task please")
            c.tool(next_prompt + 5, 2)
        return c.write(self.path(f"cal-{created}.jsonl"))

    def add_pred(self, created, raw, path, lo=0.6, hi=1.8):
        eta.append_jsonl(eta.pred_path(), {
            "id": f"p{created}", "created": created, "transcript": path, "kind": "claude",
            "raw_s": raw, "est_s": raw, "low_s": raw * lo, "high_s": raw * hi, "spc": 20,
            "pace_source": "task"})

    def test_resolves_and_corrects(self):
        base = self.now - 10 * 3600
        # agent says 10 min every time, work really takes 15 min -> factor ~1.5
        for i in range(5):
            created = base + i * 3600
            path = self.fake_session(created, created + 900, next_prompt=created + 1500)
            self.add_pred(created, 600, path)
        eta.resolve_pending(self.now)
        cal = eta.calibration()
        self.assertEqual(cal["n"], 5)
        self.assertAlmostEqual(cal["factor"], 1.5, delta=0.05)
        self.assertEqual(cal["hits"], 5)           # 900 s is inside 360..1080
        out = run(["stats"])
        self.assertIn("5/5", out)

    def test_pending_inconclusive_and_stale(self):
        # still running: transcript touched just now and no later prompt -> stays pending
        live = self.fake_session(self.now - 120, self.now - 2)
        self.add_pred(self.now - 120, 600, live)
        # agent stopped right after answering -> inconclusive, not used for calibration
        idle = self.fake_session(self.now - 7200, self.now - 7200 + 10, next_prompt=self.now - 3600)
        self.add_pred(self.now - 7200, 600, idle)
        eta.resolve_pending(self.now)
        res = {r["id"]: r for r in eta.load_jsonl(eta.res_path())}
        self.assertNotIn(f"p{self.now - 120}", res)
        self.assertEqual(res[f"p{self.now - 7200}"]["status"], "inconclusive")
        self.assertEqual(eta.calibration()["n"], 0)

    def test_factor_is_clamped(self):
        base = self.now - 20 * 3600
        for i in range(4):
            created = base + i * 3600
            path = self.fake_session(created, created + 3000, next_prompt=created + 3500)
            self.add_pred(created, 100, path)       # 30x under-estimate
        eta.resolve_pending(self.now)
        self.assertEqual(eta.calibration()["factor"], 3.0)


class TestDiscovery(Base):
    def test_literal_placeholder_is_ignored(self):
        path, kind, how = eta.find_transcript(session="${CLAUDE_SESSION_ID}", agent="pi",
                                              cwd="/definitely/not/a/real/dir")
        self.assertTrue(path is None or how.startswith("newest live"))


if __name__ == "__main__":
    unittest.main()
