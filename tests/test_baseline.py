"""Tests for baseline.py (history prior) and verify/replay.py. Synthetic transcripts only."""
import json
import os
import stat
import sys
import time
import unittest
from pathlib import Path

from test_eta import Base, Claude, iso

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "eta" / "scripts"))
sys.path.insert(0, str(ROOT / "verify"))
import baseline as B  # noqa: E402
import replay as R  # noqa: E402

T0 = 1_790_000_000.0   # fixed epoch in the past so files are never "live" unless we say so


class ClaudeX(Claude):
    """Claude builder with the fields the miner reads: model, stop_reason, cwd, flags."""

    def __init__(self, cwd="/work/proj"):
        super().__init__()
        self.cwd = cwd

    def _add(self, rec):
        rec.setdefault("cwd", self.cwd)
        self.lines.append(rec)

    def prompt(self, ts, text, **extra):
        self._add({"type": "user", "timestamp": iso(ts), "isSidechain": False,
                   "message": {"role": "user", "content": text}, **extra})

    def tool(self, ts, dur, name="Bash", inp=None, done=True, permission=None):
        self.n += 1
        tid = f"toolu_{self.n}"
        self._add({"type": "assistant", "timestamp": iso(ts), "isSidechain": False,
                   "message": {"role": "assistant", "model": "claude-opus-5-5", "stop_reason": "tool_use",
                               "content": [{"type": "tool_use", "id": tid, "name": name,
                                            "input": inp or {"command": f"secret-cmd-{self.n}"}}]}})
        if done:
            rec = {"type": "user", "timestamp": iso(ts + dur), "isSidechain": False,
                   "message": {"role": "user", "content": [
                       {"type": "tool_result", "tool_use_id": tid, "content": "ok"}]}}
            if permission:
                rec["permissionDecision"] = {"decision": "accept", "source": permission}
            self._add(rec)
        return tid

    def end(self, ts):
        self._add({"type": "assistant", "timestamp": iso(ts), "isSidechain": False,
                   "message": {"role": "assistant", "model": "claude-opus-5-5", "stop_reason": "end_turn",
                               "content": [{"type": "text", "text": "all done"}]}})

    def junk(self, ts):
        self.lines.append({"type": "api-request-blob", "timestamp": iso(ts), "blob": "x" * 5000})


def codex_lines(t0, source="vscode", abort=False):
    rows = [
        {"timestamp": iso(t0 - 1), "type": "session_meta",
         "payload": {"id": "s1", "cwd": "/work/cx", "source": source}},
        {"timestamp": iso(t0 - 1), "type": "turn_context", "payload": {"model": "gpt-6.2-astra"}},
        {"timestamp": iso(t0), "type": "event_msg", "payload": {"type": "task_started"}},
        {"timestamp": iso(t0), "type": "event_msg", "payload": {"type": "item_completed", "item": {
            "type": "UserMessage", "content": [{"type": "text", "text": "refactor the parser"}]}}},
        {"timestamp": iso(t0 + 5), "type": "response_item",
         "payload": {"type": "function_call", "name": "exec", "call_id": "c1", "arguments": "{}"}},
        {"timestamp": iso(t0 + 65), "type": "response_item",
         "payload": {"type": "function_call_output", "call_id": "c1", "output": "ok"}},
        {"timestamp": iso(t0 + 70), "type": "event_msg",
         "payload": {"type": "item_completed", "item": {"type": "AgentMessage"}}},
    ]
    rows.append({"timestamp": iso(t0 + 71), "type": "event_msg",
                 "payload": {"type": "turn_aborted" if abort else "task_complete"}})
    return rows


def write_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.utime(path, (T0 + 86400, T0 + 86400))
    return str(path)


def synth_task(wall, t0=0.0, h="claude", calls=None, end="done", origin="human"):
    calls = calls if calls is not None else [[1.0, min(wall, 5.0), "s"]]
    return {"k": f"{h}:{t0}", "h": h, "m": "claude-opus-5-5", "o": origin, "t0": t0, "wall": wall,
            "act": wall, "uw": 0, "end": end, "n": len(calls), "eta": 0, "q": [], "drop": 0,
            "c": calls, "b": [], "a": [0, int(wall)]}


class TestClassify(unittest.TestCase):
    def test_prompt_classes(self):
        cases = {
            "fix the flaky test": "human", "/eta": "eta", "how long left?": "eta",
            "<task-notification><task-id>x</task-id></task-notification>": "notify",
            "[T3 Code: the agent in T3 thread \"t\" finished]": "notify",
            "Background command completed (exit 0): make": "notify",
            "[T3 Code: scheduled task \"daily\"]": "scheduled",
            "You are in a side chat beside the main thread.": "side",
            "Delegated task node:abc do the thing": "notify",
            "[Request interrupted by user]": "interrupt",
            "/compact": "meta", "/model opus": "human", "/code-review high": "human",
            "This session is being continued from a previous conversation": "meta",
            "": None,
        }
        for text, want in cases.items():
            self.assertEqual(B.classify_prompt(text), want, text)
        self.assertEqual(B.classify_prompt("anything", notify=True), "notify")

    def test_tool_classes_and_family(self):
        self.assertEqual([B.tool_class(n) for n in ("Agent", "mcp__t3-code__delegate_task", "spawn_agent",
                                                    "mcp__t3-code__t3_background_wait", "wait", "AskUserQuestion",
                                                    "Bash", "exec", "Read")], list("dddwwasso"))
        self.assertEqual(B.model_family("claude-opus-5-5"), "opus")
        self.assertEqual(B.model_family("gpt-6.2-astra"), "gpt-6.2")
        self.assertIsNone(B.model_family("<synthetic>"))


class TestClaudeSplit(Base):
    def mine(self, c, name="s.jsonl", live=False):
        p = c.write(self.path(name))
        if not live:
            os.utime(p, (T0 + 86400, T0 + 86400))
        return B.mine_file(p, "claude", now=(time.time() if live else T0 + 86400 + 2 * B.STALE_S))["tasks"]

    def test_basic_task_with_mid_turn_eta_and_notification(self):
        c = ClaudeX()
        c.prompt(T0, "build the thing")
        c.junk(T0 + 1)
        c.tool(T0 + 10, 50)
        c.prompt(T0 + 70, "/eta")                       # mid-turn ask: does not split
        c.tool(T0 + 80, 30, inp={"command": "sleep 99", "run_in_background": True})
        c.end(T0 + 120)
        c.prompt(T0 + 600, "<task-notification>bg done</task-notification>",
                 turnOrigin="task_notification")          # wakes within the window
        c.tool(T0 + 610, 20)
        c.end(T0 + 640)
        c.prompt(T0 + 2000, "now write docs")
        c.tool(T0 + 2010, 10, name="Edit")
        c.end(T0 + 2030)
        tasks = self.mine(c)
        self.assertEqual(len(tasks), 2)
        a, b = tasks
        self.assertEqual((a["end"], a["n"], a["wall"], a["eta"], a["q"]), ("done", 3, 640.0, 1, [70.0]))
        self.assertEqual((a["m"], a["cwd"], a["h"], a["o"]), ("claude-opus-5-5", "/work/proj", "claude", "human"))
        self.assertEqual(a["b"], [[80.0, 600.0]])
        self.assertEqual((b["wall"], b["n"], b["end"]), (30.0, 1, "done"))
        # active time caps the 480 s gap at GAP_CAP (it is under it) and counts everything
        self.assertEqual(a["act"], 640.0)

    def test_end_kinds(self):
        c = ClaudeX()
        c.prompt(T0, "one")
        c.tool(T0 + 5, 10)
        c.prompt(T0 + 20, "[Request interrupted by user]")
        c.prompt(T0 + 30, "two")
        c.tool(T0 + 35, 10)                       # still mid-turn when the next prompt arrives
        c.prompt(T0 + 60, "three")
        c.tool(T0 + 65, 10, done=False)          # transcript just stops mid-tool
        kinds = [t["end"] for t in self.mine(c)]
        self.assertEqual(kinds, ["interrupted", "steered", "cut"])
        self.assertEqual([t["end"] for t in self.mine(c, "live.jsonl", live=True)][-1], "live")

    def test_late_wakeup_dropped_and_eta_after_end_closes(self):
        c = ClaudeX()
        c.prompt(T0, "task")
        c.tool(T0 + 5, 10)
        c.end(T0 + 20)
        c.prompt(T0 + 5000, "[T3 Code: the agent in T3 thread \"x\" sent a message]")
        c.tool(T0 + 5010, 10)
        c.end(T0 + 5030)
        c.prompt(T0 + 6000, "second")
        c.tool(T0 + 6005, 10)
        c.end(T0 + 6020)
        c.prompt(T0 + 6300, "/eta")             # after the turn ended: the answer is not task time
        c.tool(T0 + 6310, 5)
        c.end(T0 + 6320)
        a, b = self.mine(c)
        self.assertEqual((a["wall"], a["n"], a["drop"]), (20.0, 1, 1))
        self.assertEqual((b["wall"], b["n"], b["q"], b["drop"]), (20.0, 1, [300.0], 1))

    def test_user_waits_meta_and_chat(self):
        c = ClaudeX()
        c.prompt(T0, "<command-name>/compact</command-name>")
        c.prompt(T0 + 1, "ask me first")
        c.tool(T0 + 5, 100, name="AskUserQuestion", inp={"questions": []})
        c.tool(T0 + 110, 40, permission="user_temporary")
        c.end(T0 + 160)
        c.prompt(T0 + 200, "hi there")
        c.end(T0 + 205)
        a, chat = self.mine(c)
        self.assertEqual((a["t0"], a["uw"], a["n"]), (round(T0 + 1, 3), 140.0, 2))
        self.assertEqual((chat["n"], chat["end"]), (0, "done"))
        self.assertFalse(B.fit_eligible(chat))

    def test_sidechain_records_ignored(self):
        c = ClaudeX()
        c.prompt(T0, "go")
        c.tool(T0 + 5, 10)
        c.lines.append({"type": "assistant", "timestamp": iso(T0 + 500), "isSidechain": True,
                        "message": {"role": "assistant", "content": []}})
        c.end(T0 + 20)
        self.assertEqual(self.mine(c)[0]["wall"], 20.0)


class TestOtherHarnesses(Base):
    def test_codex(self):
        p = write_rows(self.path("rollout-a.jsonl"), codex_lines(T0))
        (t,) = B.mine_file(p, "codex", now=T0 + 90000)["tasks"]
        self.assertEqual((t["h"], t["m"], t["cwd"], t["end"], t["n"], t["wall"]),
                         ("codex", "gpt-6.2-astra", "/work/cx", "done", 1, 71.0))
        self.assertEqual(t["c"], [[5.0, 65.0, "s"]])
        p = write_rows(self.path("rollout-b.jsonl"), codex_lines(T0, abort=True))
        self.assertEqual(B.mine_file(p, "codex", now=T0 + 90000)["tasks"][0]["end"], "interrupted")
        p = write_rows(self.path("rollout-c.jsonl"), codex_lines(T0, source={"subagent": {"other": "guardian"}}))
        r = B.mine_file(p, "codex", now=T0 + 90000)
        self.assertEqual((r["skip"], r["tasks"]), ("codex subagent session", []))

    def test_pi(self):
        rows = [{"type": "session", "cwd": "/work/pi", "timestamp": iso(T0 - 1)},
                {"type": "message", "timestamp": iso(T0), "message": {"role": "user", "content": "list files"}},
                {"type": "message", "timestamp": iso(T0 + 3), "message": {
                    "role": "assistant", "model": "qwen3", "stopReason": "toolUse",
                    "content": [{"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "ls"}}]}},
                {"type": "message", "timestamp": iso(T0 + 9), "message": {"role": "toolResult", "toolCallId": "t1"}},
                {"type": "message", "timestamp": iso(T0 + 12), "message": {
                    "role": "assistant", "model": "qwen3", "stopReason": "stop", "content": []}}]
        p = write_rows(self.path("pi.jsonl"), rows)
        (t,) = B.mine_file(p, "pi", now=T0 + 90000)["tasks"]
        self.assertEqual((t["end"], t["n"], t["wall"], t["cwd"], t["m"]), ("done", 1, 12.0, "/work/pi", "qwen3"))


class TestCache(Base):
    def test_build_is_private_incremental_deduped_and_text_free(self):
        c = ClaudeX()
        c.prompt(T0, "a very private prompt body")
        c.tool(T0 + 5, 10)
        c.end(T0 + 20)
        p1 = c.write(self.path("one.jsonl"))
        p2 = c.write(self.path("fork.jsonl"))          # a fork copies the same records
        for p in (p1, p2):
            os.utime(p, (T0 + 86400, T0 + 86400))
        files = [(p1, "claude"), (p2, "claude")]
        recs = B.build(files=files, now=T0 + 90000)
        cache = B.cache_path()
        self.assertTrue(cache.startswith(os.environ["ETA_HOME"]))
        self.assertEqual(stat.S_IMODE(os.stat(cache).st_mode), 0o600)
        self.assertEqual(len(B.tasks_from_records(recs, T0 + 90000)), 1)
        text = Path(cache).read_text()
        self.assertNotIn("private prompt", text)
        self.assertNotIn("secret-cmd", text)
        calls = []
        orig = B.mine_file
        B.mine_file = lambda *a, **k: calls.append(a) or orig(*a, **k)
        try:
            B.build(files=files, now=T0 + 90000)
            self.assertEqual(calls, [])
            with open(p1, "a") as f:
                f.write("\n")
            B.build(files=files, now=T0 + 90000)
            self.assertEqual([a[0] for a in calls], [p1])
        finally:
            B.mine_file = orig

    def test_stale_live_task_becomes_cut(self):
        rec = {"mt": T0, "tasks": [synth_task(100, end="live")]}
        self.assertEqual(B.tasks_from_records([rec], now=T0 + 60)[0]["end"], "live")
        self.assertEqual(B.tasks_from_records([rec], now=T0 + 3600)[0]["end"], "cut")


class TestStateAndEstimator(Base):
    def test_state_at(self):
        t = synth_task(600, calls=[[10, 20, "s"], [30, 400, "d"], [40, -1, "s"]])
        t["b"] = [[50, 300]]
        st = B.state_at(t, 100)
        self.assertEqual((st["calls"], st["delegates_open"], st["running_cls"]), (3, 2, "d"))
        self.assertAlmostEqual(st["running_s"], 70)
        self.assertEqual(B.state_at(t, 5)["calls"], 0)
        self.assertEqual(B.features(st)["running"], "deleg")

    def test_quantiles_survival_and_never_refuses(self):
        tasks = [synth_task(w, t0=i) for i, w in enumerate([60, 120, 180, 240, 300, 600, 900, 1200, 2400, 3600] * 5)]
        b = B.Baseline(tasks)
        r = b.remaining_given({"elapsed_s": 500})
        self.assertLessEqual(r["p10"], r["p50"])
        self.assertLessEqual(r["p50"], r["p80"])
        self.assertLessEqual(r["p80"], r["p90"])
        self.assertEqual(r["n_survivors"], 25)        # tasks longer than 500 s
        self.assertTrue(100 <= r["p50"] <= 1900, r)
        far = b.remaining_given({"elapsed_s": 50000})  # longer than anything seen
        self.assertEqual(far["n"], 0)
        self.assertGreater(far["p50"], 0)
        empty = B.Baseline([]).remaining_given({"elapsed_s": 60})
        self.assertGreater(empty["p90"], empty["p10"])
        # thinner evidence gives a wider relative range
        few = B.Baseline(tasks[:10]).remaining_given({"elapsed_s": 500})
        self.assertGreater(few["p90"] / max(few["p10"], 1), r["p90"] / max(r["p10"], 1))

    def test_features_narrow_and_shrink(self):
        tasks = [synth_task(100 + i, t0=i, h="claude") for i in range(60)]
        tasks += [synth_task(5000 + i, t0=100 + i, h="codex") for i in range(60)]
        b = B.Baseline(tasks, feats=("harness",))
        cx = b.remaining_given({"elapsed_s": 60, "harness": "codex"})
        cl = b.remaining_given({"elapsed_s": 60, "harness": "claude"})
        self.assertGreater(cx["p50"], 10 * cl["p50"])
        self.assertIn("harness", cx["basis"])
        unknown = b.remaining_given({"elapsed_s": 60, "harness": None})
        self.assertEqual(unknown["n"], 120)

    def test_state_from_live_transcript(self):
        c = ClaudeX()
        now = time.time()
        c.prompt(now - 300, "long job")
        c.tool(now - 290, 10)
        c.tool(now - 200, 0, name="Agent", inp={"prompt": "x"}, done=False)
        p = c.write(self.path("live.jsonl"))
        st = B.state_from_transcript(p, "claude", now=now)
        self.assertAlmostEqual(st["elapsed_s"], 300, delta=1)
        self.assertEqual((st["calls"], st["delegates_open"], st["running_cls"]), (2, 1, "d"))
        r = B.remaining_given(st, baseline=B.Baseline([synth_task(900, t0=i) for i in range(10)]))
        self.assertGreater(r["p50"], 0)

    def test_cli_query_with_temp_home(self):
        import io
        from contextlib import redirect_stdout
        os.environ["HOME_BACKUP"] = os.environ.get("HOME", "")
        B._CACHED.clear()
        orig = B.source_files
        B.source_files = lambda: []
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                B.main(["query", "--elapsed-min", "20", "--json"])
            out = json.loads(buf.getvalue())
            self.assertGreater(out["p90"], out["p10"])
            self.assertTrue(os.path.exists(B.cache_path()))
        finally:
            B.source_files = orig
            B._CACHED.clear()


class TestReplay(Base):
    def test_folds_do_not_leak_and_scoring(self):
        tasks = [synth_task(100 + (i % 7) * 50, t0=i * 1000) for i in range(100)]
        for train, test, cut in R.folds(tasks, 3):
            self.assertTrue(test)
            self.assertTrue(all(t["t0"] + t["wall"] < cut for t in train))
            self.assertTrue(all(t["t0"] >= cut for t in test))
        m = R.score([((50, 100, 200), 100), ((50, 100, 200), 400), (None, 10)])
        self.assertAlmostEqual(m["answer"], 2 / 3)
        self.assertAlmostEqual(m["cov"], 0.5)
        self.assertAlmostEqual(m["w2x"], 0.5)

    def test_register_hook_and_modes(self):
        R.register_fn("always-60", lambda st: (30, 60, 120))
        try:
            tasks = [synth_task(700 + i * 10, t0=i * 1000) for i in range(40)]
            fl = R.folds(tasks, 2)
            preds = R.run_estimators(fl, "late", ["always-60", "baseline"])
            self.assertEqual(len(preds["always-60"]), len(preds["baseline"]))
            self.assertTrue(all(st["elapsed_s"] >= R.LATE_FROM for _, _, st, _ in preds["baseline"]))
            self.assertIsNotNone(R.closer_than(preds["always-60"], preds["baseline"]))
        finally:
            R.ESTIMATORS.pop("always-60", None)


if __name__ == "__main__":
    unittest.main()
