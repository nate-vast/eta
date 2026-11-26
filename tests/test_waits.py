import json
import os
import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/eta/scripts"))
import waits  # noqa: E402

T0 = 1_800_000_000.0


def iso(t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def claude_line(kind, t, content, **extra):
    row = {"type": kind, "timestamp": iso(t), "sessionId": "s1", "cwd": "/tmp/proj",
           "message": {"role": kind, "content": content}}
    row.update(extra)
    return json.dumps(row)


def tool_use(tid, name, inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def tool_result(tid, text="ok"):
    return {"type": "tool_result", "tool_use_id": tid, "content": text}


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def sample_claude(path):
    """A parent that delegates three agents and a T3 task, and runs a background command."""
    lines = [
        claude_line("user", T0, "build the museum and research the venue"),
        claude_line("assistant", T0 + 5, [tool_use("a1", "Agent", {"description": "Research venue",
                                                                       "prompt": "x", "run_in_background": True})]),
        claude_line("user", T0 + 6, [tool_result("a1", "Async agent launched")],
                    toolUseResult={"isAsync": True, "status": "async_launched", "agentId": "ag1",
                                   "resolvedModel": "claude-opus-5-5"}),
        claude_line("assistant", T0 + 10, [tool_use("t1", "mcp__t3-code__delegate_task",
                                                    {"title": "Museum redesign", "task": "x",
                                                     "target": {"providerInstanceId": "claudeAgent",
                                                                "model": "claude-fable-5-2"}})]),
        claude_line("user", T0 + 11, [tool_result("t1", '{"taskId":"node:delegated-task:command%3Aabc"}')]),
        claude_line("assistant", T0 + 20, [tool_use("b1", "Bash", {"command": "python seg.py 160 > seg.log 2>&1",
                                                                    "description": "Segment frames"})]),
        claude_line("user", T0 + 620, [tool_result("b1", "moved to background. Output is being written to: "
                                                       "/nonexistent/tasks/bg1.output")],
                    toolUseResult={"backgroundTaskId": "bg1", "stdout": ""}),
        claude_line("assistant", T0 + 630, [{"type": "text", "text": "waiting"}]),
        claude_line("user", T0 + 900, "<task-notification>\n<task-id>ag1</task-id>\n<tool-use-id>a1</tool-use-id>\n"
                                     "<status>completed</status>\n</task-notification>",
                    turnOrigin="task_notification"),
        claude_line("assistant", T0 + 905, [{"type": "text", "text": "got research"}]),
        claude_line("user", T0 + 2000, "Delegated task node:delegated-task:command%3Aabc (Museum redesign) completed"),
        claude_line("assistant", T0 + 2010, [tool_use("e1", "Edit", {"file_path": "a"})]),
        claude_line("user", T0 + 2011, [tool_result("e1")]),
        claude_line("assistant", T0 + 2300, [{"type": "text", "text": "done"}]),
        claude_line("user", T0 + 2400, "<command-message>eta</command-message>\n<command-name>/eta</command-name>"),
        claude_line("user", T0 + 3000, "next task please"),
    ]
    write_lines(path, lines)


def hist_delegates(model, durs, source="agent"):
    return [dict(source=source, harness="claude", model=model, start=T0 - 10 * 86400 + i, dur=d, bg=1)
            for i, d in enumerate(durs)]


class ProgressParsing(unittest.TestCase):
    def test_tqdm_with_series(self):
        text = ("Fetching 144 files:  38%|###       | 55/144 [06:19<09:21,  6.31s/it]\r"
                "Fetching 144 files:  67%|######    | 97/144 [10:57<04:35,  5.87s/it]\r"
                "Fetching 144 files:  85%|########  | 122/144 [13:45<03:00,  8.20s/it]")
        p = waits.parse_progress(text)
        self.assertEqual((p["done"], p["total"], p["kind"]), (122, 144, "tqdm"))
        self.assertEqual(p["remaining_s"], 180)
        self.assertEqual(len(p["series"]), 3)
        est = waits.progress_estimate(p, 900)
        self.assertGreater(est["p50_s"], 0)
        self.assertLessEqual(est["low_s"], est["p50_s"])
        self.assertGreaterEqual(est["high_s"], est["p50_s"])

    def test_tqdm_bytes(self):
        p = waits.parse_progress("model.safetensors:  45%|####5     | 1.21G/2.69G [00:30<00:37, 40.1MB/s]")
        self.assertEqual(p["unit"], "bytes")
        self.assertAlmostEqual(p["total"], 2.69e9)

    def test_k_of_n_words_and_brackets(self):
        self.assertEqual(waits.parse_progress("rendered frame 40/160\nrendered frame 41/160")["done"], 41)
        self.assertEqual(waits.parse_progress("[12/50] compiling foo.c")["total"], 50)
        self.assertEqual(waits.parse_progress("Processing item 3 of 9")["done"], 3)

    def test_dates_and_ratios_are_not_progress(self):
        self.assertIsNone(waits.parse_progress("built on 2026/11/25 at noon"))
        self.assertIsNone(waits.parse_progress("hello world\nno numbers here"))

    def test_percent_and_curl(self):
        self.assertEqual(waits.parse_progress("Uploading... 42%")["done"], 42)
        row = ("  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n"
               " 45 1000M   45  450M    0     0  10.0M      0  0:01:40  0:00:45  0:00:55 10.2M")
        p = waits.parse_progress(row)
        self.assertEqual(p["kind"], "curl")
        self.assertAlmostEqual(p["done"], 450e6)
        self.assertEqual(p["remaining_s"], 55)

    def test_bytes_pair(self):
        p = waits.parse_progress("downloaded 1.5GB/6GB")
        self.assertEqual(p["unit"], "bytes")
        est = waits.progress_estimate(p, 300)
        self.assertAlmostEqual(est["p50_s"], 900, delta=1)

    def test_recent_rate_preferred_over_average(self):
        p = dict(done=100, total=200, unit="items", kind="k/N")
        est = waits.progress_estimate(p, 1000, prev=(T0 - 100, 90), now=T0)
        self.assertAlmostEqual(est["p50_s"], 1000)  # 10 items per 100 s recently
        self.assertIn("since last check", est["basis"])

    def test_finished_batch(self):
        self.assertEqual(waits.progress_estimate(dict(done=5, total=5, unit="items"), 10)["p50_s"], 0)


class Estimators(unittest.TestCase):
    def test_conditional_uses_only_runs_still_going(self):
        self.assertEqual(waits.conditional([10, 60, 120], 50), [10, 70])

    def test_model_level_then_fallbacks(self):
        h = waits.History(hist_delegates("claude-opus-5-5", [600 + 60 * i for i in range(20)])
                          + hist_delegates("claude-fable-5-2", [3000 + 100 * i for i in range(20)], "t3"))
        est = waits.estimate_delegate("claude-fable-5-2", "t3", 600, h)
        self.assertIn("same model", est["basis"])
        self.assertGreater(est["p50_s"], 2000)
        est = waits.estimate_delegate("claude-fable-6-0", "t3", 600, h)
        self.assertIn("same model family", est["basis"])
        est = waits.estimate_delegate("unknown-model", "codex", 100, h)
        self.assertIn("all delegates", est["basis"])

    def test_never_refuses(self):
        est = waits.estimate_delegate("m", "agent", 7200, waits.History())
        self.assertGreater(est["p50_s"], 0)
        self.assertLess(est["low_s"], est["p50_s"])
        self.assertGreater(est["high_s"], est["p50_s"])
        span = waits.Span("x", "command", "shell", T0, "make")
        est = waits.estimate_command(span, 30, waits.History(), "claude")
        self.assertGreater(est["high_s"], est["p50_s"])

    def test_exact_command_history(self):
        span = waits.Span("x", "command", "shell", T0, "make test")
        span.chash = waits.command_hash("make  test")
        cmds = [dict(harness="claude", end=T0 - 100, dur=d, bg=0, hash=span.chash, status="completed")
                for d in (100, 110, 120, 130)]
        est = waits.estimate_command(span, 50, waits.History(commands=cmds), "claude")
        self.assertIn("exact command", est["basis"])
        self.assertAlmostEqual(est["p50_s"], 65, delta=1)

    def test_combine(self):
        a = dict(low_s=60, p50_s=120, high_s=300)
        b = dict(low_s=100, p50_s=200, high_s=600)
        lo, p50, hi = waits.combine([a, b])
        self.assertGreaterEqual(p50, 200)
        self.assertGreaterEqual(hi, 600)
        self.assertLessEqual(lo, p50)
        lo, p50, hi = waits.combine([a, b], "sequential")
        self.assertEqual(p50, 320)
        self.assertLess(hi, 300 + 600)
        self.assertEqual(waits.combine([]), (0.0, 0.0, 0.0))

    def test_parent_tail_conditions_on_wait_length(self):
        tails = ([dict(harness="claude", kind="delegate", n=1, wait=120, tail=60, end=T0 - 1000, last=1)] * 15
                 + [dict(harness="claude", kind="delegate", n=1, wait=3600, tail=1200, end=T0 - 1000, last=1)] * 15)
        h = waits.History(tails=tails)
        self.assertEqual(waits.parent_tail("claude", 1, 4000, history=h)["p50_s"], 1200)
        self.assertEqual(waits.parent_tail("claude", 1, 100, history=h)["p50_s"], 60)
        self.assertEqual(waits.parent_tail("claude", 1, 100, history=h, final_only=True)["p50_s"], 60)
        not_last = waits.History(tails=[dict(t, last=0) for t in tails])
        self.assertEqual(waits.parent_tail("claude", 1, 100, history=not_last, final_only=True)["n"], 0)
        self.assertEqual(waits.parent_tail(history=waits.History())["n"], 0)

    def test_history_cut_avoids_future(self):
        h = waits.History(hist_delegates("m", [100]), [], [])
        self.assertEqual(len(h.before(T0 - 20 * 86400).delegates), 0)

    def test_model_family(self):
        self.assertEqual(waits.model_family("claude-opus-5-5"), "claude-opus")
        self.assertEqual(waits.model_family("gpt-6.2-astra"), "gpt-astra")
        self.assertIsNone(waits.model_family(None))


class Transcripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.env = mock.patch.dict(os.environ, {"ETA_HOME": os.path.join(self.dir, "eta"),
                                                "T3_STATE_DB": os.path.join(self.dir, "none.sqlite")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_claude_spans_and_prompts(self):
        path = os.path.join(self.dir, "s1.jsonl")
        sample_claude(path)
        tl = waits.scan_claude(path)
        spans = {s.id: s for s in tl.spans}
        self.assertEqual(spans["a1"].end, T0 + 900)
        self.assertEqual(spans["a1"].model, "claude-opus-5-5")
        self.assertEqual(spans["t1"].end, T0 + 2000)
        self.assertEqual(spans["t1"].model, "claude-fable-5-2")
        self.assertIsNone(spans["b1"].end)
        self.assertTrue(spans["b1"].bg)
        self.assertEqual(len(tl.prompts), 3)          # the two tasks and the /eta ask
        self.assertTrue(tl.prompts[1][1])             # /eta flagged
        tails = waits.tails_from_timeline(tl)
        dele = [t for t in tails if t[0] == "delegate"]
        self.assertEqual(len(dele), 1)
        kind, n, wait, tail, end, last, age = dele[0]
        self.assertEqual((n, end, last, age), (2, T0 + 2000, 1, 5))
        self.assertEqual(tail, 300)                    # last parent activity before the next prompt

    def test_open_waits_cut_mid_session(self):
        path = os.path.join(self.dir, "s1.jsonl")
        sample_claude(path)
        log = os.path.join(self.dir, "seg.log")
        with open(log, "w") as f:
            f.write("frame 40/160\nframe 80/160\n")
        with mock.patch.object(waits, "output_sources", return_value=[log]):
            ws = waits.open_waits(path, "claude", now=T0 + 700, history=waits.History())
        kinds = sorted(w["kind"] for w in ws)
        self.assertEqual(kinds, ["batch", "delegate", "delegate"])
        batch = next(w for w in ws if w["kind"] == "batch")
        self.assertAlmostEqual(batch["p50_s"], 680, delta=5)   # 80 frames in 680 s, 80 left
        for w in ws:
            self.assertIsNotNone(w["p50_s"])
            self.assertLessEqual(w["low_s"], w["p50_s"])
            self.assertLessEqual(w["p50_s"], w["high_s"])
        ws = waits.open_waits(path, "claude", now=T0 + 2500, history=waits.History())
        self.assertEqual([w["kind"] for w in ws], ["command"])

    def test_progress_memory_is_private(self):
        waits._remember_progress("k", T0, 5)
        prev = waits._remember_progress("k", T0 + 10, 7)
        self.assertEqual(prev, (T0, 5))
        mode = stat.S_IMODE(os.stat(waits.progress_path()).st_mode)
        self.assertEqual(mode, 0o600)

    def test_services_are_not_waits(self):
        s = waits.Span("x", "command", "shell", T0, "Serve the museum on port 8791")
        self.assertTrue(waits.is_service(s))
        s = waits.Span("x", "command", "shell", T0, "Segment frames")
        s.extra["command"] = "python seg.py"
        self.assertFalse(waits.is_service(s))

    def test_codex_subagents_and_cells(self):
        path = os.path.join(self.dir, "rollout-x.jsonl")

        def ev(t, item):
            return json.dumps({"timestamp": iso(t), "type": "event_msg",
                               "payload": {"type": "item_completed", "item": item}})
        lines = [
            json.dumps({"timestamp": iso(T0), "type": "session_meta", "payload": {"id": "c1", "cwd": "/tmp"}}),
            ev(T0 + 1, {"type": "UserMessage", "content": [{"type": "text", "text": "do it"}]}),
            ev(T0 + 2, {"type": "SubAgentActivity", "model": "gpt-6.2-astra", "kind": "started",
                        "id": "call_1", "agent_thread_id": "th1", "agent_path": "/root/worker"}),
            ev(T0 + 3, {"type": "SubAgentActivity", "model": "gpt-6.2-astra", "kind": "started",
                        "id": "call_2", "agent_thread_id": "th2", "agent_path": "/root/other"}),
            ev(T0 + 50, {"type": "SubAgentActivity", "model": None, "kind": "completed",
                         "id": "done1", "agent_thread_id": "th1", "agent_path": "/root/worker"}),
            json.dumps({"timestamp": iso(T0 + 60), "type": "response_item",
                        "payload": {"type": "custom_tool_call", "name": "exec", "call_id": "c9",
                                    "input": 'text(await tools.exec_command({cmd:"make build"}))'}}),
            json.dumps({"timestamp": iso(T0 + 61), "type": "response_item",
                        "payload": {"type": "custom_tool_call_output", "call_id": "c9",
                                    "output": "Script running with cell ID g1:3\nWall time 1.0 seconds"}}),
            ev(T0 + 70, {"type": "CommandExecution", "id": "e1", "command": ["/bin/zsh", "-lc", "ls"],
                         "exit_code": 0, "duration": {"secs": 2, "nanos": 0}}),
        ]
        write_lines(path, lines)
        tl = waits.scan_codex(path)
        open_ = tl.open_spans()
        self.assertEqual(sorted(s.label for s in open_), ["make build", "other"])
        done = [s for s in tl.spans if s.what == "delegate" and s.end]
        self.assertEqual(done[0].end - done[0].start, 48)
        ws = waits.open_waits(path, "codex", now=T0 + 100, history=waits.History())
        self.assertEqual(sorted(w["kind"] for w in ws), ["command", "delegate"])

    def test_pi_open_bash(self):
        path = os.path.join(self.dir, "pi.jsonl")
        lines = [json.dumps({"type": "session", "id": "p1", "cwd": "/tmp", "timestamp": iso(T0)}),
                 json.dumps({"type": "message", "timestamp": iso(T0 + 1),
                             "message": {"role": "assistant", "content": [
                                 {"type": "toolCall", "id": "x1", "name": "bash",
                                  "arguments": {"command": "pytest -q"}}]}})]
        write_lines(path, lines)
        ws = waits.open_waits(path, "pi", now=T0 + 40, history=waits.History())
        self.assertEqual(ws[0]["label"], "pytest -q")
        self.assertAlmostEqual(ws[0]["elapsed_s"], 39)

    def test_t3_database_fills_missing_completion(self):
        db = os.path.join(self.dir, "state.sqlite")
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE orchestration_v2_projection_subagents (subagent_id TEXT, thread_id TEXT,
            provider TEXT, origin TEXT, status TEXT, started_at TEXT, completed_at TEXT, child_thread_id TEXT,
            payload_json TEXT)""")
        con.execute("""CREATE TABLE orchestration_v2_projection_provider_threads (thread_id TEXT, payload_json TEXT)""")
        con.execute("INSERT INTO orchestration_v2_projection_subagents VALUES (?,?,?,?,?,?,?,?,?)",
                    ("node:delegated-task:command%3Aabc", "p", "claudeAgent", "app_owned", "completed",
                     iso(T0 + 10), iso(T0 + 1500), "child", json.dumps({"model": "claude-fable-5-2"})))
        con.commit()
        con.close()
        path = os.path.join(self.dir, "s1.jsonl")
        sample_claude(path)
        lines = [ln for ln in open(path).read().splitlines() if "Delegated task" not in ln]
        write_lines(path, lines)
        with mock.patch.dict(os.environ, {"T3_STATE_DB": db}):
            tl = waits.scan(path, "claude")
            span = next(s for s in tl.spans if s.via == "t3")
            self.assertEqual(span.end, T0 + 1500)
            rows = waits.t3_delegate_rows()
            self.assertEqual(rows[0]["model"], "claude-fable-5-2")
            ws = waits.open_waits(path, "claude", now=T0 + 1000, history=waits.History())
            self.assertIn("delegate", [w["kind"] for w in ws])
            self.assertTrue(any(w["model"] == "claude-fable-5-2" for w in ws))

    def test_mine_is_incremental_and_private(self):
        home = os.path.join(self.dir, "home")
        proj = os.path.join(home, ".claude/projects/-tmp-proj")
        os.makedirs(proj)
        sample_claude(os.path.join(proj, "s1.jsonl"))
        with mock.patch.dict(os.environ, {"HOME": home, "CODEX_HOME": os.path.join(home, ".codex"),
                                          "PI_CODING_AGENT_DIR": os.path.join(home, ".pi")}):
            cache = waits.mine(now=T0 + 10 * 86400)
            self.assertEqual(cache["pending"], 0)
            self.assertEqual(stat.S_IMODE(os.stat(waits.cache_path()).st_mode), 0o600)
            raw = open(waits.cache_path()).read()
            self.assertNotIn("museum", raw.lower())
            self.assertNotIn("Research", raw)
            with mock.patch.object(waits, "file_records", side_effect=AssertionError("reparsed")):
                waits.mine(now=T0 + 10 * 86400)
            h = waits.load_history(budget_s=0)
            self.assertEqual(len(h.delegates), 1)      # the T3 run comes from the database only
            self.assertEqual(len(h.tails), 1)


if __name__ == "__main__":
    unittest.main()
