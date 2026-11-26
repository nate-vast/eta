#!/usr/bin/env python3
"""Count-versus-minutes replay: can a model count what is left better than it can state time?

    python3 verify/llm_replay.py prepare --n 60 --out verify/llm_cuts/ [--seed S]
    python3 verify/llm_replay.py score --answers verify/llm_cuts/answers/ [--module PATH]

prepare: samples finished tasks from the /eta task index (~/.eta/turns.jsonl, built by
skills/eta/scripts/baseline.py), cuts each at one moment, and writes
  cut-XXX.md       what the working agent had seen up to that moment, nothing after it
  key.json         the answers (actual remaining time, history prior, waits forecast);
                   never show it to an estimating model
  INSTRUCTIONS.md  the exact prompt and JSON answer format for an estimating model
Everything under ~/.claude, ~/.codex, ~/.t3 and ~/.eta is only read. The output
directory holds private transcript text and is git-ignored.

score: reads one JSON answer per cut (a file may also hold a list) and scores the
model's minutes guess, count x measured pace, count x pace + waits forecast, the history
prior alone, log-space blends of count and prior, and optionally an external combiner,
with the metrics of verify/replay.py. Writes verify/LLM_REPLAY_RESULTS.md (numbers only).

Combiner hook (--module PATH): the module may define
    llm_replay_estimate(key_entry, answer) -> (low, mid, high[, parts]) in seconds, or None
or a function `combine` (or the name given by --func) with the signature
    fn(state, waits_forecast, own_steps, pace) -> (low, mid, high, parts)
state is baseline.state_at() at the cut plus "prior" (the leak-free prior quantiles);
waits_forecast is the leak-free waits.forecast() dict (or None) plus "required_waits" from
the answer; own_steps is the model's own_calls_left; pace is seconds per tool call so far.
While it runs, baseline.remaining_given and baseline.default_baseline are patched to return
the leak-free prior, so a combiner that asks the history itself cannot see the future.
Python 3.9+, stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "skills", "eta", "scripts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
import baseline as B  # noqa: E402
import replay as R  # noqa: E402

FLOOR = R.FLOOR
MIN_REMAIN = 60.0        # skip cuts with less than this actually remaining
MIN_ELAPSED = 30.0
MAX_USER_WAIT_SHARE = 0.2  # skip tasks that spent more than this share waiting on the user
MAX_IDLE_S = 600          # skip tasks with a longer stretch of no activity and nothing running
MIN_TRAIN = 300         # fit-eligible tasks that must have finished before a sampled task
RECENCY_HALF_LIFE_D = 14.0
CHAR_BUDGET = 42000      # about 12k tokens of mixed prose, code and logs
HEAD_BUDGET = 7000
TAIL_EVENTS = 40
LONG_OP = 90.0           # tool calls at least this long are left out of the "core" pace
DEFAULT_SPC = 25.0       # eta.py's pace when fewer than 5 calls were measured
MIN_CALLS = 5
GAP_CAP = B.GAP_CAP

LENGTHS = [("2-10m", 150, 600), ("10-30m", 600, 1800), ("30m-2h", 1800, 7200), ("2h+", 7200, B.MAX_TASK_S)]
QUOTA = {"2-10m": 12, "10-30m": 16, "30m-2h": 20, "2h+": 12}    # per 60 cuts; long tasks over-weighted
FRAC_BINS = [(0.05, 0.35), (0.35, 0.65), (0.65, 0.95)]
CODEX_SHARE = 0.30
WAIT_SHARE = 0.40
WEIGHTS = [round(i / 10, 1) for i in range(11)]

HARNESS_NAME = {"claude": "Claude Code", "codex": "Codex", "pi": "pi"}


# ================================================================ small helpers

def length_bucket(wall):
    for name, lo, hi in LENGTHS:
        if lo <= wall < hi:
            return name
    return None


def frac_bucket(f):
    return "early (<35%)" if f < 0.35 else "middle (35-65%)" if f < 0.65 else "late (>65%)"


def fmt_off(s):
    s = max(0, int(round(s)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"+{h}:{m:02d}:{sec:02d}" if h else f"+{m:02d}:{sec:02d}"


def fmt_dur(s):
    if s is None:
        return "?"
    s = max(0, int(round(s)))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m} min {sec:02d} s" if m < 10 else f"{m} min"
    h, m = divmod(round(s / 60), 60)
    return f"{h} h {m:02d} min"


def _write(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


# ---------------------------------------------------------------- redaction
# Anything that would let a reader open the full transcript, a tool's later output, the
# ETA history or the T3 database is replaced. Project paths stay: the agent saw them and
# they carry meaning (the instructions forbid opening files).

_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_STORE = re.compile(
    r"(?:/Users/[^/\s]+|/home/[^/\s]+|~|\$HOME)/\.(?:claude/projects|codex/(?:sessions|archived_sessions)|eta|pi/agent/sessions|"
    r"t3/userdata)[^\s\"'`)\]>,;]*"
    r"|/(?:private/)?tmp/claude-\d+/[^\s\"'`)\]>,;]*"
    r"|[^\s\"'`(\[<]*/tool-results/[^\s\"'`)\]>,;]*"
    r"|\brollout-\d{4}-\d\d-\d\dT[\w-]+\.jsonl")
_T3ID = re.compile(r"(?:(?:node|thread)(?::|%3A|%253A))?delegated-task(?::|%3A|%253A)[^\s\"'`)\],;]+")


def scrub(text):
    text = _STORE.sub("<temp or transcript path removed>", text or "")
    text = _T3ID.sub("<delegated-task id>", text)
    return _UUID.sub("<id>", text)


def scrub_deep(x):
    if isinstance(x, str):
        return scrub(x)
    if isinstance(x, list):
        return [scrub_deep(v) for v in x]
    if isinstance(x, dict):
        return {k: scrub_deep(v) for k, v in x.items()}
    return x


def scrub_event(e):
    """Redact before any clipping, so a cut never leaves a recognisable fragment."""
    out = dict(e)
    for k in ("text", "name", "input"):
        if k in out:
            out[k] = scrub_deep(out[k])
    return out


def clip(text, n):
    t = (text or "").strip()
    if len(t) <= n:
        return t
    head = int(n * 0.6)
    tail = n - head
    return f"{t[:head].rstrip()} [... {len(t) - n} chars cut ...] {t[-tail:].lstrip()}"


def one_line(text, n):
    return clip(re.sub(r"\s+", " ", text or ""), n)


def block_text(content):
    if isinstance(content, str):
        return content
    out = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                t = b.get("type")
                if t in ("text", "input_text", "output_text", "Text"):
                    out.append(b.get("text") or "")
                elif t in ("image", "input_image"):
                    out.append("[image]")
                elif t == "tool_reference":
                    out.append(f"[tool loaded: {b.get('tool_name')}]")
                elif isinstance(b.get("content"), (str, list)):
                    out.append(block_text(b["content"]))
    return "\n".join(x for x in out if x)


# ================================================================ transcript parsing (rich, with text)
# Events: dict(ts, kind, ...) with kind in
#   prompt  (text, cls)        a user turn; cls from baseline.classify_prompt
#   note    (text)             notification, queued message, interrupt, compaction marker
#   agent   (text)             visible agent text
#   think   (text)             reasoning text, when the transcript has it in clear
#   tool    (id, name, input)  tool call
#   result  (id, text, error, bg)   tool result; bg = launched in the background

def _claude_prompt_event(ts, content, notify):
    text = scrub_prompt_text(content)
    cls = B.classify_prompt(text, notify)
    if cls is None:
        return None
    if cls in ("notify", "interrupt", "meta"):
        return {"ts": ts, "kind": "note", "text": text, "cls": cls}
    return {"ts": ts, "kind": "prompt", "text": text, "cls": cls}


def scrub_prompt_text(content):
    text = B._eta._clean_prompt(content)
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "image" for b in content):
        text = (text + "\n[image attached]").strip()
    return text


def parse_claude_rich(path):
    ev = []
    with open(path, "rb") as f:
        for raw in f:
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            typ = d.get("type")
            ts = B._eta.parse_ts(d.get("timestamp"))
            if ts is None or d.get("isSidechain"):
                continue
            if typ == "attachment":
                att = d.get("attachment") if isinstance(d.get("attachment"), dict) else {}
                if att.get("type") == "queued_command" and att.get("prompt"):
                    p = str(att.get("prompt"))
                    notify = "<task-notification>" in p[:200]
                    e = _claude_prompt_event(ts, p, notify)
                    if e:
                        if e["kind"] == "prompt":   # a person typed mid-turn; shown, never a task start
                            e = {"ts": ts, "kind": "note", "text": "[user message queued mid-turn] " + e["text"],
                                 "cls": "queued"}
                        ev.append(e)
                continue
            if typ not in ("user", "assistant"):
                continue
            msg = d.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if typ == "assistant":
                for b in content if isinstance(content, list) else []:
                    if not isinstance(b, dict):
                        continue
                    t = b.get("type")
                    if t == "text" and (b.get("text") or "").strip():
                        ev.append({"ts": ts, "kind": "agent", "text": b["text"]})
                    elif t == "thinking" and (b.get("thinking") or "").strip():
                        ev.append({"ts": ts, "kind": "think", "text": b["thinking"]})
                    elif t == "tool_use":
                        ev.append({"ts": ts, "kind": "tool", "id": b.get("id"), "name": b.get("name") or "?",
                                   "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
                continue
            if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result"
                                                 for b in content):
                info = d.get("toolUseResult") if isinstance(d.get("toolUseResult"), dict) else {}
                bg = bool(info.get("backgroundTaskId") or info.get("isAsync")
                          or info.get("status") == "async_launched")
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_result":
                        ev.append({"ts": ts, "kind": "result", "id": b.get("tool_use_id"),
                                   "text": block_text(b.get("content")), "error": bool(b.get("is_error")), "bg": bg})
                    elif b.get("type") == "text" and str(b.get("text", "")).startswith("[Request interrupted"):
                        ev.append({"ts": ts, "kind": "note", "text": b["text"], "cls": "interrupt"})
                continue
            if d.get("isCompactSummary"):
                ev.append({"ts": ts, "kind": "note", "text": "[context compacted; summary the agent saw] "
                           + block_text(content), "cls": "compact"})
                continue
            if d.get("isMeta"):
                continue
            origin = d.get("origin") if isinstance(d.get("origin"), dict) else {}
            notify = d.get("turnOrigin") in ("task_notification", "system") or origin.get("kind") == "task-notification"
            e = _claude_prompt_event(ts, content, notify)
            if e:
                ev.append(e)
    return ev


def parse_codex_rich(path):
    ev, alt_agent, have_agent_items = [], [], False
    with open(path, "rb") as f:
        for raw in f:
            if b'"token_count"' in raw[:300] or b'"token_usage_record"' in raw[:300] or b'"world_state"' in raw[:300]:
                continue
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            ts = B._eta.parse_ts(d.get("timestamp"))
            if ts is None:
                continue
            typ, p = d.get("type"), d.get("payload") or {}
            if not isinstance(p, dict):
                continue
            pt = p.get("type")
            if typ == "event_msg" and pt == "item_completed":
                it = p.get("item") or {}
                k = it.get("type")
                if k == "UserMessage":
                    blocks = [{"type": "text", "text": b.get("text", "")} for b in it.get("content") or []
                              if isinstance(b, dict)]
                    text = B._eta._clean_prompt(blocks)
                    cls = B.classify_prompt(text)
                    if cls in ("notify", "interrupt", "meta"):
                        ev.append({"ts": ts, "kind": "note", "text": text, "cls": cls})
                    elif cls:
                        ev.append({"ts": ts, "kind": "prompt", "text": text, "cls": cls})
                elif k == "AgentMessage":
                    have_agent_items = True
                    text = block_text(it.get("content"))
                    if text.strip():
                        ev.append({"ts": ts, "kind": "agent", "text": text})
                elif k == "Reasoning":
                    text = "\n".join(str(x) for x in (it.get("summary_text") or []) if x)
                    if text.strip():
                        ev.append({"ts": ts, "kind": "think", "text": text})
            elif typ == "event_msg" and pt == "turn_aborted":
                ev.append({"ts": ts, "kind": "note", "text": "[turn aborted]", "cls": "interrupt"})
            elif typ == "response_item":
                if pt in ("custom_tool_call", "function_call", "local_shell_call"):
                    name = p.get("name") or ("shell" if pt == "local_shell_call" else "?")
                    inp = p.get("input") if pt == "custom_tool_call" else p.get("arguments")
                    if pt == "local_shell_call":
                        inp = p.get("action")
                    if isinstance(inp, str):
                        try:
                            parsed = json.loads(inp)
                            inp = parsed if isinstance(parsed, dict) else {"input": inp}
                        except ValueError:
                            inp = {"input": inp}
                    if p.get("namespace"):
                        name = f"{p['namespace']}__{name}" if name == "delegate_task" else name
                    ev.append({"ts": ts, "kind": "tool", "id": p.get("call_id") or p.get("id"), "name": name,
                               "input": inp if isinstance(inp, dict) else {"input": str(inp)}})
                elif pt in ("custom_tool_call_output", "function_call_output", "local_shell_call_output"):
                    out = p.get("output")
                    text = out if isinstance(out, str) else block_text(out)
                    if isinstance(out, dict):
                        text = json.dumps(out)
                    ev.append({"ts": ts, "kind": "result", "id": p.get("call_id"), "text": text or "",
                               "error": False, "bg": "Script running with cell ID" in (text or "")[:300]})
                elif pt == "message" and p.get("role") == "assistant":
                    text = block_text(p.get("content"))
                    if text.strip():
                        alt_agent.append({"ts": ts, "kind": "agent", "text": text})
    if not have_agent_items and alt_agent:
        ev += alt_agent
    ev.sort(key=lambda e: e["ts"])
    return ev


RICH_PARSERS = {"claude": parse_claude_rich, "codex": parse_codex_rich}


# ================================================================ rendering

def tool_input_text(name, inp, full=True):
    """A useful, bounded slice of a tool call's input."""
    n = name or "?"
    base = n.split("__")[-1]
    lim = 700 if full else 160
    if not isinstance(inp, dict):
        return one_line(str(inp), lim)
    flags = []
    if inp.get("run_in_background"):
        flags.append("background")
    if inp.get("timeout"):
        flags.append(f"timeout {inp['timeout']}")
    fl = f" [{', '.join(flags)}]" if flags else ""
    if base in ("Bash", "bash", "shell", "exec_command", "local_shell") or "command" in inp and isinstance(inp.get("command"), (str, list)):
        cmd = inp.get("command") or inp.get("cmd") or ""
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        desc = inp.get("description")
        s = (f"{desc}: " if desc else "") + (clip(cmd, lim) if full else one_line(cmd, lim))
        return s + fl
    if base == "exec" and "input" in inp:
        return (clip(inp["input"], lim + 300) if full else one_line(inp["input"], lim)) + fl
    if base == "TodoWrite" and isinstance(inp.get("todos"), list):
        items = [f"[{t.get('status', '?')}] {one_line(t.get('content') or t.get('activeForm') or '', 90)}"
                 for t in inp["todos"] if isinstance(t, dict)]
        return ("\n    " + "\n    ".join(items)) if full else f"{len(items)} items: " + "; ".join(items)[:lim]
    if base == "update_plan" and isinstance(inp.get("plan"), list):
        items = [f"[{t.get('status', '?')}] {one_line(t.get('step') or '', 90)}" for t in inp["plan"] if isinstance(t, dict)]
        return ("\n    " + "\n    ".join(items)) if full else "; ".join(items)[:lim]
    if base in ("Read",):
        extra = "".join(f" {k}={inp[k]}" for k in ("offset", "limit", "pages") if inp.get(k))
        return f"{inp.get('file_path')}{extra}"
    if base in ("Edit", "MultiEdit"):
        old, new = str(inp.get("old_string") or ""), str(inp.get("new_string") or "")
        s = f"{inp.get('file_path')} (replace {len(old)} chars with {len(new)} chars)"
        return s + (f": {one_line(new, 200)}" if full and new else "")
    if base == "Write":
        c = str(inp.get("content") or "")
        return f"{inp.get('file_path')} ({len(c)} chars)" + (f": {one_line(c, 200)}" if full else "")
    if base in ("Grep", "Glob"):
        return " ".join(f"{k}={inp[k]!r}" for k in ("pattern", "path", "glob", "type") if inp.get(k))
    if base in ("Agent", "Task"):
        s = f"{inp.get('description') or ''} (type {inp.get('subagent_type') or 'general'}"
        s += f", model {inp['model']}" if inp.get("model") else ""
        s += ")" + fl
        return s + (f"\n    prompt: {clip(str(inp.get('prompt') or ''), 500)}" if full else "")
    if base == "delegate_task" or base.endswith("delegate_task"):
        tgt = inp.get("target") if isinstance(inp.get("target"), dict) else {}
        s = f"{inp.get('title') or ''} (to {tgt.get('model') or tgt.get('providerInstanceId') or '?'}, mode {inp.get('mode') or '?'})"
        return s + (f"\n    task: {clip(str(inp.get('task') or ''), 500)}" if full else "")
    if base in ("TaskCreate", "TaskUpdate"):
        return " ".join(f"{k}={one_line(str(inp[k]), 100)!s}" for k in ("taskId", "subject", "status") if inp.get(k))
    raw = json.dumps(inp, ensure_ascii=False)
    return clip(raw, 400) if full else one_line(raw, lim)


def render_event(e, t0, full=True, res_lim=650):
    at = fmt_off(e["ts"] - t0)
    k = e["kind"]
    if k == "agent":
        return f"[{at}] agent: " + (clip(e["text"], 1500) if full else one_line(e["text"], 160))
    if k == "think":
        return f"[{at}] agent (reasoning): " + (clip(e["text"], 500) if full else one_line(e["text"], 120))
    if k == "tool":
        return f"[{at}] tool call {e['name']}: " + tool_input_text(e["name"], e["input"], full)
    if k == "result":
        txt = e.get("text") or ""
        tag = "error" if e.get("error") else "launched in background" if e.get("bg") else "result"
        size = f"{len(txt)} chars"
        if not full:
            return f"[{at}]   -> {tag} ({size})"
        body = clip(txt, res_lim) if txt.strip() else "(empty)"
        body = body.replace("\n", "\n      ")
        return f"[{at}]   -> {tag} ({size}):\n      {body}"
    if k == "note":
        return f"[{at}] [{e.get('cls') or 'note'}] " + (clip(e["text"], 700) if full else one_line(e["text"], 140))
    if k == "prompt":
        who = {"eta": "user (asks for an ETA)"}.get(e.get("cls"), f"user ({e.get('cls')})")
        return f"[{at}] {who}: " + (clip(e["text"], 1500) if full else one_line(e["text"], 160))
    return f"[{at}] {k}"


def open_foreground(events, cut_abs):
    """Tool calls with no result by the cut: (name, input slice, age_s)."""
    done = {e["id"] for e in events if e["kind"] == "result"}
    return [(e["name"], tool_input_text(e["name"], e["input"], False), cut_abs - e["ts"])
            for e in events if e["kind"] == "tool" and e["id"] not in done]


def open_spans(path, harness, cut_abs, t0):
    """Delegates and background commands still open at the cut, from waits.scan (read only).
    Unlike waits.open_waits, this never reads output files or subagent metadata written later."""
    try:
        import waits as W
        tl = W.scan(path, harness, until=cut_abs)
    except Exception:
        return []
    out = []
    for s in tl.open_spans():
        if s.extra.get("history_only"):
            continue
        age = cut_abs - s.start
        limit = W.ORPHAN if s.what == "delegate" else W.ORPHAN_COMMAND
        if age > limit or age < 0:
            continue
        service = W.is_service(s)
        if not s.bg and s.what == "command":
            continue   # foreground calls are listed from the transcript itself
        out.append({"what": s.what, "label": scrub(s.label), "model": s.model, "age_s": age,
                    "before_task": s.start < t0, "service": service})
    return out


def pace_numbers(task, cut):
    """Tool calls so far, active seconds (quiet gaps capped like eta.py), and two paces."""
    calls = [(s, e) for s, e, c in task["c"] if s <= cut]
    acts = sorted([a for a in task.get("a") or [] if a <= cut] + [s for s, _ in calls])
    active, prev = 0.0, 0.0
    for a in acts:
        active += min(max(0.0, a - prev), GAP_CAP)
        prev = a
    active += min(max(0.0, cut - prev), GAP_CAP)
    n = len(calls)
    long_t, long_n = 0.0, 0
    for s, e in calls:
        e2 = cut if (e == -1 or e > cut) else e
        if e2 - s >= LONG_OP:
            long_t += min(e2 - s, GAP_CAP)
            long_n += 1
    if n >= MIN_CALLS and active > 0:
        spc = active / n
        core = (active - long_t) / (n - long_n) if n > long_n else spc
        src = "task"
    else:
        spc = core = DEFAULT_SPC
        src = "default"
    return {"calls": n, "active_s": round(active, 1), "spc": round(spc, 2), "spc_core": round(max(1.0, core), 2),
            "pace_source": src, "long_ops": long_n, "long_ops_s": round(long_t, 1)}


def render_cut(cut_id, task, path, cut, events):
    """Markdown for one cut. Returns (text, stats)."""
    t0 = task["t0"]
    cut_abs = t0 + cut
    start_i = None
    for i, e in enumerate(events):
        if e["kind"] == "prompt" and abs(e["ts"] - t0) < 0.01 and e.get("cls") == task.get("o", "human"):
            start_i = i
            break
    if start_i is None:
        for i, e in enumerate(events):
            if e["kind"] == "prompt" and abs(e["ts"] - t0) < 1.0:
                start_i = i
                break
    if start_i is None:
        raise ValueError("task start prompt not found in transcript")
    request = scrub_event(events[start_i])
    window = [scrub_event(e) for e in events[start_i + 1:] if e["ts"] <= cut_abs + 1e-6]
    before = [scrub_event(e) for e in events[max(0, start_i - 400):start_i] if e["ts"] <= t0]

    # ---- header facts
    pace = pace_numbers(task, cut)
    st = B.state_at(task, cut)
    fg = open_foreground([e for e in events if t0 - 1 <= e["ts"] <= cut_abs], cut_abs)
    spans = open_spans(path, task["h"], cut_abs, t0)
    last_act = max([e["ts"] for e in window if e["kind"] in ("agent", "think", "tool", "result")] or [t0])
    H = [f"# {cut_id}", "",
         "## Measured facts at this moment (what /eta would also measure)", "",
         f"- Harness: {HARNESS_NAME.get(task['h'], task['h'])}; model: {task.get('m') or 'unknown'}",
         f"- Elapsed since the request that started this task: {fmt_dur(cut)} (wall clock)",
         f"- Tool calls so far in this task: {pace['calls']}",
         (f"- Pace so far: {pace['spc']:.1f} s per tool call over {fmt_dur(pace['active_s'])} of active time "
          f"(quiet gaps capped at 20 min); {pace['spc_core']:.1f} s per call excluding the "
          f"{pace['long_ops']} calls that ran 90 s or longer"
          if pace["pace_source"] == "task" else
          "- Pace so far: too few tool calls to measure (under 5)"),
         f"- Last agent activity: {fmt_dur(cut_abs - last_act)} ago"]
    running = []
    for name, desc, age in fg:
        running.append(f"  - foreground tool call {name} running for {fmt_dur(age)}: {scrub(desc)}")
    for s in spans:
        what = "delegated agent" if s["what"] == "delegate" else "background command"
        extra = " (started before this task)" if s["before_task"] else ""
        svc = " (looks like a server or watcher, usually not waited on)" if s["service"] else ""
        model = f", model {s['model']}" if s.get("model") else ""
        running.append(f"  - {what} open for {fmt_dur(s['age_s'])}{model}{extra}{svc}: {s['label']}")
    if running:
        H.append("- Running or open right now:")
        H += running
    else:
        H.append("- Running or open right now: nothing")
    H.append("- The agent has not ended its turn yet.")

    # ---- context before the task
    ctx = []
    prev_prompts = [e for e in before if e["kind"] == "prompt" and e.get("cls") in ("human", "scheduled", "side",
                                                                                     "delegated")][-2:]
    prev_agent = [e for e in before if e["kind"] == "agent"][-1:]
    for e in sorted(prev_prompts + prev_agent, key=lambda x: x["ts"]):
        who = "user" if e["kind"] == "prompt" else "agent (final reply of the previous task)"
        ctx.append(f"[{fmt_dur(t0 - e['ts'])} before] {who}: {clip(e['text'], 900)}")

    # ---- events: head and tail in full, middle compact, oldest middle elided if needed
    req_txt = clip(request["text"], 7000)
    full = [render_event(e, t0, True) for e in window]
    compact = [render_event(e, t0, False) for e in window]
    fixed = sum(map(len, H)) + sum(map(len, ctx)) + len(req_txt) + 1500
    budget = CHAR_BUDGET - fixed
    n = len(window)
    elided = 0
    if sum(map(len, full)) + n <= budget:
        body = full
        mode = "full"
    else:
        tail_n = min(TAIL_EVENTS, n)
        tail = full[n - tail_n:]
        res_lim = 650
        while sum(map(len, tail)) > budget * 0.6 and res_lim > 120:
            res_lim = int(res_lim * 0.7)
            tail = [render_event(e, t0, True, res_lim) for e in window[n - tail_n:]]
        head, used = [], 0
        for line in full[:n - tail_n]:
            if used + len(line) > HEAD_BUDGET and len(head) >= 3:
                break
            head.append(line)
            used += len(line) + 1
        mid_lo, mid_hi = len(head), n - tail_n
        room = budget - used - sum(map(len, tail)) - 400
        keep_from = mid_hi
        acc = 0
        while keep_from > mid_lo and acc + len(compact[keep_from - 1]) + 1 <= room:
            keep_from -= 1
            acc += len(compact[keep_from]) + 1
        dropped = window[mid_lo:keep_from]
        elided = len(dropped)
        mid = []
        if dropped:
            from collections import Counter
            tools = Counter(e["name"] for e in dropped if e["kind"] == "tool")
            n_agent = sum(e["kind"] == "agent" for e in dropped)
            span = f"{fmt_off(dropped[0]['ts'] - t0)} to {fmt_off(dropped[-1]['ts'] - t0)}"
            mid.append(f"[... {len(dropped)} events from {span} elided to fit: {sum(tools.values())} tool calls ("
                       + ", ".join(f"{k} {v}" for k, v in tools.most_common(8))
                       + f"), {n_agent} agent messages ...]")
        if keep_from < mid_hi:
            mid.append("[the following events are shown one line each, results omitted]")
            mid += compact[keep_from:mid_hi]
            mid.append("[full detail resumes]")
        body = head + mid + tail
        mode = "elided" if dropped else "compact-middle"

    L = H + [""]
    if ctx:
        L += ["## Earlier in this session (before this task started)", ""] + ctx + [""]
    L += ["## The request that started this task", "", f"[{fmt_off(0)}] user: {req_txt}", "",
          "## What the agent has seen and done since (oldest first)", ""]
    L += body
    L += ["", f"[{fmt_off(cut)}] <- now. Nothing after this moment is shown.", ""]
    text = scrub("\n".join(L))
    stats = {"events": n, "elided_events": elided, "render": mode, "chars": len(text),
             "approx_tokens": approx_tokens(text), "open_fg": len(fg), "open_spans": len(spans),
             "open_items": [{"what": "foreground " + name, "age_s": round(age, 1)} for name, _, age in fg]
             + [{"what": s["what"], "age_s": round(s["age_s"], 1), "service": s["service"],
                 "before_task": s["before_task"]} for s in spans]}
    return text, stats


def approx_tokens(text):
    """Rough token count: about 4 characters per token for prose, 3 for code and logs."""
    words = len(re.findall(r"\w+", text))
    symbols = len(re.findall(r"[^\w\s]", text))
    return int(words * 1.3 + symbols * 0.6)


# ================================================================ sampling

def task_records(out_dir, cache=None, now=None):
    """Task-index records, never writing ~/.eta. Uses ~/.eta/turns.jsonl when it matches the
    current baseline.py schema; otherwise (or with --cache) builds a private copy in the
    output directory from the transcripts, which are only read."""
    if cache:
        recs = list(B.load_cache(cache).values())
        return recs or B.build(path=cache, now=now), cache
    recs = list(B.load_cache().values())
    if recs:
        return recs, B.cache_path()
    private = os.path.join(out_dir, ".turns-cache.jsonl")
    print(f"~/.eta task index is missing or from another schema; building a private copy at {private}",
          file=sys.stderr)
    return B.build(path=private, now=now), private


def load_tasks_with_files(now=None, recs=None):
    """Like baseline.tasks_from_records, but keeps the source file of the chosen copy. Read only."""
    now = now or time.time()
    best = {}
    for r in (recs if recs is not None else B.load_cache().values()):
        for t in r.get("tasks") or []:
            if t["end"] == "live" and now - r["mt"] > B.STALE_S:
                t = dict(t, end="cut")
            k = t["k"]
            if k not in best or (t["n"], t["wall"]) > (best[k][0]["n"], best[k][0]["wall"]):
                best[k] = (t, r["f"])
    return sorted(best.values(), key=lambda tf: tf[0]["t0"])


def wait_intervals(t):
    """Stretches where the task waited on something external: (start, end, kind), task-relative."""
    W_ = t["wall"]
    out = []
    for s, e, c in t["c"]:
        e2 = W_ if e == -1 else e
        if c in "dw" and e2 - s >= 60:
            out.append((s, e2, "delegate" if c == "d" else "wait"))
        elif c == "s" and e2 - s >= 120:
            out.append((s, e2, "command"))
    for s, e in t.get("b") or []:
        e2 = W_ if e == -1 else e
        if e2 - s >= 60:
            out.append((s, e2, "background"))
    return out


def max_idle_gap(t):
    """Longest stretch with no agent activity and nothing open (no tool call, delegate or
    background job running). Long ones are a sleeping machine, a stalled harness or a late
    wake-up by a leftover notification: nothing in the transcript can foresee them."""
    pts = sorted(set([0.0, t["wall"]] + [float(a) for a in t.get("a") or []]
                     + [x for s, e, _ in t["c"] for x in (s, e) if x != -1]
                     + [x for s, e in t.get("b") or [] for x in (s, e) if x != -1]))
    busy = [(s, t["wall"] if e == -1 else e) for s, e, _ in t["c"]] + \
           [(s, t["wall"] if e == -1 else e) for s, e in t.get("b") or []]
    worst = 0.0
    for a, b in zip(pts, pts[1:]):
        if b - a <= worst:
            continue
        if any(s <= a and e >= b for s, e in busy):
            continue
        worst = b - a
    return worst


def open_at(t, cut):
    return [k for s, e, k in wait_intervals(t) if s + 15 <= cut < e]


def asking_user_at(t, cut):
    return any(c == "a" and s <= cut and (e == -1 or e > cut) for s, e, c in t["c"])


def place_cut(t, mode, rng, fbin=None):
    W_ = t["wall"]
    lo_ok, hi_ok = MIN_ELAPSED, W_ - MIN_REMAIN
    if hi_ok <= lo_ok:
        return None
    for _ in range(40):
        if mode == "wait":
            ivs = [(max(s + 20, lo_ok), min(e, hi_ok)) for s, e, _ in wait_intervals(t)]
            ivs = [(a, b) for a, b in ivs if b > a]
            if not ivs:
                return None
            tot = sum(b - a for a, b in ivs)
            x = rng.uniform(0, tot)
            for a, b in ivs:
                if x <= b - a:
                    c = a + x
                    break
                x -= b - a
            else:
                c = ivs[-1][1]
        else:
            a, b = fbin if _ < 20 else FRAC_BINS[_ % len(FRAC_BINS)]
            c = rng.uniform(max(a * W_, lo_ok), min(b * W_, hi_ok)) if min(b * W_, hi_ok) > max(a * W_, lo_ok) else None
            if c is None:
                continue
        if asking_user_at(t, c):
            continue
        if mode == "wait" and not open_at(t, c):
            continue
        if mode != "wait" and _ < 25 and open_at(t, c):
            continue   # fraction cuts prefer moments with nothing open, so both kinds are represented
        return c
    return None


def candidates(tasks_files, now):
    fit_ends = sorted(t["t0"] + t["wall"] for t, _ in tasks_files if B.fit_eligible(t))
    import bisect
    out = []
    for t, f in tasks_files:
        if not B.fit_eligible(t) or t["h"] not in ("claude", "codex"):
            continue
        if length_bucket(t["wall"]) is None or not os.path.isfile(f):
            continue
        if t.get("uw", 0) > MAX_USER_WAIT_SHARE * t["wall"]:
            continue   # mostly waiting on a person's answer: remaining time is about the person
        if max_idle_gap(t) > MAX_IDLE_S:
            continue   # unexplained idle stretch (sleep, stall, leftover wake-up), see max_idle_gap
        if bisect.bisect_left(fit_ends, t["t0"]) < MIN_TRAIN:
            continue
        out.append((t, f))
    return out


def scaled_quota(n):
    tot = sum(QUOTA.values())
    q = {k: int(round(v * n / tot)) for k, v in QUOTA.items()}
    diff = n - sum(q.values())
    q["30m-2h"] += diff
    return q


def select(cands, n, seed, now):
    """Stratified, recency-weighted pick of tasks with an assigned cut mode. One cut per task."""
    rng = random.Random(seed)
    quota = scaled_quota(n)
    by_bucket = {name: [] for name, _, _ in LENGTHS}
    for t, f in cands:
        age_d = max(0.0, (now - t["t0"]) / 86400)
        w = 0.5 ** (age_d / RECENCY_HALF_LIFE_D)
        key = rng.random() ** (1.0 / max(w, 1e-9))
        by_bucket[length_bucket(t["wall"])].append((key, t, f))
    order = [name for name, _, _ in LENGTHS][::-1]          # longest first, deficits flow shorter
    chosen, carry = [], 0
    for bi, name in enumerate(order):
        pool = sorted(by_bucket[name], key=lambda x: -x[0])
        want = quota[name] + carry
        picks = []
        if len(pool) <= want:
            picks = [(t, f) for _, t, f in pool]
        else:
            codex_t = math.ceil(CODEX_SHARE * want)
            wait_t = math.ceil(WAIT_SHARE * want)
            taken = set()

            def take(pred, limit):
                for _, t, f in pool:
                    if len(picks) >= want or limit <= 0:
                        return
                    if t["k"] in taken or not pred(t):
                        continue
                    picks.append((t, f))
                    taken.add(t["k"])
                    limit -= 1
            # codex first, preferring ones with a wait so they also count toward the wait share
            take(lambda t: t["h"] == "codex" and wait_intervals(t), codex_t)
            have_codex = sum(t["h"] == "codex" for t, _ in picks)
            take(lambda t: t["h"] == "codex", codex_t - have_codex)
            have_wait = sum(bool(wait_intervals(t)) for t, _ in picks)
            take(lambda t: bool(wait_intervals(t)), wait_t - have_wait)
            take(lambda t: True, want)
        carry = want - len(picks)
        # assign modes: wait-capable tasks fill the wait share first, the rest are spread over fractions
        wait_cap = [p for p in picks if wait_intervals(p[0])]
        rng.shuffle(wait_cap)
        n_wait = min(len(wait_cap), math.ceil(WAIT_SHARE * len(picks)))
        wait_keys = {p[0]["k"] for p in wait_cap[:n_wait]}
        others = [p for p in picks if p[0]["k"] not in wait_keys]
        rng.shuffle(others)
        for p in picks:
            if p[0]["k"] in wait_keys:
                chosen.append((p[0], p[1], "wait", None))
        for i, p in enumerate(others):
            chosen.append((p[0], p[1], "frac", FRAC_BINS[i % len(FRAC_BINS)]))
        # leftovers for replacement if a cut cannot be placed or rendered
        by_bucket[name] = [x for x in pool if x[1]["k"] not in {p[0]["k"] for p in picks}]
    return chosen, by_bucket


# ================================================================ answer key pieces

@contextmanager
def no_future_reads(W):
    """waits.open_waits reads some live files (background output tails, subagent metadata,
    Hugging Face partial downloads) as they are now, which for a past cut is the future.
    Disable those reads while forecasting a historical cut."""
    saved = (W._output_finished, W.output_sources, W.incomplete_downloads, W._read_json)
    real_read = W._read_json

    def read_json(path, default):
        if str(path).endswith(".meta.json"):
            return default
        return real_read(path, default)
    W._output_finished = lambda span: False
    W.output_sources = lambda span, cwd: []
    W.incomplete_downloads = lambda now, window=300: []
    W._read_json = read_json
    try:
        yield
    finally:
        W._output_finished, W.output_sources, W.incomplete_downloads, W._read_json = saved


def load_waits_history(out_dir):
    """waits.History from the existing ~/.eta cache without refreshing it. If that cache is
    missing or from another version, mine a private copy under the output directory."""
    try:
        import waits as W
    except Exception as exc:  # pragma: no cover
        return None, f"waits.py import failed: {type(exc).__name__}"
    cache = W._read_json(W.cache_path(), None)
    src = W.cache_path()
    if not cache or cache.get("version") != W.CACHE_VERSION:
        home = os.path.join(out_dir, ".waits-home")
        os.makedirs(home, mode=0o700, exist_ok=True)
        old = os.environ.get("ETA_HOME")
        os.environ["ETA_HOME"] = home
        try:
            print(f"waits cache missing or another version; mining a private copy in {home}", file=sys.stderr)
            cache = W.mine(budget_s=None)
            src = W.cache_path()
        finally:
            if old is None:
                os.environ.pop("ETA_HOME", None)
            else:
                os.environ["ETA_HOME"] = old
    try:
        return W.load_history(cache=cache), src
    except Exception as exc:
        return None, f"waits history failed: {type(exc).__name__}"


def waits_forecast_at(path, harness, cut_abs, history):
    if history is None:
        return None
    import waits as W
    try:
        with no_future_reads(W):
            fc = W.forecast(path, harness, now=cut_abs, history=history)
    except Exception as exc:
        return {"error": type(exc).__name__}
    keep = ("kind", "label", "model", "elapsed_s", "p50_s", "low_s", "high_s", "n", "background")
    out = {"waits": [{k: (scrub(v) if isinstance(v, str) else v) for k, v in w.items() if k in keep}
                     for w in fc["waits"]],
           "wait": fc["wait"], "tail": None, "total": fc["total"]}
    if fc.get("tail"):
        out["tail"] = {k: fc["tail"][k] for k in ("p50_s", "low_s", "high_s", "n")}
    return out


def new_wait_defaults(history, cut_abs):
    """Median length of a fresh delegate and of a fresh long command, from history before the cut."""
    if history is None:
        return None
    h = history.before(cut_abs)
    dl = sorted(d["dur"] for d in h.delegates)
    cm = sorted(c["dur"] for c in h.commands if c["dur"] >= 60 and c.get("status") == "completed")
    q = B._eta.quantile
    return {"delegate_s": q(dl, 0.5) if dl else None, "n_delegate": len(dl),
            "command_s": q(cm, 0.5) if cm else None, "n_command": len(cm)}


# ================================================================ prepare

INSTRUCTIONS = """# Estimating instructions (one cut at a time)

Give the estimating model exactly the prompt below, with `{CUT_ID}` replaced and the cut
file's full text pasted where shown. Run each cut in a fresh conversation. Do not give the
model tools if you can avoid it; if it has tools, the prompt forbids opening anything.
Never show it `key.json`, other cut files, or this repository.

Save each reply as `answers/<cut id>.json` (or many replies as a JSON list in one file).

---

## Prompt

You are helping test how well a coding agent can tell its user when the current task will
be done.

Below is a record of a coding agent's work on one task, cut at a moment while the agent
was still working. It shows the user's request, what the agent said, the tool calls it made
with trimmed results, and timestamps as offsets from the request (+MM:SS or +H:MM:SS). A
short header lists facts that a timing tool measured at that moment: time elapsed, tool
calls so far, pace per tool call, and anything still running.

Put yourself in the working agent's position at the moment of the cut. You know only what
it knew then. The task ends at the agent's last action on this request: normally its final
reply. Time spent waiting for delegated agents, background jobs, CI or other external work
counts. If the agent ends a turn while something it launched is still running and is woken
when that finishes, the task continues until the agent is done after the wake-up.

Rules:
- Use only the text below. Do not open, search for or run anything else: no files, no
  commands, no web, no other cut files. If you have tools, do not use them.
- Answer in two steps, in this order.

Step 1. Before you analyse anything in detail, write down your honest gut estimate of how
many more minutes of wall-clock time this task will take from this moment. This is
`minutes_guess`. Do not revise it after Step 2.

Step 2. Now count what is left, as the working agent would plan it:
- `own_calls_left`: how many more tool calls the agent itself will make before its final
  reply (edits, commands, reads, checks, waits it polls, messages to tools). Count calls,
  not minutes. Include verification and cleanup it will do. 0 if it only needs to reply.
- `required_waits`: each external thing that must finish before the agent can finish, such
  as a delegated agent, background job, build, test run, CI, deploy or download. For each:
  `what` (a few words), `already_open` (true if it is already running at the cut, false if
  it still has to be started), and `calls_or_units_left` (remaining countable units if the
  work shows progress, such as 40 of 120 items left, else null). Leave the list empty if
  nothing external is required.
- `done_already`: true only if the agent has nothing left but its final reply.
- `notes`: one line on what remains.

Reply with only this JSON object, no other text:

{"cut":"{CUT_ID}","minutes_guess":<number>,"own_calls_left":<integer>,"required_waits":[{"what":"...","already_open":true,"calls_or_units_left":null}],"done_already":false,"notes":"<one line>"}

Record for {CUT_ID}:

<<<
(paste the full text of {CUT_ID}.md here)
>>>
"""


def prepare(args):
    now = time.time()
    out = args.out
    os.makedirs(out, mode=0o700, exist_ok=True)
    for p in glob.glob(os.path.join(out, "cut-*.md")):
        os.unlink(p)
    recs, src = task_records(out, args.cache, now)
    tf = load_tasks_with_files(now, recs)
    tasks = [t for t, _ in tf]
    cands = candidates(tf, now)
    chosen, leftovers = select(cands, args.n, args.seed, now)
    rng = random.Random(args.seed + 1)
    hist, wsrc = load_waits_history(out)
    if hist is None:
        print(f"waits forecast unavailable: {wsrc}", file=sys.stderr)
    parsed = {}
    made = []
    queue = list(chosen)
    spare = {b: list(v) for b, v in leftovers.items()}
    tried = set()
    while queue:
        t, f, mode, fbin = queue.pop(0)
        tried.add(t["k"])
        cut = place_cut(t, mode, rng, fbin)
        rendered = None
        if cut is not None:
            if f not in parsed:
                try:
                    parsed[f] = RICH_PARSERS[t["h"]](f)
                except (OSError, ValueError) as exc:
                    parsed[f] = exc
            ev = parsed[f]
            if not isinstance(ev, Exception):
                try:
                    rendered = (cut, ev)
                    render_cut("cut-000", t, f, cut, ev)   # dry run: raises if the start is missing
                except ValueError:
                    rendered = None
        if rendered is None:
            b = length_bucket(t["wall"])
            names = [name for name, _, _ in LENGTHS]
            pool = []
            # same length bucket first, then the next shorter ones
            for bb in names[names.index(b)::-1]:
                pool = [x for x in spare.get(bb, []) if x[1]["k"] not in tried
                        and (mode != "wait" or wait_intervals(x[1]))]
                if pool:
                    b = bb
                    break
            if pool:
                nxt = max(pool, key=lambda x: x[0])
                spare[b].remove(nxt)
                queue.append((nxt[1], nxt[2], mode, fbin))
            continue
        made.append((t, f, mode, cut))
    rng.shuffle(made)       # cut numbers carry no information about length or order
    key = {"_meta": {"created": datetime.now().isoformat(timespec="seconds"), "seed": args.seed, "n": len(made),
                     "note": "Answer key. Never show this file to an estimating model.",
                     "task_index": "private copy" if src != B.cache_path() else "~/.eta/turns.jsonl",
                     "waits_history": "private copy" if hist is not None and wsrc != os.path.expanduser(
                         "~/.eta/waits-cache.json") else ("~/.eta/waits-cache.json" if hist else wsrc)}}
    sizes = []
    for i, (t, f, mode, cut) in enumerate(made, 1):
        cid = f"cut-{i:03d}"
        text, stats = render_cut(cid, t, f, cut, parsed[f])
        _write(os.path.join(out, f"{cid}.md"), text)
        sizes.append(stats["approx_tokens"])
        cut_abs = t["t0"] + cut
        train = [x for x in tasks if x["t0"] + x["wall"] < t["t0"]]
        base = B.Baseline(train)
        st = B.state_at(t, cut)
        prior = base.remaining_given(st)
        pace = pace_numbers(t, cut)
        opens = open_at(t, cut)
        key[cid] = {
            "task": t["k"], "harness": t["h"], "model": t.get("m"),
            "started": datetime.fromtimestamp(t["t0"]).strftime("%Y-%m-%d %H:%M"),
            "task_wall_s": t["wall"], "length_bucket": length_bucket(t["wall"]),
            "elapsed_s": round(cut, 1), "frac": round(cut / t["wall"], 3), "mode": mode,
            "actual_remaining_s": round(t["wall"] - cut, 1),
            "calls_so_far": pace["calls"], "actual_calls_left": sum(1 for s, _, _ in t["c"] if s > cut),
            "active_s": pace["active_s"], "pace_spc": pace["spc"], "pace_core_spc": pace["spc_core"],
            "pace_source": pace["pace_source"],
            "wait_open": wait_open(stats), "open_kinds": sorted(set(opens)),
            "open_age_s": round(max([cut - s for s, e, _ in wait_intervals(t) if s <= cut < e] or [0.0]), 1),
            "prior": {k: prior[k] for k in ("p10", "p50", "p80", "p90", "n")},
            "prior_n_train": base.n_fit,
            "waits_forecast": waits_forecast_at(f, t["h"], cut_abs, hist),
            "new_wait_defaults": new_wait_defaults(hist, cut_abs),
            "state": {k: v for k, v in st.items()},
            "render": stats,
        }
    _write(os.path.join(out, "key.json"), json.dumps(key, indent=1))
    _write(os.path.join(out, "INSTRUCTIONS.md"), INSTRUCTIONS)
    os.makedirs(os.path.join(out, "answers"), mode=0o700, exist_ok=True)
    print(f"wrote {len(made)} cuts to {out} (key.json, INSTRUCTIONS.md, answers/ for replies)")
    print_composition(key)
    if sizes:
        s = sorted(sizes)
        print(f"cut size, approx tokens: median {statistics.median(s):.0f}, min {s[0]}, max {s[-1]}")
    return key


def wait_open(stats):
    """Something external is open at the cut: a delegate or background job that is not a
    server or watcher, or a foreground tool call that has run at least 30 s."""
    for it in stats.get("open_items") or []:
        if it["what"].startswith("foreground"):
            if it["age_s"] >= 30:
                return True
        elif not it.get("service"):
            return True
    return False


def print_composition(key):
    from collections import Counter
    rows = [v for k, v in key.items() if not k.startswith("_")]
    print(f"\ncomposition of {len(rows)} cuts")
    for title, f in (("task length", lambda r: r["length_bucket"]), ("harness", lambda r: r["harness"]),
                     ("elapsed fraction", lambda r: frac_bucket(r["frac"])),
                     ("wait open at the cut", lambda r: "yes" if r["wait_open"] else "no"),
                     ("cut placed", lambda r: r["mode"]),
                     ("remaining", lambda r: R.elapsed_bucket(r["actual_remaining_s"])),
                     ("render", lambda r: r["render"]["render"])):
        c = Counter(f(r) for r in rows)
        print(f"  {title}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    c = Counter((r["length_bucket"], r["harness"], "wait" if r["wait_open"] else "-") for r in rows)
    print("  length x harness x wait: " + ", ".join(f"{a}/{b}/{w} {n}" for (a, b, w), n in sorted(c.items())))


# ================================================================ score

def load_key(path):
    with open(path) as f:
        key = json.load(f)
    return {k: v for k, v in key.items() if not k.startswith("_")}


def _parse_answer_text(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        pass
    out = []
    for line in t.splitlines():   # JSON lines
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    if out:
        return out
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except ValueError:
            pass
    return None


def load_answers(path):
    """{cut id: answer dict} from a directory of .json files, or one file. Later files win."""
    files = sorted(glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.jsonl"))) \
        if os.path.isdir(path) else [path]
    out, bad = {}, []
    for p in files:
        try:
            with open(p) as f:
                data = _parse_answer_text(f.read())
        except OSError:
            data = None
        if data is None:
            bad.append(os.path.basename(p))
            continue
        items = data if isinstance(data, list) else [data]
        for a in items:
            if isinstance(a, dict) and a.get("cut"):
                out[str(a["cut"])] = a
    return out, bad


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _wait_kind(what):
    w = (what or "").lower()
    if re.search(r"agent|delegat|subagent|review|codex|claude|opus|sonnet|gpt|thread|worker", w):
        return "delegate"
    return "command"


def est_minutes(k, a):
    m = _num(a.get("minutes_guess"))
    return None if m is None else (m * 60, m * 60, m * 60)


def est_count_pace(k, a):
    n = _num(a.get("own_calls_left"))
    if n is None:
        return None
    s = 0.0 if a.get("done_already") is True else max(0.0, n) * k["pace_spc"]
    return (s, s, s)


def waits_part(k, a):
    """Seconds of external waiting the answer says is still required, priced from measurements."""
    req = [w for w in (a.get("required_waits") or []) if isinstance(w, dict)]
    total = 0.0
    fc = k.get("waits_forecast") or {}
    if any(w.get("already_open") is True for w in req):
        if fc.get("wait") and fc["wait"].get("p50_s") is not None:
            total += fc["wait"]["p50_s"]
        else:   # nothing measurable: about as long again as it has been open (waits.elapsed_fallback)
            total += 0.75 * max(60.0, k.get("open_age_s") or 0.0)
    nd = k.get("new_wait_defaults") or {}
    for w in req:
        if w.get("already_open") is True:
            continue
        kind = _wait_kind(w.get("what"))
        d = nd.get(f"{kind}_s") or nd.get("command_s") or nd.get("delegate_s") or 300.0
        total += d
    return total


def est_count_pace_waits(k, a):
    n = _num(a.get("own_calls_left"))
    if n is None:
        return None
    if a.get("done_already") is True:
        return (0.0, 0.0, 0.0)
    s = max(0.0, n) * k["pace_core_spc"] + waits_part(k, a)
    return (s, s, s)


def est_prior(k, a):
    p = k["prior"]
    return (p["p10"], p["p50"], p["p90"])


def est_oracle(k, a):
    s = k["actual_calls_left"] * k["pace_spc"]
    return (s, s, s)


def blend(mid_c, mid_p, w):
    lc, lp = math.log(max(mid_c, FLOOR)), math.log(max(mid_p, FLOOR))
    return math.exp(w * lc + (1 - w) * lp)


def make_blend(fn, w):
    def est(k, a):
        c = fn(k, a)
        if c is None:
            return None
        m = blend(c[1], k["prior"]["p50"], w)
        return (m, m, m)
    return est


@contextmanager
def patched_prior(mod, prior):
    """Make baseline.remaining_given/default_baseline answer with the leak-free prior."""
    class _Fixed:
        def remaining_given(self, state):
            return dict(prior, n_survivors=prior.get("n"), basis="replay prior (history before the task)")

        def estimate(self, state):
            return prior["p10"], prior["p50"], prior["p90"]

        def quantiles(self, state, qs=(0.1, 0.5, 0.8, 0.9)):
            m = {0.1: prior["p10"], 0.5: prior["p50"], 0.8: prior["p80"], 0.9: prior["p90"]}
            return [m.get(q, prior["p50"]) for q in qs], prior.get("n"), []
    fixed = _Fixed()
    targets = [sys.modules.get("baseline")] + [getattr(mod, n, None) for n in ("baseline", "B", "_baseline")]
    saved = []
    for t in targets:
        if t is None or not hasattr(t, "remaining_given"):
            continue
        saved.append((t, t.remaining_given, getattr(t, "default_baseline", None)))
        t.remaining_given = lambda state, baseline=None: fixed.remaining_given(state)
        t.default_baseline = lambda *a, **kw: fixed
    if hasattr(mod, "remaining_given"):
        saved.append((mod, mod.remaining_given, getattr(mod, "default_baseline", None)))
        mod.remaining_given = lambda state, baseline=None: fixed.remaining_given(state)
    try:
        yield
    finally:
        for t, rg, db in saved:
            t.remaining_given = rg
            if db is not None:
                t.default_baseline = db


def load_module_estimator(path, func=None):
    spec = importlib.util.spec_from_file_location("llm_replay_combiner_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    if hasattr(mod, "llm_replay_estimate") and not func:
        hook = mod.llm_replay_estimate

        def est(k, a):
            with patched_prior(mod, k["prior"]):
                r = hook(k, a)
            return None if r is None else tuple(r[:3])
        return est, f"{os.path.basename(path)}:llm_replay_estimate"
    names = [func] if func else ["combine", "combine_eta", "combine_forecast", "combiner"]
    fn = next((getattr(mod, n) for n in names if n and callable(getattr(mod, n, None))), None)
    if fn is None:
        raise SystemExit(f"{path}: no llm_replay_estimate and none of {names}")

    def est(k, a):
        n = _num(a.get("own_calls_left"))
        if n is None:
            return None
        state = dict(k["state"], prior=k["prior"], calls=k["calls_so_far"],
                     pace_spc=k["pace_spc"], pace_core_spc=k["pace_core_spc"])
        wf = dict(k["waits_forecast"] or {}, required_waits=a.get("required_waits") or [],
                  done_already=bool(a.get("done_already")))
        with patched_prior(mod, k["prior"]):
            r = fn(state, wf, 0 if a.get("done_already") is True else int(round(n)), k["pace_spc"])
        if r is None:
            return None
        return tuple(float(x) for x in r[:3])
    return est, f"{os.path.basename(path)}:{fn.__name__}"


def _rows(est, key, answers, ids):
    rows = []
    for cid in ids:
        k, a = key[cid], answers.get(cid)
        p = None
        if a is not None or est in (est_prior, est_oracle):
            try:
                p = est(k, a or {})
            except Exception:
                p = None
        if p is not None:
            p = tuple(max(0.0, float(x)) for x in p)
        st = dict(k["state"])
        rows.append((p, k["actual_remaining_s"], st, {"k": cid, "h": k["harness"], "wall": k["task_wall_s"]}))
    return rows


def _boot_closer(rows, ref, reps=1000, seed=3):
    pairs = [(R._err(p, a), R._err(q, a)) for (p, a, _, _), (q, _, _, _) in zip(rows, ref)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    rng = random.Random(seed)
    shares = []
    for _ in range(reps):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        shares.append(sum(1.0 if x < y else 0.5 if x == y else 0.0 for x, y in s) / len(s))
    shares.sort()
    return shares[int(0.05 * reps)], shares[int(0.95 * reps) - 1]


def _boot_median(rows, reps=1000, seed=1):
    errs = [R._err(p, a) for p, a, _, _ in rows]
    errs = [e for e in errs if e is not None]
    if len(errs) < 5:
        return None
    rng = random.Random(seed)
    meds = sorted(statistics.median([errs[rng.randrange(len(errs))] for _ in errs]) for _ in range(reps))
    return math.exp(meds[int(0.05 * reps)]), math.exp(meds[int(0.95 * reps) - 1])


def weight_curve(fn, key, answers, ids, reps=1000, seed=5):
    """Median miss of log blends count^w * prior^(1-w), with a paired bootstrap on the best weight."""
    usable = []
    for cid in ids:
        a = answers.get(cid)
        if a is None:
            continue
        try:
            c = fn(key[cid], a)
        except Exception:
            c = None
        if c is None:
            continue
        usable.append((c[1], key[cid]["prior"]["p50"], key[cid]["actual_remaining_s"]))
    if not usable:
        return None

    def errs(w, sample):
        return [abs(math.log(max(blend(c, p, w), FLOOR) / max(act, FLOOR))) for c, p, act in sample]
    curve = []
    for w in WEIGHTS:
        e = errs(w, usable)
        curve.append((w, statistics.median(e), sum(x <= math.log(2) for x in e) / len(e)))
    best_w = min(curve, key=lambda r: r[1])[0]
    rng = random.Random(seed)
    from collections import Counter
    wins = Counter()
    d_prior, d_count = [], []
    for _ in range(reps):
        s = [usable[rng.randrange(len(usable))] for _ in usable]
        meds = {w: statistics.median(errs(w, s)) for w in WEIGHTS}
        wins[min(meds, key=meds.get)] += 1
        d_prior.append(meds[best_w] - meds[0.0])
        d_count.append(meds[best_w] - meds[1.0])
    d_prior.sort()
    d_count.sort()
    ci = lambda d: (math.exp(d[int(0.05 * reps)]), math.exp(d[int(0.95 * reps) - 1]))  # noqa: E731
    return {"n": len(usable), "curve": curve, "best_w": best_w, "best_w_share": wins[best_w] / reps,
            "wins": {w: wins[w] / reps for w in WEIGHTS},
            "vs_prior_ratio_ci": ci(d_prior), "vs_count_ratio_ci": ci(d_count)}


def _verdict(wc):
    lo_p, hi_p = wc["vs_prior_ratio_ci"]
    lo_c, hi_c = wc["vs_count_ratio_ci"]
    b = wc["best_w"]
    beats_prior = hi_p < 1.0 or b == 0.0
    beats_count = hi_c < 1.0 or b == 1.0
    if b in (0.0, 1.0):
        other_hi = hi_c if b == 0.0 else hi_p
        return (f"best weight {b:.1f} is an endpoint; chosen in {wc['best_w_share']:.0%} of resamples")
    if hi_p < 1.0 and hi_c < 1.0:
        return (f"weight {b:.1f} is clearly better than both prior alone and count alone "
                f"(90% intervals of the median-miss ratio exclude 1)")
    return (f"weight {b:.1f} looks best but is not clearly better than "
            + ("prior alone" if not hi_p < 1.0 else "") + (" and " if not hi_p < 1.0 and not hi_c < 1.0 else "")
            + ("count alone" if not hi_c < 1.0 else "")
            + f" at n={wc['n']} (it wins only {wc['best_w_share']:.0%} of resamples)")


def count_accuracy(key, answers, ids):
    errs, signed, n0 = [], [], 0
    for cid in ids:
        a = answers.get(cid)
        if a is None:
            continue
        n = _num(a.get("own_calls_left"))
        if n is None:
            continue
        act = key[cid]["actual_calls_left"]
        r = math.log(max(n, 1.0) / max(act, 1.0))
        errs.append(abs(r))
        signed.append(r)
        n0 += 1
    if not errs:
        return None
    return {"n": n0, "med": math.exp(statistics.median(errs)), "w2x": sum(e <= math.log(2) for e in errs) / n0,
            "bias": math.exp(statistics.median(signed))}


GROUPS = [
    ("all", lambda k: "all"),
    ("length", lambda k: "task " + k["length_bucket"]),
    ("wait", lambda k: "wait open" if k["wait_open"] else "no wait open"),
    ("harness", lambda k: "harness " + k["harness"]),
    ("fraction", lambda k: "elapsed " + frac_bucket(k["frac"])),
]
GROUP_ORDER = (["all"] + [f"task {n}" for n, _, _ in LENGTHS] + ["wait open", "no wait open", "harness claude",
               "harness codex"] + [f"elapsed {frac_bucket(x)}" for x in (0.1, 0.5, 0.9)])


def score(args):
    key = load_key(args.key)
    ids = sorted(key)
    answer_sets = []
    for p in args.answers:
        ans, bad = load_answers(p)
        label = os.path.basename(os.path.normpath(p)) or "answers"
        if label in ("answers",) and len(args.answers) == 1:
            label = "model"
        answer_sets.append((label, ans, bad))
    out = []

    def say(line=""):
        print(line)
        out.append(line)
    say("# LLM replay results")
    say("")
    say(f"Generated by `python3 verify/llm_replay.py score` on {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "Numbers only; no transcript text.")
    say("")
    say(f"Cuts in key: {len(ids)}. Miss = |ln(mid / actual remaining)| as a factor, both floored at "
        f"{FLOOR:.0f} s. Median-miss 90% intervals and head-to-head intervals come from 1000 bootstrap "
        "resamples of cuts (one cut per task).")
    mod_est = None
    if args.module:
        mod_est = load_module_estimator(args.module, args.func)
    for label, answers, bad in answer_sets:
        answered = [c for c in ids if c in answers]
        say("")
        say(f"## Answer set: {label}")
        say("")
        say(f"Answers: {len(answered)} of {len(ids)} cuts"
            + (f"; unreadable files: {len(bad)}" if bad else "")
            + (f"; answers for unknown cuts ignored: {len(set(answers) - set(ids))}" if set(answers) - set(ids) else ""))
        ests = [("model minutes guess", est_minutes), ("count x pace", est_count_pace),
                ("count x pace + waits", est_count_pace_waits), ("history prior", est_prior)]
        for w in (0.3, 0.5, 0.7):
            ests.append((f"blend count+waits w={w}", make_blend(est_count_pace_waits, w)))
        if mod_est:
            ests.append((f"module {mod_est[1]}", mod_est[0]))
        ests.append(("oracle count x pace (reference)", est_oracle))
        rows = {name: _rows(fn, key, answers, ids) for name, fn in ests}
        refs = ["model minutes guess", "history prior"]
        say("")
        say("| estimator | answered | median miss (90% CI) | p90 miss | within 2x | in 80% range | median bias | "
            "closer than minutes guess (90% CI) | closer than prior (90% CI) |")
        say("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, _ in ests:
            m = R.score([(p, a) for p, a, _, _ in rows[name]])
            if not m.get("answer"):
                say(f"| {name} | 0% | | | | | | | |")
                continue
            ci = _boot_median(rows[name])
            ci_t = f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "n/a"
            cov = f"{m['cov']:.0%}" if name == "history prior" or name.startswith("module") else "-"
            cells = []
            for ref in refs:
                if ref == name:
                    cells.append("-")
                    continue
                share = R.closer_than(rows[name], rows[ref])
                bci = _boot_closer(rows[name], rows[ref])
                cells.append("-" if share is None else f"{share:.0%}" + (f" ({bci[0]:.0%}-{bci[1]:.0%})" if bci else ""))
            say(f"| {name} | {m['answer']:.0%} | {math.exp(m['med']):.2f}x ({ci_t}) | {math.exp(m['p90']):.1f}x | "
                f"{m['w2x']:.0%} | {cov} | {math.exp(m['bias']):.2f}x | {cells[0]} | {cells[1]} |")
        ca = count_accuracy(key, answers, ids)
        if ca:
            say("")
            say(f"Count alone (own_calls_left vs tool calls actually made after the cut, floored at 1): "
                f"n {ca['n']}, median miss {ca['med']:.2f}x, within 2x {ca['w2x']:.0%}, median bias {ca['bias']:.2f}x.")
        dn = sum(1 for c in answered if answers[c].get("done_already") is True)
        if answered:
            say(f"Answers saying done_already: {dn} (every cut had at least {MIN_REMAIN:.0f} s left).")
        say("")
        say("### Blend weight curves: exp(w ln(count-based) + (1 - w) ln(prior median))")
        for title, fn in (("count x pace + waits", est_count_pace_waits), ("count x pace", est_count_pace),
                          ("model minutes guess", est_minutes)):
            wc = weight_curve(fn, key, answers, ids)
            if not wc:
                continue
            say("")
            say(f"#### {title} (n={wc['n']})")
            say("")
            say("| w | median miss | within 2x | best in resamples |")
            say("|---:|---:|---:|---:|")
            for w, med, w2 in wc["curve"]:
                say(f"| {w:.1f} | {math.exp(med):.2f}x | {w2:.0%} | {wc['wins'][w]:.0%} |")
            lo_p, hi_p = wc["vs_prior_ratio_ci"]
            lo_c, hi_c = wc["vs_count_ratio_ci"]
            say("")
            say(f"Best w {wc['best_w']:.1f}. Ratio of its median miss to prior alone (w=0): 90% CI "
                f"{lo_p:.2f}-{hi_p:.2f}; to count alone (w=1): {lo_c:.2f}-{hi_c:.2f}. Verdict: {_verdict(wc)}.")
        say("")
        say("### By group")
        for name in [n for n, _ in ests if not n.startswith("blend")]:
            groups = {}
            for (p, a, st, meta), cid in zip(rows[name], ids):
                for _, g in GROUPS:
                    groups.setdefault(g(key[cid]), []).append((p, a))
            say("")
            say(f"#### {name}")
            say("")
            say("| group | cuts | answered | median miss | p90 miss | within 2x | median bias |")
            say("|---|---:|---:|---:|---:|---:|---:|")
            for g in GROUP_ORDER:
                if g not in groups:
                    continue
                m = R.score(groups[g])
                if not m.get("answer"):
                    say(f"| {g} | {m['n']} | 0% | | | | |")
                    continue
                say(f"| {g} | {m['n']} | {m['answer']:.0%} | {math.exp(m['med']):.2f}x | {math.exp(m['p90']):.1f}x | "
                    f"{m['w2x']:.0%} | {math.exp(m['bias']):.2f}x |")
    say("")
    say("Estimators: **model minutes guess** = minutes_guess. **count x pace** = own_calls_left x seconds per "
        "tool call so far (active time). **count x pace + waits** = own_calls_left x core pace (calls under "
        "90 s) + the waits.py forecast for already-open waits the answer says are required (or 0.75 x their "
        "age when waits.py saw nothing) + a history median per not-yet-started wait. **history prior** = "
        "baseline.py survival median fitted only on tasks that ended before the task began. **oracle** uses "
        "the true number of calls left and is a ceiling for perfect counting, not an estimator. "
        "\"in 80% range\" is shown only for estimators that state a range.")
    if not args.no_write:
        _write(args.out, "\n".join(out) + "\n")
        print(f"\nwrote {args.out}")


# ================================================================ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="sample cuts and write cut files, key and instructions")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--out", default=os.path.join(HERE, "llm_cuts"))
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--cache", help="task index to read (default ~/.eta/turns.jsonl, else a private copy in --out)")
    s = sub.add_parser("score", help="score answers against the key")
    s.add_argument("--answers", action="append", required=True,
                   help="directory of answer files or one file; repeat to compare answer sets")
    s.add_argument("--key", default=os.path.join(HERE, "llm_cuts", "key.json"))
    s.add_argument("--module", help="python file with llm_replay_estimate() or a combine() function")
    s.add_argument("--func", help="combiner function name inside --module")
    s.add_argument("--out", default=os.path.join(HERE, "LLM_REPLAY_RESULTS.md"))
    s.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args)
    else:
        score(args)


if __name__ == "__main__":
    os.umask(0o077)
    main()
