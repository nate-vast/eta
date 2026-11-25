#!/usr/bin/env python3
"""Estimate the working agent's remaining wall time and score it at completion.

The paired estimate uses measured batch progress or earlier confirmed comparable
tasks. Legacy call-count estimates and transcript-based pace reports remain
available. Local JSONL history lives in ~/.eta (override with ETA_HOME).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import stat
import statistics
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

GAP_CAP = 20 * 60          # a single quiet gap counts as at most this much active time
LONG_OP = 90               # tool calls at least this long are "long operations"
DEFAULT_SPC = 25.0         # seconds per tool call when nothing has been measured yet
MIN_CALLS = 5              # calls needed before trusting this task's own pace
MIN_CAL = 3                # resolved predictions needed before calibration kicks in
CAL_WINDOW = 40            # use the most recent N resolved predictions
STALE = 20 * 60            # a transcript quiet this long counts as finished
DEFAULT_SPREAD = (0.6, 1.8)

ETA_START = re.compile(r"^\s*(/eta\b|/skill:eta\b|\$eta\b|eta\b|eta[?:])", re.I)
ETA_ASK = re.compile(
    r"\b(eta\b|how long|how much longer|time left|time remaining|"
    r"when (will|is) (it|this|that|you)\b.{0,20}\b(done|finished|ready|complete))", re.I)
SYSTEM_PREFIXES = ("<task-notification>", "This session is being continued",
                   "<local-command", "Caveat: The messages below")


def home() -> str:
    return os.path.expanduser(os.environ.get("ETA_HOME", "~/.eta"))


def secure_history(directory, private_dir):
    """Restrict existing state without following links or chmodding a custom parent."""
    if os.name != "posix" or not os.path.lexists(directory):
        return
    if private_dir:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fchmod(fd, 0o700)
        finally:
            os.close(fd)
    for name in ("predictions.jsonl", "resolutions.jsonl", "done.lock", "ci-history.sqlite",
                 "ci-history.sqlite-journal", "ci-history.sqlite-shm", "ci-history.sqlite-wal"):
        path = os.path.join(directory, name)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            continue
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                sys.exit(f"Unexpected ETA history entry: {path}")
            if stat.S_IMODE(info.st_mode) != 0o600:
                os.fchmod(fd, 0o600)
        finally:
            os.close(fd)


def pred_path() -> str:
    return os.path.join(home(), "predictions.jsonl")


def res_path() -> str:
    return os.path.join(home(), "resolutions.jsonl")


# ---------------------------------------------------------------- utilities

def parse_ts(s):
    if not s:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None


def fmt_dur(s: float) -> str:
    s = max(0, int(round(s)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{round(s / 60)} min"
    h, m = divmod(round(s / 60), 60)
    return f"{h}h {m:02d}m"


def fmt_clock(ts: float, now: float | None = None) -> str:
    now = now or time.time()
    d = datetime.fromtimestamp(ts)
    clock = d.strftime("%I:%M %p").lstrip("0")
    days = (d.date() - datetime.fromtimestamp(now).date()).days
    if days == 1:
        return f"{clock} tomorrow"
    if days > 1:
        return d.strftime("%a %b %d ") + clock
    return clock


def one_line(text: str, n: int = 120) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def is_eta_prompt(text: str) -> bool:
    t = (text or "").strip()
    if ETA_START.match(t):
        return True
    return len(t) <= 120 and bool(ETA_ASK.search(t))


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def norm_status(s) -> str:
    s = str(s or "").lower()
    if s in ("completed", "complete", "done", "finished"):
        return "done"
    if s in ("in_progress", "in-progress", "active", "doing", "running"):
        return "doing"
    if s in ("deleted", "cancelled", "canceled", "removed"):
        return "deleted"
    return "pending"


def tool_desc(inp) -> str:
    if not isinstance(inp, dict):
        return one_line(str(inp), 70)
    for k in ("description", "command", "file_path", "pattern", "url", "query", "prompt", "skill"):
        v = inp.get(k)
        if v:
            return one_line(v if isinstance(v, str) else json.dumps(v), 70)
    return ""


# ---------------------------------------------------------------- transcript readers
# Both readers emit the same event shapes:
#   {"ts", "kind": "prompt", "text", "eta"}
#   {"ts", "kind": "agent"}
#   {"ts", "kind": "tool", "id", "name", "desc"}     tool call started
#   {"ts", "kind": "tool_done", "id"}                tool result came back
#   {"ts", "kind": "todos", "items": [[key, label, status], ...]}   full snapshot
#   {"ts", "kind": "task", "key", "label"|None, "status"|None}      single-item update

def _clean_prompt(content) -> str:
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") in ("text", "input_text"))
    text = content if isinstance(content, str) else ""
    m = re.search(r"<command-name>/?(.*?)</command-name>", text, re.S)
    if m:
        args = re.search(r"<command-args>(.*?)</command-args>", text, re.S)
        return ("/" + m.group(1).strip() + " " + (args.group(1).strip() if args else "")).strip()
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S).strip()
    # pi expands `/skill:name args` into the skill body wrapped in <skill name="..."> tags
    m = re.match(r'<skill name="([^"]+)"', text)
    if m:
        after = text.split("</skill>", 1)[1].strip() if "</skill>" in text else ""
        return ("/" + m.group(1) + " " + after).strip()
    return text


def read_claude(path):
    events, task_n = [], 0
    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = d.get("type")
            if typ not in ("user", "assistant") or d.get("isSidechain"):
                continue
            ts = parse_ts(d.get("timestamp"))
            if ts is None:
                continue
            msg = d.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if typ == "assistant":
                events.append({"ts": ts, "kind": "agent"})
                for b in content if isinstance(content, list) else []:
                    if not isinstance(b, dict) or b.get("type") != "tool_use":
                        continue
                    name, inp = b.get("name") or "?", b.get("input") or {}
                    events.append({"ts": ts, "kind": "tool", "id": b.get("id"),
                                   "name": name, "desc": tool_desc(inp)})
                    if name == "TodoWrite" and isinstance(inp.get("todos"), list):
                        items = [[t.get("content") or t.get("activeForm") or str(i),
                                  t.get("content") or "", norm_status(t.get("status"))]
                                 for i, t in enumerate(inp["todos"]) if isinstance(t, dict)]
                        events.append({"ts": ts, "kind": "todos", "items": items})
                    elif name == "TaskCreate":
                        task_n += 1
                        events.append({"ts": ts, "kind": "task", "key": str(task_n),
                                       "label": inp.get("subject") or inp.get("description") or "",
                                       "status": "pending"})
                    elif name == "TaskUpdate" and inp.get("taskId") is not None:
                        events.append({"ts": ts, "kind": "task", "key": str(inp.get("taskId")),
                                       "label": inp.get("subject"),
                                       "status": norm_status(inp["status"]) if inp.get("status") else None})
                continue
            # user records
            if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        events.append({"ts": ts, "kind": "tool_done", "id": b.get("tool_use_id")})
                continue
            if d.get("isMeta") or d.get("isCompactSummary"):
                continue
            if d.get("turnOrigin") in ("task_notification", "system"):
                continue
            text = _clean_prompt(content)
            if not text or text.startswith(SYSTEM_PREFIXES):
                continue
            events.append({"ts": ts, "kind": "prompt", "text": text, "eta": is_eta_prompt(text)})
    return events


CODEX_SKIP_ITEMS = {"UserMessage", "AgentMessage", "Reasoning"}


def read_codex(path):
    events, last_ts = [], None
    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(d.get("timestamp"))
            if ts is None:
                continue
            p = d.get("payload") or {}
            if d.get("type") == "event_msg" and p.get("type") == "item_completed":
                it = p.get("item") or {}
                k = it.get("type")
                if k == "UserMessage":
                    text = _clean_prompt([{"type": "text", "text": b.get("text", "")}
                                          for b in it.get("content") or [] if isinstance(b, dict)])
                    if text and not text.startswith(SYSTEM_PREFIXES):
                        events.append({"ts": ts, "kind": "prompt", "text": text,
                                       "eta": is_eta_prompt(text)})
                elif k in ("AgentMessage", "Reasoning"):
                    events.append({"ts": ts, "kind": "agent"})
                elif k:
                    # Codex logs tool items when they complete; start ~= previous event.
                    tid = it.get("id") or uuid.uuid4().hex
                    cmd = it.get("command")
                    desc = cmd[-1] if isinstance(cmd, list) and cmd else (
                        it.get("tool") or it.get("query") or it.get("path") or "")
                    start = last_ts if last_ts is not None else ts
                    events.append({"ts": start, "kind": "tool", "id": tid, "name": k,
                                   "desc": one_line(str(desc), 70)})
                    events.append({"ts": ts, "kind": "tool_done", "id": tid})
            elif d.get("type") == "response_item" and p.get("type") == "function_call" \
                    and p.get("name") == "update_plan":
                try:
                    plan = json.loads(p.get("arguments") or "{}").get("plan") or []
                except (json.JSONDecodeError, AttributeError):
                    plan = []
                items = [[t.get("step") or str(i), t.get("step") or "", norm_status(t.get("status"))]
                         for i, t in enumerate(plan) if isinstance(t, dict)]
                if items:
                    events.append({"ts": ts, "kind": "todos", "items": items})
            last_ts = ts
    events.sort(key=lambda e: e["ts"])
    return events


def read_pi(path):
    events = []
    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "message":
                continue
            ts = parse_ts(d.get("timestamp"))
            msg = d.get("message") or {}
            role, content = msg.get("role"), msg.get("content")
            if ts is None:
                continue
            if role == "user":
                text = _clean_prompt(content)
                if text and not text.startswith(SYSTEM_PREFIXES):
                    events.append({"ts": ts, "kind": "prompt", "text": text,
                                   "eta": is_eta_prompt(text)})
            elif role == "assistant":
                events.append({"ts": ts, "kind": "agent"})
                for b in content if isinstance(content, list) else []:
                    if isinstance(b, dict) and b.get("type") == "toolCall":
                        events.append({"ts": ts, "kind": "tool", "id": b.get("id"),
                                       "name": b.get("name") or "?",
                                       "desc": tool_desc(b.get("arguments") or {})})
            elif role == "toolResult":
                events.append({"ts": ts, "kind": "tool_done", "id": msg.get("toolCallId")})
    return events


def pi_sessions_dir() -> str:
    if os.environ.get("PI_CODING_AGENT_SESSION_DIR"):
        return os.path.expanduser(os.environ["PI_CODING_AGENT_SESSION_DIR"])
    return os.path.join(os.path.expanduser(os.environ.get("PI_CODING_AGENT_DIR", "~/.pi/agent")),
                        "sessions")


def sniff_kind(path) -> str:
    """Tell transcript formats apart by their first record, falling back to the path."""
    try:
        with open(path, errors="replace") as f:
            first = json.loads(f.readline() or "{}")
        if first.get("type") == "session_meta" or "payload" in first:
            return "codex"
        if first.get("type") == "session" and "cwd" in first:
            return "pi"
    except (OSError, json.JSONDecodeError):
        pass
    if "/.codex/" in path or os.path.basename(path).startswith("rollout-"):
        return "codex"
    if "/.pi/" in path:
        return "pi"
    return "claude"


READERS = {"claude": read_claude, "codex": read_codex, "pi": read_pi}


def read_events(path, kind=None):
    return READERS[kind or sniff_kind(path)](path)


def _codex_files(days=3):
    base = os.path.expanduser("~/.codex/sessions")
    out = []
    for i in range(days):
        d = datetime.now() - timedelta(days=i)
        out += glob.glob(os.path.join(base, d.strftime("%Y/%m/%d"), "rollout-*.jsonl"))
    return out


def _first_cwd(path, kind):
    """cwd recorded at the top of a Codex or pi session file."""
    try:
        with open(path, errors="replace") as f:
            for _ in range(5):
                d = json.loads(f.readline() or "{}")
                if kind == "codex" and d.get("type") == "session_meta":
                    return (d.get("payload") or {}).get("cwd")
                if kind == "pi" and d.get("type") == "session":
                    return d.get("cwd")
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _recent(paths, days=3):
    cutoff = time.time() - days * 86400
    return [p for p in paths if os.path.getmtime(p) >= cutoff]


def _all_candidates(agent=None):
    out = []
    if agent in (None, "claude"):
        out += [(p, "claude") for p in _recent(glob.glob(os.path.expanduser(
            "~/.claude/projects/*/*.jsonl")), 1)]
    if agent in (None, "codex"):
        out += [(p, "codex") for p in _codex_files(2)]
    if agent in (None, "pi"):
        out += [(p, "pi") for p in _recent(glob.glob(os.path.join(
            pi_sessions_dir(), "*", "*.jsonl")), 1)]
    return out


def find_transcript(transcript=None, session=None, agent=None, cwd=None):
    """Return (path, kind, how) for the live session, or (None, None, reason)."""
    if transcript:
        return os.path.expanduser(transcript), sniff_kind(transcript), "given"
    cwd = cwd or os.getcwd()
    sids = [s for s in (session, os.environ.get("CLAUDE_SESSION_ID"),
                        os.environ.get("CODEX_THREAD_ID")) if s and "${" not in s and "$" not in s]
    for sid in sids:
        if agent in (None, "claude"):
            hit = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{glob.escape(sid)}.jsonl"))
            if hit:
                return hit[0], "claude", "session id"
        if agent in (None, "codex"):
            hit = [p for p in _codex_files(30) if sid in os.path.basename(p)]
            if hit:
                return hit[0], "codex", "session id"
        if agent in (None, "pi"):
            hit = glob.glob(os.path.join(pi_sessions_dir(), "*", f"*{glob.escape(sid)}*.jsonl"))
            if hit:
                return hit[0], "pi", "session id"
    cands = []
    if agent in (None, "claude"):
        slug = re.sub(r"[^A-Za-z0-9]", "-", cwd)
        cands += [(p, "claude") for p in glob.glob(
            os.path.expanduser(f"~/.claude/projects/{glob.escape(slug)}/*.jsonl"))]
    for p, k in _all_candidates(agent):
        if k in ("codex", "pi") and _first_cwd(p, k) == cwd:
            cands.append((p, k))
    if cands:
        p, k = max(cands, key=lambda c: os.path.getmtime(c[0]))
        return p, k, "this folder's newest session"
    # Last resort: whatever transcript is being written right now.
    live = [c for c in _all_candidates(agent) if time.time() - os.path.getmtime(c[0]) < 180]
    if live:
        p, k = max(live, key=lambda c: os.path.getmtime(c[0]))
        others = len(live) - 1
        return p, k, ("newest live transcript (GUESS: " + (f"{others} other sessions also active, "
                      if others else "") + "pass --session to be sure)")
    return None, None, "no transcript found"


# ---------------------------------------------------------------- measuring

def reduce_todos(events, until):
    """Replay todo/task events up to `until`. Returns (items{key:[label,status]}, done_times{key:ts})."""
    items, done_at = {}, {}
    for e in events:
        if e["ts"] > until:
            break
        if e["kind"] == "todos":
            new = {}
            for key, label, st in e["items"]:
                new[key] = [label, st]
                if st == "done" and key not in done_at and items.get(key, [None, None])[1] != "done":
                    done_at[key] = e["ts"]
            items = new
        elif e["kind"] == "task":
            cur = items.setdefault(e["key"], [e.get("label") or "", "pending"])
            if e.get("label"):
                cur[0] = e["label"]
            if e.get("status"):
                if e["status"] == "deleted":
                    items.pop(e["key"], None)
                    continue
                if e["status"] == "done" and cur[1] != "done":
                    done_at[e["key"]] = e["ts"]
                cur[1] = e["status"]
    return items, done_at


def history_spc():
    xs = [p.get("spc") for p in load_jsonl(pred_path())
          if p.get("spc") and p.get("pace_source") == "task"]
    return statistics.median(xs[-30:]) if len(xs) >= 3 else None


def measure(events, from_n=None, now=None, started=None, calls_done=None):
    """Pace and progress for the current task. `started`/`calls_done` come from the agent itself
    when no transcript is readable (or override what the transcript says)."""
    now = now or time.time()
    prompts = [e for e in events if e["kind"] == "prompt"]
    end, eta_prompt = now, None
    if prompts and prompts[-1]["eta"]:
        eta_prompt = prompts[-1]
        end = eta_prompt["ts"]
    task_prompts = [p for p in prompts if not p["eta"] and p["ts"] <= end]
    idx = max(1, min(from_n or 1, len(task_prompts))) if task_prompts else 0
    task = task_prompts[-idx] if task_prompts else None
    start = task["ts"] if task else (events[0]["ts"] if events else now)
    if started:
        start = started
    win = [e for e in events if start <= e["ts"] <= end]

    active = 0.0
    for a, b in zip(win, win[1:]):
        if b["kind"] == "prompt":      # waiting on the user is not working time
            continue
        active += min(b["ts"] - a["ts"], GAP_CAP)

    starts = {e["id"]: e for e in win if e["kind"] == "tool"}
    durs = {}
    for e in win:
        if e["kind"] == "tool_done" and e["id"] in starts:
            durs[e["id"]] = e["ts"] - starts[e["id"]]["ts"]
    calls = len(starts)
    if calls_done is not None and (not events or calls_done > calls):
        calls = calls_done
        if not win:                 # self-reported: assume steady work since the start
            active = max(0.0, end - start)
    long_ops = sorted(((durs[i], starts[i]["name"], starts[i]["desc"])
                       for i in durs if durs[i] >= LONG_OP), reverse=True)
    long_total = sum(min(d, GAP_CAP) for d, _, _ in long_ops)
    open_tools = [starts[i] for i in starts if i not in durs and now - starts[i]["ts"] >= 10]

    hist = history_spc()
    if calls >= MIN_CALLS and active > 0:
        spc, pace_source = active / calls, "task"
        core_n = calls - len(long_ops)
        spc_core = (active - long_total) / core_n if core_n > 0 else spc
    else:
        spc = spc_core = hist or DEFAULT_SPC
        pace_source = "history" if hist else "default"

    items, done_at = reduce_todos(events, end)
    n_done = sum(1 for v in items.values() if v[1] == "done")
    n_doing = sum(1 for v in items.values() if v[1] == "doing")
    n_pending = sum(1 for v in items.values() if v[1] == "pending")
    done_in_win = sorted(t for k, t in done_at.items() if t >= start and k in items)
    step_s = steps_eta = None
    if done_in_win:
        step_s = (done_in_win[-1] - start) / len(done_in_win)
        steps_eta = step_s * (n_pending + 0.5 * n_doing)

    recent = []
    for i, p in enumerate(reversed(task_prompts[-6:]), 1):
        recent.append({"n": i, "ts": p["ts"], "text": one_line(p["text"], 90)})

    return {
        "now": now, "start": start, "end": end,
        "task": one_line(task["text"], 160) if task else None, "self_reported": not events,
        "task_n": idx, "eta_prompt": eta_prompt is not None,
        "elapsed": end - start, "active": active, "calls": calls,
        "spc": spc, "spc_core": spc_core, "pace_source": pace_source,
        "long_ops": [{"secs": d, "name": n, "desc": s} for d, n, s in long_ops[:5]],
        "open_tools": [{"name": t["name"], "desc": t["desc"], "running": now - t["ts"]}
                       for t in open_tools[-3:]],
        "todos": {"done": n_done, "doing": n_doing, "pending": n_pending,
                  "doing_labels": [one_line(v[0], 60) for v in items.values() if v[1] == "doing"],
                  "pending_labels": [one_line(v[0], 60) for v in items.values() if v[1] == "pending"],
                  "step_s": step_s, "steps_eta": steps_eta},
        "recent_prompts": recent,
        "last_event_age": now - events[-1]["ts"] if events else None,
    }


# ---------------------------------------------------------------- calibration

def find_finish(pred, now):
    """When did the work that `pred` was about actually stop? Returns (finish_ts|None, status)."""
    path = pred.get("transcript")
    if not path or not os.path.exists(path):
        return None, ("lost" if now - pred["created"] > 2 * 86400 else "pending")
    events = read_events(path, pred.get("kind"))
    after = [e for e in events if e["ts"] > pred["created"] + 1]
    nxt = next((e for e in after if e["kind"] == "prompt" and not e["eta"]), None)
    if nxt:
        body = [e for e in after if e["ts"] < nxt["ts"] and e["kind"] != "prompt"]
    else:
        last = events[-1]["ts"] if events else pred["created"]
        if now - last < STALE and now - os.path.getmtime(path) < STALE:
            return None, "pending"
        body = [e for e in after if e["kind"] != "prompt"]
    finish = body[-1]["ts"] if body else pred["created"]
    actual = finish - pred["created"]
    if actual < 30:
        return finish, "inconclusive"      # agent stopped right after answering: no signal
    if actual > 12 * 3600:
        return finish, "abandoned"
    return finish, "ok"


def resolve_pending(now=None):
    now = now or time.time()
    done = {r["id"] for r in load_jsonl(res_path())}
    for p in load_jsonl(pred_path()):
        if p["id"] in done or p.get("raw_s", 0) < 60 or now - p["created"] < 60:
            continue
        try:
            finish, status = find_finish(p, now)
        except Exception as exc:  # never let one bad transcript break /eta
            finish, status = None, f"error: {exc}"[:80]
        if status == "pending":
            continue
        actual = (finish - p["created"]) if finish else None
        append_jsonl(res_path(), {
            "id": p["id"], "resolved": now, "status": status, "finish": finish,
            "actual_s": actual, "ratio": (actual / p["raw_s"]) if actual and p.get("raw_s") else None,
        })


def calibration():
    preds = {p["id"]: p for p in load_jsonl(pred_path())}
    latest = {}
    for r in load_jsonl(res_path()):
        latest[r["id"]] = r
    rows = []
    for r in latest.values():
        p = preds.get(r["id"])
        if p and r.get("status") == "ok" and r.get("ratio"):
            rows.append((p, r))
    rows.sort(key=lambda pr: pr[0]["created"])
    rows = rows[-CAL_WINDOW:]
    ratios = [r["ratio"] for _, r in rows]
    n = len(ratios)
    hits = sum(1 for p, r in rows if p["low_s"] <= r["actual_s"] <= p["high_s"])
    out = {"n": n, "factor": 1.0, "lo": DEFAULT_SPREAD[0], "hi": DEFAULT_SPREAD[1],
           "hits": hits, "rows": rows}
    if n >= MIN_CAL:
        med = statistics.median(ratios)
        out["factor"] = min(3.0, max(0.33, med))
        if n >= 5:
            out["lo"] = min(0.85, quantile(ratios, 0.2) / med)
            out["hi"] = max(1.2, quantile(ratios, 0.8) / med)
    return out


# ---------------------------------------------------------------- commands

def parse_started(v, now=None):
    """'14:05', '2:05pm', '25m' / '1h10m' ago, or ISO -> epoch seconds."""
    if not v:
        return None
    now = now or time.time()
    v = v.strip().lower()
    m = re.fullmatch(r"(?:(\d+)h)?\s*(?:(\d+)m(?:in)?)?(?:\s*ago)?", v)
    if m and (m.group(1) or m.group(2)):
        return now - int(m.group(1) or 0) * 3600 - int(m.group(2) or 0) * 60
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(am|pm)?", v)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if m.group(3) == "pm" and h < 12:
            h += 12
        if m.group(3) == "am" and h == 12:
            h = 0
        d = datetime.fromtimestamp(now).replace(hour=h, minute=mi, second=0, microsecond=0)
        if d.timestamp() > now + 60:
            d -= timedelta(days=1)
        return d.timestamp()
    ts = parse_ts(v.upper() if "t" in v else v)
    if ts is None:
        sys.exit(f"can't read --started {v!r}; use 14:05, 2:05pm, 25m, or an ISO time")
    return ts


def _load(args):
    path, kind, how = find_transcript(args.transcript, args.session, args.agent)
    args.found_how = how
    if path and not os.path.exists(path):
        sys.exit(f"transcript not found: {path}")
    if not path:
        return None, None, []
    return path, kind, read_events(path, kind)


def _measure(args, events, now=None):
    return measure(events, args.from_n, now, parse_started(args.started, now), args.calls_done)


def render_measure(path, m, cal, how="", next_cmd="") -> str:
    L = ["ETA MEASUREMENTS"]
    if not path:
        L.append("transcript: none readable for this harness, so this is self-reported mode.")
        if m["calls"] == 0:
            L.append("  Rerun with --started <when the task began, e.g. 14:05 or 25m> and "
                     "--calls-done <tool calls you've made on it> from your own context.")
    else:
        L.append(f"transcript: {path}  (found by {how})")
        if m["last_event_age"] and m["last_event_age"] > STALE:
            L.append(f"WARNING: newest event is {fmt_dur(m['last_event_age'])} old; "
                     "this may be the wrong session (pass --session or --transcript).")
    if m["task"]:
        L.append(f"task (prompt #{m['task_n']}): \"{m['task']}\"")
        L.append(f"  started {fmt_clock(m['start'])}, {fmt_dur(m['elapsed'])} ago "
                 f"({fmt_dur(m['active'])} active work)")
    if len(m["recent_prompts"]) > 1:
        L.append("recent prompts (if the task began earlier, rerun with --from N):")
        for p in m["recent_prompts"]:
            L.append(f"  [{p['n']}] {fmt_clock(p['ts'])}  {p['text']}")
    L.append(f"tool calls so far: {m['calls']}")
    src = {"task": "measured on this task", "history": "your median from past tasks",
           "default": "default, nothing measured yet"}[m["pace_source"]]
    L.append(f"pace: {m['spc']:.0f} s per call all-in, {m['spc_core']:.0f} s per call "
             f"excluding long operations ({src})")
    for op in m["long_ops"]:
        L.append(f"  long op: {op['name']} {fmt_dur(op['secs'])}  {op['desc']}")
    for t in m["open_tools"]:
        L.append(f"  still running: {t['name']} for {fmt_dur(t['running'])}  {t['desc']}")
    td = m["todos"]
    if td["done"] + td["doing"] + td["pending"]:
        L.append(f"todo list: {td['done']} done, {td['doing']} in progress, {td['pending']} pending")
        for lab in td["doing_labels"]:
            L.append(f"  doing: {lab}")
        for lab in td["pending_labels"][:8]:
            L.append(f"  next:  {lab}")
        if td["steps_eta"] is not None:
            L.append(f"  step pace {fmt_dur(td['step_s'])} per finished step -> "
                     f"~{fmt_dur(td['steps_eta'])} left by step count (cross-check)")
    if cal["n"] >= MIN_CAL:
        L.append(f"calibration: {cal['n']} past ETAs checked; you finish at x{cal['factor']:.2f} "
                 f"your raw estimate (median); {cal['hits']}/{cal['n']} landed inside the range")
    else:
        L.append(f"calibration: {cal['n']} past ETAs checked (needs {MIN_CAL}); no correction yet")
    L.append("")
    L.append("REQUIRED NEXT STEP (do not reply to the user yet): list the remaining steps, count "
             "the tool calls each will really take (include test runs, fixes, verification, "
             "commits), add known long waits in minutes, then run:")
    L.append(f"  {next_cmd}")
    L.append("Reply only after it prints ETA RESULT, using its numbers.")
    return "\n".join(L)


def cmd_measure(args):
    resolve_pending()
    path, kind, events = _load(args)
    m = _measure(args, events)
    cal = calibration()
    if args.json:
        cal = {k: v for k, v in cal.items() if k != "rows"}
        print(json.dumps({"transcript": path, "kind": kind, "measure": m, "calibration": cal},
                         indent=2))
    else:
        flags = []
        if args.session and "$" not in args.session:
            flags.append(f'--session "{args.session}"')
        if args.transcript:
            flags.append(f'--transcript "{args.transcript}"')
        if args.agent:
            flags.append(f"--agent {args.agent}")
        if args.from_n:
            flags.append(f"--from {args.from_n}")
        if args.started:
            flags.append(f'--started "{args.started}"')
        if args.calls_done is not None:
            flags.append(f"--calls-done {args.calls_done}")
        script = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0].endswith(".py") else "~/.eta/eta.py"
        if script == os.path.abspath(os.path.expanduser("~/.eta/eta.py")):
            script = "~/.eta/eta.py"
        next_cmd = (f"python3 {script} estimate " + " ".join(flags) +
                    ' --calls <N> [--wait-min <M>] --note "<what is left>"').replace("  ", " ")
        print(render_measure(path, m, cal, args.found_how, next_cmd))


def paired_duration(seconds):
    """Keep the displayed point estimate close to what gets scored."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def confirmed_pairs():
    """Original paired forecasts with explicit finish markers, joined once per id."""
    resolutions = {r["id"]: r for r in load_jsonl(res_path())
                   if r.get("by") == "done" and r.get("status") == "ok"}
    return [(p, resolutions[p["id"]]) for p in load_jsonl(pred_path())
            if p.get("agent_s") is not None and p["id"] in resolutions]


def reference_scope(args):
    return [os.path.realpath(os.getcwd()), args.agent or "other", args.model.strip().casefold()]


def cmd_references(args):
    scope = reference_scope(args)
    labels = {p["cohort"][-1] for p, _ in confirmed_pairs() if p["cohort"][:-1] == scope}
    print("\n".join(sorted(labels)) if labels else "No confirmed reference labels in this scope.")


def reference_point(method, guess, samples):
    """Forecast variable work; samples contain (agent seconds, actual seconds)."""
    if method == "working agent":
        return guess
    if method == "agent and history":
        return (guess + statistics.median(actual for _, actual in samples)) / 2
    ratios = [actual / baseline for baseline, actual in samples if baseline > 0]
    return guess * statistics.median(ratios) if ratios else guess


def paired_forecast(pred, pairs):
    """At most one observation per earlier task. Reference labels describe remaining work."""
    by_run = {}
    for old, result in pairs:
        if (old["cohort"] != pred["cohort"] or old["run"] == pred["run"]
                or result["resolved"] > pred["created"] or result["actual_s"] < old["wait_s"]):
            continue
        # Use the first forecast for that stage of each task, even for repeated check-ins.
        if old["run"] not in by_run or old["created"] < by_run[old["run"]][0]["created"]:
            by_run[old["run"]] = old, result
    recent = sorted(by_run.values(), key=lambda pair: pair[1]["resolved"])[-CAL_WINDOW:]
    wait = pred["wait_s"]
    if pred.get("batch"):
        completed, total, sample_s = pred["batch"]
        remaining = sample_s * (total - completed) / completed
        return dict(est_s=wait + remaining, low_s=wait + remaining * DEFAULT_SPREAD[0],
                    high_s=wait + remaining * DEFAULT_SPREAD[1], n_cal=len(recent),
                    method="measured progress", basis=f"{total - completed:g} units left at "
                    f"{sample_s / completed:.2f}s per observed unit + {fmt_dur(wait)} of fixed waits")
    if pred.get("measured"):
        if len(recent) < MIN_CAL:
            return dict(est_s=None, low_s=None, high_s=None, n_cal=len(recent),
                        method="insufficient evidence", basis=f"{len(recent)} comparable verified finishes; need {MIN_CAL}")
        observed = [r["actual_s"] - p["wait_s"] for p, r in recent]
        return dict(est_s=wait + statistics.median(observed),
                    low_s=wait + quantile(observed, .1), high_s=wait + quantile(observed, .9),
                    n_cal=len(recent), method="verified task history",
                    basis=f"median remaining duration of {len(recent)} comparable verified tasks plus fixed waits")
    if len(recent) < MIN_CAL:
        point = max(wait, pred["agent_s"])
        return dict(est_s=point, low_s=wait + (point - wait) * DEFAULT_SPREAD[0],
                    high_s=wait + (point - wait) * DEFAULT_SPREAD[1], n_cal=len(recent),
                    method="agent fallback", basis=f"working agent estimate; {len(recent)} comparable finishes")
    samples = [(max(0, p["agent_s"] - p["wait_s"]), r["actual_s"] - p["wait_s"])
               for p, r in recent]
    guess = max(0, pred["agent_s"] - wait)
    # Estimate each historical task without its own result. On tied error prefer
    # the unadjusted agent. No outcome from the current task enters this choice.
    errors = {}
    for method in ("working agent", "agent and history", "agent bias correction"):
        residuals = []
        for i, (past_guess, actual) in enumerate(samples):
            estimate = reference_point(method, past_guess, samples[:i] + samples[i + 1:])
            residuals.append(actual - estimate)
        errors[method] = residuals
    method = min(errors, key=lambda name: sum(abs(e) for e in errors[name]))
    point = reference_point(method, guess, samples)
    low = max(0, point + min(0, quantile(errors[method], .1)))
    high = point + max(0, quantile(errors[method], .9))
    return dict(est_s=wait + point, low_s=wait + low, high_s=wait + high, n_cal=len(recent),
                method=method, basis=f"{method}, selected by absolute error on {len(recent)} "
                f"earlier comparable tasks; fixed wait {fmt_dur(wait)}")


def apply_command_plan(pred, args):
    """Time a known sequence of remaining commands without running any of them."""
    import job_history
    if not args.measured or not args.agent or args.tail_min is not None:
        sys.exit("--remaining-command needs --measured and --agent; include every remaining command")
    elapsed = args.elapsed_s or 0
    cwd = os.path.realpath(os.getcwd())
    files = job_history.recent_project_files(args.agent, cwd)
    history = {}
    point = low = high = pred["wait_s"]
    counts = []
    for index, command in enumerate(args.remaining_command):
        if command not in history:
            history[command] = job_history.local_durations(((p, args.agent) for p in files),
                                                          cwd, command, pred["created"])
        samples = job_history.remaining(history[command], elapsed if index == 0 else 0)
        if len(samples) < MIN_CAL:
            pred["basis"] += f"; remaining command {index + 1} has only {len(samples)} matching runs"
            return
        counts.append(len(samples))
        point += statistics.median(samples)
        low += quantile(samples, .1)
        high += quantile(samples, .9)
    pred.update(est_s=point, low_s=low, high_s=high, method="remaining command history",
                basis=f"{len(counts)} sequential commands, each timed from {min(counts)} or more "
                      f"previous successes; includes {paired_duration(pred['wait_s'])} in separate waits")


def apply_job_history(pred, args):
    import job_history
    import sqlite3

    step = None
    method = "job history"
    if args.command is not None:
        if args.elapsed_s is None or not args.agent:
            sys.exit("--command needs --elapsed-s and --agent")
        cwd = os.path.realpath(os.getcwd())
        files = job_history.recent_project_files(args.agent, cwd)
        durations = job_history.local_durations(((f, args.agent) for f in files),
                                               cwd, args.command, pred["created"])
        elapsed = args.elapsed_s
        samples = job_history.remaining(durations, elapsed)
    else:
        if args.elapsed_s is not None:
            sys.exit("--ci-job reads the attempt start time from GitHub; omit --elapsed-s")
        try:
            if args.checkout_ci is not None:
                history = job_history.checkout_workflow_history(args.checkout_ci, pred["created"])
            else:
                history = job_history.ci_history(args.ci_job, home(), pred["created"])
            samples, elapsed, state = history.remaining, history.elapsed, history.state
            step, method = history.step, history.method
        except job_history.WorkflowSelectionError as error:
            pred["basis"] += f"; {error}"
            return
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
            pred["basis"] += f"; CI history unavailable ({type(error).__name__})"
            return
        if state != "running":
            if state in ("queued", "waiting", "requested", "pending"):
                pred["basis"] += f"; CI job {state}, check again when it starts"
                return
            return state
    if len(samples) < MIN_CAL:
        pred["basis"] += f"; only {len(samples)} comparable successful runs exceeded {elapsed:.0f}s"
        return
    match = f"matching step {step!r}" if step else f"still active at {elapsed:.0f}s"
    tail = 60 * args.tail_min
    extra = tail + pred["wait_s"]
    point = statistics.median(samples)
    estimate_basis = "median time left"
    if step is not None and pred.get("measured"):
        if not history.late and history.elapsed_median is not None:
            point = history.elapsed_median
            estimate_basis = "whole-job median at the current elapsed time"
    elif step is not None:
        guess = max(0, pred["agent_s"] - extra)
        lower, upper = quantile(samples, .25), quantile(samples, .75)
        if lower <= guess <= upper:
            point = guess
            estimate_basis = "agent estimate within the historical middle half"
        elif history.late:
            point = min(upper, max(lower, guess))
            estimate_basis = "agent estimate bounded by late-step history"
        elif history.elapsed_median is not None:
            point = history.elapsed_median
            estimate_basis = "whole-job median at the current elapsed time"
    pred.update(est_s=extra + point, low_s=extra + min(point, quantile(samples, .1)),
                high_s=extra + max(point, quantile(samples, .9)), method=method,
                job_samples=len(samples), job_elapsed_s=elapsed, tail_s=tail,
                basis=f"{len(samples)} prior successful runs {match}; "
                      f"{estimate_basis} plus {paired_duration(extra)} for subsequent work and waits")


def cmd_paired(args, now):
    if not args.reference or not args.reference.strip() or not args.model or not args.model.strip():
        sys.exit("--agent-min needs --reference (remaining work) and --model")
    if args.calls is not None or args.minutes is not None:
        sys.exit("use --agent-min on its own with --reference and --model")
    if not args.task_id or not args.task_id.strip():
        sys.exit("--agent-min needs --task-id; reuse the id for the same job across check-ins")
    pid = uuid.uuid4().hex[:12]
    batch = args.batch
    if batch and (not all(math.isfinite(x) and x > 0 for x in batch) or batch[0] > batch[1]):
        sys.exit("--batch needs positive completed units, total units and elapsed seconds (completed <= total)")
    reference = " ".join(args.reference.casefold().split())
    pred = dict(id=pid, created=now, measured=args.measured, batch=batch, agent_s=args.agent_min * 60,
                wait_s=(args.wait_min or 0) * 60,
                cohort=reference_scope(args) + [reference], run=args.task_id,
                kind=args.agent, cwd=os.getcwd(),
                note=args.note or reference)
    pred.update(paired_forecast(pred, confirmed_pairs()))
    pred["event"] = args.event
    if args.remaining_command:
        apply_command_plan(pred, args)
    elif args.command is not None or args.ci_job or args.checkout_ci is not None:
        if args.tail_min is None:
            sys.exit("specify --tail-min for work after the command (0 if its success completes the task)")
        state = apply_job_history(pred, args)
        if state:
            message = f"CI job state: {state}. Reassess the remaining task before forecasting."
            print(json.dumps({"state": state, "recorded": False, "message": message}) if args.json else message)
            return
    elif args.tail_min is not None or args.elapsed_s is not None:
        sys.exit("--tail-min and --elapsed-s require a job history source")
    if not args.no_record:
        append_jsonl(pred_path(), pred)
    if pred["est_s"] is None:
        message = "No measured ETA yet. " + pred["basis"] + "."
        if args.json:
            print(json.dumps(dict(id=pid, recorded=not args.no_record, state="insufficient_evidence",
                                  est_s=None, finish=None, message=message, method=pred["method"], basis=pred["basis"])))
        else:
            print(message)
            print("Report the missing measurement and next checkpoint. Keep the agent baseline private; omit any rough guess or finish time.")
            if not args.no_record:
                print(f"after verified completion: python3 ~/.eta/eta.py done --id {pid}")
        return
    out = dict(id=pid, remaining=paired_duration(pred["est_s"]), finish=fmt_clock(now + pred["est_s"], now),
               low=paired_duration(pred["low_s"]), high=paired_duration(pred["high_s"]),
               recorded=not args.no_record, **{k: pred[k] for k in
               ("est_s", "agent_s", "low_s", "high_s", "method", "basis", "n_cal")})
    if pred.get("job_samples") is not None:
        out.update(job_samples=pred["job_samples"], job_elapsed_s=pred["job_elapsed_s"],
                   tail_s=pred["tail_s"])
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("ETA RESULT")
        print(f"remaining: ~{out['remaining']} (finish ~{out['finish']})")
        print(f"rough range: {out['low']} to {out['high']}")
        print(f"basis: {out['basis']}")
        if not args.no_record:
            print(f"after verified completion: python3 ~/.eta/eta.py done --id {pid}")


def cmd_estimate(args):
    for value in (args.agent_min, args.wait_min, args.calls, args.minutes, args.elapsed_s, args.tail_min):
        if value is not None and (not math.isfinite(value) or value < 0):
            sys.exit("durations and call counts must be finite and nonnegative")
    if args.agent_min is not None:
        return cmd_paired(args, time.time())
    if args.remaining_command or args.measured or args.batch or args.reference or args.model or args.task_id or args.command or args.ci_job or args.checkout_ci is not None or args.tail_min is not None or args.elapsed_s is not None:
        sys.exit("paired forecast options require --agent-min")
    if args.calls is None and args.minutes is None:
        sys.exit("estimate needs --calls N (preferred) or --minutes M")
    resolve_pending()
    path, kind, events = _load(args)
    now = time.time()
    m = _measure(args, events, now)
    cal = calibration()
    wait_s = (args.wait_min or 0) * 60
    if args.calls is not None:
        spc = m["spc_core"] if args.wait_min else m["spc"]
        raw = args.calls * spc + wait_s
        basis = f"{args.calls} tool calls x {spc:.0f} s"
        if wait_s:
            basis += f" + {fmt_dur(wait_s)} of waits"
    else:
        raw = args.minutes * 60 + wait_s
        basis = f"agent judgment {fmt_dur(args.minutes * 60)}" + (
            f" + {fmt_dur(wait_s)} of waits" if wait_s else "")
    work = max(0.0, raw - wait_s) * cal["factor"]
    est = wait_s + work
    low, high = wait_s + work * cal["lo"], wait_s + work * cal["hi"]
    if cal["n"] >= 10 and cal["hi"] / cal["lo"] <= 2.5 and m["pace_source"] == "task":
        conf = "high"
    elif cal["n"] >= MIN_CAL or m["pace_source"] == "task":
        conf = "medium"
    else:
        conf = "low"
    pid = uuid.uuid4().hex[:12]
    if not args.no_record:
        append_jsonl(pred_path(), {
            "id": pid, "created": now, "transcript": path, "kind": kind,
            "session": os.path.splitext(os.path.basename(path))[0] if path else None,
            "cwd": os.getcwd(), "task": m["task"], "task_start": m["start"],
            "calls": args.calls, "minutes": args.minutes, "wait_s": wait_s,
            "spc": m["spc"], "spc_core": m["spc_core"], "pace_source": m["pace_source"],
            "raw_s": raw, "factor": cal["factor"], "n_cal": cal["n"],
            "est_s": est, "low_s": low, "high_s": high, "note": args.note,
        })
    cal_txt = (f"calibration x{cal['factor']:.2f} from {cal['n']} past ETAs "
               f"({cal['hits']}/{cal['n']} inside range)") if cal["n"] >= MIN_CAL else \
        f"no calibration yet ({cal['n']}/{MIN_CAL} past ETAs checked)"
    out = {
        "id": pid, "remaining": fmt_dur(est), "low": fmt_dur(low), "high": fmt_dur(high),
        "finish": fmt_clock(now + est, now), "finish_low": fmt_clock(now + low, now),
        "finish_high": fmt_clock(now + high, now), "confidence": conf,
        "basis": f"{basis} = {fmt_dur(raw)} raw; {cal_txt}",
        "elapsed": fmt_dur(m["elapsed"]), "recorded": not args.no_record,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print("ETA RESULT")
    print(f"remaining:  ~{out['remaining']}  (likely {out['low']} to {out['high']})")
    print(f"finish:     ~{out['finish']}  (between {out['finish_low']} and {out['finish_high']})")
    print(f"confidence: {conf}")
    print(f"basis:      {out['basis']}")
    print(f"so far:     {out['elapsed']} on this task, {m['calls']} tool calls")
    print(f"logged:     {'yes, id ' + pid if out['recorded'] else 'no (--no-record)'}")
    print("NEXT: reply to the user with these numbers (ETA, finish time, range, what's done, "
          "what's left, basis). If the task isn't finished, do not end your turn: send the ETA "
          "text together with your next task tool call in the same message, then keep going. "
          "/eta is a check-in, not a stop.")


@contextmanager
def finish_lock():
    """Serialize simultaneous close notifications. The OS releases locks on exit."""
    with open(os.path.join(home(), "done.lock"), "a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def cmd_done(args):
    """The agent (or user) says the task is finished now: close its open predictions exactly."""
    now = time.time()
    if args.id:
        predictions = {p["id"]: p for p in load_jsonl(pred_path())}
        unknown = set(args.id) - predictions.keys()
        if unknown:
            sys.exit("unknown prediction id: " + ", ".join(sorted(unknown)))
        with finish_lock():
            closed = {r["id"] for r in load_jsonl(res_path()) if r.get("by") == "done"}
            for pid in dict.fromkeys(args.id):
                if pid in closed:
                    print(f"{pid} already closed")
                    continue
                p = predictions[pid]
                elapsed = now - p["created"]
                append_jsonl(res_path(), dict(id=pid, resolved=now, finish=now, actual_s=elapsed,
                                             status="ok", by="done",
                                             ratio=elapsed / p["raw_s"] if p.get("raw_s") else None))
                print(f"{pid} marked done: actual {fmt_dur(elapsed)}")
        return
    path, kind, how = find_transcript(args.transcript, args.session, args.agent)
    resolved = {r["id"] for r in load_jsonl(res_path()) if r.get("status") == "ok" and r.get("by") == "done"}
    cwd = os.getcwd()
    hits = []
    for p in load_jsonl(pred_path()):
        if p.get("agent_s") is not None or p["id"] in resolved or now - p["created"] > 12 * 3600:
            continue
        same = (path and p.get("transcript") == path) or (
            not path and not p.get("transcript") and p.get("cwd") == cwd)
        if same:
            hits.append(p)
    for p in hits:
        actual = now - p["created"]
        append_jsonl(res_path(), {"id": p["id"], "resolved": now, "status": "ok", "by": "done",
                                  "finish": now, "actual_s": actual,
                                  "ratio": actual / p["raw_s"] if p.get("raw_s") else None})
    if hits:
        last = hits[-1]
        print(f"marked done: {len(hits)} ETA(s) closed. Latest said {fmt_dur(last['est_s'])}, "
              f"took {fmt_dur(now - last['created'])}.")
    else:
        print("nothing to close (no open ETA for this session in the last 12 h).")


def long_task_stats(minutes):
    if not math.isfinite(minutes) or minutes <= 0:
        sys.exit("--min-remaining-min needs a positive finite duration")
    # One observation per task prevents heavily monitored tasks dominating the score.
    first = {}
    for pred, result in sorted(confirmed_pairs(), key=lambda pair: pair[0]["created"]):
        if pred.get("measured"):
            first.setdefault((tuple(pred["cohort"][:-1]), pred["run"]), (pred, result))
    eligible = [(p, r["actual_s"]) for p, r in first.values() if r["actual_s"] >= minutes * 60]
    issued = [(p, actual) for p, actual in eligible if p["est_s"] is not None]
    print(f"At least {minutes:g} minutes actually remained at first measured check-in: "
          f"{len(issued)}/{len(eligible)} verified tasks received an ETA.")
    if not issued:
        print("No paired long-task forecasts to score yet.")
        return
    for label, key in (("Skill", "est_s"), ("Agent", "agent_s")):
        misses = [p[key] - actual for p, actual in issued]
        overs = [e for e in misses if e > 0]
        print(f"{label}: mean absolute miss {paired_duration(statistics.mean(map(abs, misses)))}; "
              f"too long on {len(overs)}/{len(issued)} tasks"
              + (f" by {paired_duration(statistics.mean(overs))} on average." if overs else "."))
    wins = sum(abs(p["est_s"] - actual) < abs(p["agent_s"] - actual) for p, actual in issued)
    ties = sum(abs(p["est_s"] - actual) == abs(p["agent_s"] - actual) for p, actual in issued)
    print(f"Skill vs agent: {wins} closer, {ties} tied, {len(issued)-wins-ties} further.")


def cmd_stats(args):
    if args.min_remaining_min is not None:
        return long_task_stats(args.min_remaining_min)
    resolve_pending()
    cal = calibration()
    preds = load_jsonl(pred_path())
    res = {r["id"]: r for r in load_jsonl(res_path())}   # last resolution wins
    print(f"ETA TRACK RECORD  ({len(preds)} logged, {cal['n']} scored by legacy calibration)")
    pairs = confirmed_pairs()
    paired_count = sum(p.get("agent_s") is not None for p in preds)
    if paired_count:
        print(f"confirmed paired outcomes: {len(pairs)}/{paired_count}; "
              f"{paired_count - len(pairs)} awaiting verified completion")
    measured = [p for p in preds if p.get("measured")]
    if measured:
        print(f"measured ETA coverage: {sum(p['est_s'] is not None for p in measured)}/{len(measured)} check-ins")
    pairs = [(p, r) for p, r in pairs if p["est_s"] is not None]
    if pairs:
        skill = [abs(p["est_s"] - r["actual_s"]) for p, r in pairs]
        agent = [abs(p["agent_s"] - r["actual_s"]) for p, r in pairs]
        wins = sum(x < y for x, y in zip(skill, agent))
        ties = sum(x == y for x, y in zip(skill, agent))
        print(f"paired forecasts: {len(pairs)} explicitly closed; mean absolute error "
              f"skill {statistics.mean(skill):.1f}s, agent {statistics.mean(agent):.1f}s; "
              f"{wins} wins, {ties} ties, {len(pairs)-wins-ties} losses")
        hits = sum(p["low_s"] <= r["actual_s"] <= p["high_s"] for p, r in pairs)
        print(f"paired range coverage {hits}/{len(pairs)}; median absolute miss "
              f"skill {statistics.median(skill)/60:.2f}m, agent {statistics.median(agent)/60:.2f}m; "
              f"worst miss skill {max(skill)/60:.2f}m, agent {max(agent)/60:.2f}m")
    if cal["n"]:
        errs = [abs(r["actual_s"] - p["est_s"]) / max(r["actual_s"], 60) for p, r in cal["rows"]]
        print(f"inside predicted range: {cal['hits']}/{cal['n']} "
              f"({100 * cal['hits'] / cal['n']:.0f}%)   median error: {100 * statistics.median(errs):.0f}%")
        print(f"current correction: x{cal['factor']:.2f}   range band: "
              f"x{cal['lo']:.2f} to x{cal['hi']:.2f} around the estimate")
    pending = [p for p in preds if p["id"] not in res and p.get("raw_s", 0) >= 60]
    if pending:
        print(f"waiting to be checked: {len(pending)}")
    print()
    print(f"{'when':<16} {'said':>8} {'range':>17} {'actual':>8}  hit  task")
    for p in preds[-args.limit:]:
        r = res.get(p["id"])
        actual = fmt_dur(r["actual_s"]) if r and r.get("actual_s") and r["status"] == "ok" else (
            r["status"] if r else ("awaiting done" if p.get("agent_s") is not None else
             "too short" if p.get("raw_s", 0) < 60 else "pending"))
        if p["est_s"] is None:
            print(f"no measured ETA; actual {actual}; {one_line(p.get('note') or '', 50)}")
            continue
        hit = ""
        if r and r.get("status") == "ok":
            hit = "yes" if p["low_s"] <= r["actual_s"] <= p["high_s"] else "no"
        when = datetime.fromtimestamp(p["created"]).strftime("%m-%d %I:%M%p").lower()
        rng = f"{fmt_dur(p['low_s'])}-{fmt_dur(p['high_s'])}"
        print(f"{when:<16} {fmt_dur(p['est_s']):>8} {rng:>17} {actual:>8}  {hit:<4} "
              f"{one_line(p.get('note') or p.get('task') or '', 50)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eta", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--session", help="session id (Claude Code: ${CLAUDE_SESSION_ID})")
        p.add_argument("--transcript", help="explicit transcript .jsonl path")
        p.add_argument("--agent", choices=["claude", "codex", "pi"],
                       help="limit discovery to one harness")
        p.add_argument("--from", dest="from_n", type=int,
                       help="task started at recent prompt N (1 = latest non-ETA prompt)")
        p.add_argument("--json", action="store_true")
        p.add_argument("--started", help="any harness: when the task began (14:05, 2:05pm, 25m, ISO)")
        p.add_argument("--calls-done", type=int, help="any harness: tool calls made on the task so far")

    p = sub.add_parser("measure", help="read the live transcript and print pace + context")
    common(p)
    p.set_defaults(fn=cmd_measure)
    p = sub.add_parser("estimate", help="compute, print and log the ETA")
    common(p)
    p.add_argument("--measured", action="store_true", help="use observations only; agent minutes are a scoring baseline")
    p.add_argument("--agent-min", type=float, help="working agent's independent total remaining minutes")
    p.add_argument("--reference", help="repeatable description of the remaining work and its scale")
    p.add_argument("--model", help="working agent model identity")
    sources = p.add_mutually_exclusive_group()
    sources.add_argument("--remaining-command", action="append", metavar="COMMAND",
                         help="repeat for each sequential command through verified completion")
    sources.add_argument("--command", help="exact local shell command, measured from project transcripts")
    sources.add_argument("--checkout-ci", nargs="?", const="", metavar="WORKFLOW",
                         help="discover a running GitHub workflow for the checked-out commit")
    sources.add_argument("--ci-job", help="GitHub Actions job URL")
    p.add_argument("--elapsed-s", type=float, help="observed time since this command attempt started")
    p.add_argument("--tail-min", type=float, help="estimated work after successful job completion")
    p.add_argument("--event", default="check-in", choices=["check-in", "started", "retry", "overrun", "scope-change"])
    sources.add_argument("--batch", type=float, nargs=3, metavar=("DONE", "TOTAL", "SECONDS"),
                   help="homogeneous batch accounting for all variable work: observed units done, total, elapsed seconds")
    p.add_argument("--task-id", help="stable identifier across forecasts for one task")
    p.add_argument("--calls", type=int, help="tool calls still needed")
    p.add_argument("--wait-min", type=float, help="known long waits (builds, downloads, CI), minutes")
    p.add_argument("--minutes", type=float, help="direct estimate when calls can't be counted")
    p.add_argument("--note", help="what is left, a few words")
    p.add_argument("--no-record", action="store_true", help="don't log this prediction")
    p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("done", help="the task just finished: close its open ETAs (exact calibration)")
    p.add_argument("--session")
    p.add_argument("--transcript")
    p.add_argument("--agent", choices=["claude", "codex", "pi"])
    p.add_argument("--id", nargs="+", help="explicitly close these verified predictions")
    p.set_defaults(fn=cmd_done)
    p = sub.add_parser("references", help="show this model's confirmed remaining-work labels in this project")
    p.add_argument("--agent", choices=["claude", "codex", "pi"])
    p.add_argument("--model", required=True)
    p.set_defaults(fn=cmd_references)
    p = sub.add_parser("stats", help="past ETAs vs actual finish times")
    p.add_argument("--min-remaining-min", type=float, help="compare first measured check-ins by verified remaining time")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    # Applies to new ledgers, locks and SQLite sidecars, including under ETA_HOME.
    os.umask(0o077)
    secure_history(home(), private_dir=os.path.abspath(home()) == os.path.expanduser("~/.eta"))
    main()
