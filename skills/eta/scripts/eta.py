#!/usr/bin/env python3
"""/eta: when will the working agent's current task be done?

    python3 ~/.eta/eta.py now --steps N [--note "what is left"]     the front door, always answers
    python3 ~/.eta/eta.py done [--id ID ...]                        close it at verified completion
    python3 ~/.eta/eta.py stats                                     track record

`now` finds the live transcript itself, reads what the task has done so far, what it is
waiting on (waits.py) and how long tasks like it ran on this machine (baseline.py), turns
the agent's count of its own remaining tool calls into time with the pace measured on
this task, and blends the parts into one midpoint and an 80% range (combine()).
The agent never states time. Every prediction and each of its parts is logged in
~/.eta (override with ETA_HOME) so each part can be scored on its own later.
Python 3.9+, stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sqlite3
import stat
import statistics
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:            # job_history, baseline and waits live next to this file
    sys.path.insert(0, HERE)
# baseline.py and waits.py `import eta`; let them share this module instead of loading a copy.
if __name__ in sys.modules:
    sys.modules.setdefault("eta", sys.modules[__name__])

GAP_CAP = 20 * 60          # a single quiet gap counts as at most this much active time
STALE = 20 * 60            # a transcript quiet this long counts as finished
ASK_WINDOW = 15 * 60       # an /eta ask in the transcript this recent is the one being answered

# ---- pace: seconds of the agent's own work per tool call
PACE_GAP_CAP = 300.0       # chosen: a quiet gap counts as at most 5 min of own work
MIN_CALLS = 5              # calls needed before trusting this task's own pace
DEFAULT_PACE = 25.0        # used only when there is no history at all

# ---- combiner (see combine() and verify/COMBINE.md for where each number comes from)
W_STEPS = 0.5              # CHOSEN, not fit: weight of (own work + open waits) vs the history
                           # prior, in log space, when the agent gives a step count. A real count
                           # cannot be replayed without a model; verify/llm_replay.py tests this.
                           # With the true count (oracle) 0.5 to 1.0 all beat history alone.
W_WAITS = 0.0              # FIT (verify/COMBINE.md): weight of waits (+ tail) vs the prior when
                           # there is no step count. Every weight above 0 made the midpoint worse,
                           # so without a count the answer is the history prior.
USE_TAIL = True            # FIT: with W_WAITS > 0 the tail helped check-in asks, hurt one-off asks
WIDTH = 1.3                # FIT: scale on blended log half-widths; 80% coverage on the replay
                           # with oracle counts given 2x random noise (an assumption)
STEP_SPREAD = (0.5, 2.0)   # CHOSEN: own work could take half to twice the count x pace
CAP_FROM_P90 = 3.0         # CHOSEN: sanity bound when the prior has no p98: 3x its p90
CAP_Q = 0.98               # sanity bound: history's 98th percentile at this elapsed time
FLOOR_S = 15.0             # nothing finishes in less time than one more reply

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
                 "ci-history.sqlite-journal", "ci-history.sqlite-shm", "ci-history.sqlite-wal",
                 "turns.jsonl", "waits-cache.json", "waits-progress.json"):
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
    if days < 0:
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
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    return out


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


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


# ---------------------------------------------------------------- transcripts
# Claude Code and Codex transcripts are parsed by baseline.py (one shared parser for prompts,
# tool calls, interrupts, notifications and turn ends). pi's message stream is read here.

def read_pi(path):
    """pi session -> [{"ts", "kind": prompt|agent|tool|tool_done, ...}] in file order."""
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
                                       "name": b.get("name") or "?"})
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
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    if "/.codex/" in path or os.path.basename(path).startswith("rollout-"):
        return "codex"
    if "/.pi/" in path:
        return "pi"
    return "claude"


def _codex_glob(sid):
    base = os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex"))
    return (glob.glob(os.path.join(base, "sessions", "*", "*", "*", f"rollout-*{glob.escape(sid)}.jsonl"))
            + glob.glob(os.path.join(base, "archived_sessions", f"rollout-*{glob.escape(sid)}.jsonl")))


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


def t3_db_path():
    return os.environ.get("T3_STATE_DB", os.path.expanduser("~/.t3/userdata/statev2.sqlite"))


def _t3_query(sql, args=()):
    """Read-only query on the T3 Code state database; [] when it is absent or busy."""
    path = t3_db_path()
    if not os.path.isfile(path):
        return []
    try:
        db = sqlite3.connect("file:" + path + "?mode=ro", uri=True, timeout=0.5)
    except sqlite3.Error:
        return []
    try:
        return db.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        db.close()


def t3_native_ids(thread_id):
    """Provider session ids (Claude session id, Codex thread id) of a T3 thread, newest run first."""
    rows = _t3_query("""SELECT json_extract(payload_json, '$.nativeThreadRef.nativeId')
        FROM orchestration_v2_projection_provider_threads WHERE thread_id=?
        ORDER BY COALESCE(last_run_ordinal, 0) DESC, updated_at DESC""", (thread_id,))
    return [r[0] for r in rows if r[0]]


def t3_child_sessions():
    """Native session ids of T3 delegate children. They share the parent's project folder, so a
    newest-file guess could otherwise pick a child that is still working while the parent waits."""
    rows = _t3_query("""SELECT json_extract(p.payload_json, '$.nativeThreadRef.nativeId')
        FROM orchestration_v2_projection_provider_threads p
        JOIN orchestration_v2_projection_threads t ON t.thread_id = p.thread_id
        WHERE json_extract(t.payload_json, '$.lineage.relationshipToParent') = 'subagent'""")
    return {r[0] for r in rows if r[0]}


def _by_session_id(sid, agent=None):
    if agent in (None, "claude"):
        hit = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{glob.escape(sid)}.jsonl"))
        if hit:
            return hit[0], "claude"
    if agent in (None, "codex"):
        hit = _codex_glob(sid)
        if hit:
            return hit[0], "codex"
    if agent in (None, "pi"):
        hit = glob.glob(os.path.join(pi_sessions_dir(), "*", f"*{glob.escape(sid)}*.jsonl"))
        if hit:
            return hit[0], "pi"
    return None, None


def _usable_id(s):
    return bool(s) and "$" not in s and "{" not in s and "<" not in s


def find_transcript(transcript=None, session=None, agent=None, cwd=None, now=None):
    """Return (path, kind, how) for the live session, or (None, None, reason).

    Order: an explicit path; a session id (argument, then CLAUDE_CODE_SESSION_ID,
    CLAUDE_SESSION_ID, CODEX_THREAD_ID); the T3 thread in T3CODE_THREAD_ID mapped to its
    provider session through T3's state database; this folder's newest session that is not
    a T3 delegate child; the newest transcript written in the last 3 minutes (a guess)."""
    if transcript:
        p = os.path.expanduser(transcript)
        return p, sniff_kind(p), "given"
    now = now or time.time()
    cwd = cwd or os.getcwd()
    ids = [(session, "--session")] if _usable_id(session) else []
    for var, harness in (("CLAUDE_CODE_SESSION_ID", "claude"), ("CLAUDE_SESSION_ID", "claude"),
                         ("CODEX_THREAD_ID", "codex")):
        if _usable_id(os.environ.get(var)) and agent in (None, harness):
            ids.append((os.environ[var], var))
    for sid, how in ids:
        p, k = _by_session_id(sid, agent)
        if p:
            return p, k, f"session id ({how})"
    if _usable_id(os.environ.get("T3CODE_THREAD_ID")):
        for sid in t3_native_ids(os.environ["T3CODE_THREAD_ID"]):
            p, k = _by_session_id(sid, agent)
            if p:
                return p, k, "T3 thread"
    children = None
    cands = []
    if agent in (None, "claude"):
        slug = re.sub(r"[^A-Za-z0-9]", "-", cwd)
        cands += [(p, "claude") for p in glob.glob(
            os.path.expanduser(f"~/.claude/projects/{glob.escape(slug)}/*.jsonl"))]
    if agent in (None, "codex"):
        for i in range(3):
            d = datetime.fromtimestamp(now) - timedelta(days=i)
            for p in glob.glob(os.path.join(os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex")),
                                            "sessions", d.strftime("%Y/%m/%d"), "rollout-*.jsonl")):
                if _first_cwd(p, "codex") == cwd:
                    cands.append((p, "codex"))
    if agent in (None, "pi"):
        cands += [(p, "pi") for p in glob.glob(os.path.join(pi_sessions_dir(), "*", "*.jsonl"))
                  if _first_cwd(p, "pi") == cwd]
    if cands:
        children = t3_child_sessions()
        own = [c for c in cands if os.path.splitext(os.path.basename(c[0]))[0] not in children
               and not any(ch in os.path.basename(c[0]) for ch in children if c[1] == "codex")]
        if own:
            p, k = max(own, key=lambda c: os.path.getmtime(c[0]))
            return p, k, "this folder's newest session (GUESS: pass --session to be sure)"
    live = []
    for pat, k in ((os.path.expanduser("~/.claude/projects/*/*.jsonl"), "claude"),
                   (os.path.join(pi_sessions_dir(), "*", "*.jsonl"), "pi")):
        if agent in (None, k):
            live += [(p, k) for p in glob.glob(pat) if now - os.path.getmtime(p) < 180]
    if live:
        p, k = max(live, key=lambda c: os.path.getmtime(c[0]))
        return p, k, f"newest live transcript (GUESS: {len(live) - 1} other sessions also active)"
    return None, None, "no transcript found"


# ---------------------------------------------------------------- the task right now

def own_pace(task, e):
    """(seconds of own work per tool call or None, calls so far, own-work seconds) at e seconds in.

    Own work = quiet gaps between the agent's actions, each capped at PACE_GAP_CAP, leaving out
    gaps spent inside a delegate, a wait tool or a question to the user (those are waits)."""
    calls = [c for c in task.get("c") or [] if c[0] <= e]
    waits_iv = [(s, (end if end != -1 else 1e18)) for s, end, cls in calls if cls in "dwa"]
    pts = sorted([0.0] + [a for a in task.get("a") or [] if a <= e])
    work = 0.0
    for x, y in zip(pts, pts[1:]):
        if y <= x or any(s <= x and y <= f for s, f in waits_iv):
            continue
        work += min(y - x, PACE_GAP_CAP)
    n = len(calls)
    return (work / n if n >= MIN_CALLS and work > 0 else None), n, work


def history_pace(tasks, harness=None, model=None):
    """Median own-work seconds per call over finished tasks: same model family and harness when
    30 tasks qualify, else the harness, else everything. Returns (pace, basis)."""
    import baseline as B
    rows = []
    for t in tasks:
        if not B.fit_eligible(t) or t["n"] < MIN_CALLS:
            continue
        p, _, _ = own_pace(t, t["wall"])
        if p:
            rows.append((t["h"], B.model_family(t.get("m")), p))
    fam = B.model_family(model) if model else None
    for label, keep in ((f"{harness} {fam} tasks", lambda r: r[0] == harness and r[1] == fam),
                        (f"{harness} tasks", lambda r: r[0] == harness),
                        ("all tasks", lambda r: True)):
        xs = [r[2] for r in rows if keep(r)]
        if len(xs) >= 30 or (label == "all tasks" and xs):
            return statistics.median(xs), f"median of {len(xs)} past {label}"
    return DEFAULT_PACE, "default, no history"


def _ask_info(events, task, now):
    """When was the /eta being answered asked, and had the agent already ended its turn by then?"""
    t0 = task["t0"] if task else 0.0
    asks = [e[1] for e in events if e[0] == "P" and e[2] == "eta" and t0 <= e[1] <= now + 1]
    ask = asks[-1] if asks and now - asks[-1] <= ASK_WINDOW else None
    if ask is None:
        return None, None
    ended_at = None
    for e in events:
        if e[1] >= ask:
            break
        if e[0] == "A":
            ended_at = e[1] if e[2] else None
        elif e[0] == "T":
            ended_at = None
    return ask, ended_at


def session_state(path, kind, now):
    """The newest task in a transcript as a live observer sees it at `now`.

    Returns baseline's state (elapsed_s, calls, running tool, delegates open, ...) plus
    started, task (the cached-shape timeline), ask_ts, turn_ended_at (set when the agent had
    already ended its turn before this ask arrived), last_ts, model."""
    import baseline as B
    events, meta = B.PARSERS[kind](path)
    events = [e for e in events if e[1] <= now]
    tasks = B.split_tasks(events, kind, live=True)
    task = tasks[-1] if tasks else None
    if task is None:
        return {"elapsed_s": 0.0, "harness": kind, "calls": 0, "task": None, "started": None,
                "ask_ts": None, "turn_ended_at": None, "last_ts": None, "model": None}
    st = B.state_at(task, max(0.0, now - task["t0"]))
    ask, ended = _ask_info(events, task, now)
    st.update(started=task["t0"], task=task, ask_ts=ask, turn_ended_at=ended,
              last_ts=task["t0"] + task["wall"], model=task.get("m"), cwd=meta.get("cwd"))
    return st


# ---------------------------------------------------------------- the combiner

def _seq(parts):
    """Sum of (low, mid, high) parts done one after another: mids add, spreads add in quadrature."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    mid = sum(p[1] for p in parts)
    lo = mid - math.sqrt(sum((p[1] - p[0]) ** 2 for p in parts))
    hi = mid + math.sqrt(sum((p[2] - p[1]) ** 2 for p in parts))
    return (max(lo, max(p[0] for p in parts), 0.0), mid, hi)


def _trip(d, lo="low_s", mid="p50_s", hi="high_s"):
    return (float(d[lo]), float(d[mid]), float(d[hi])) if d and d.get(mid) is not None else None


def combine(state, waits_forecast=None, own_steps=None, pace=None, *, w_steps=None, w_waits=None,
            width=None, use_tail=None):
    """Blend history, waits and the agent's own step count into (low, mid, high, parts), seconds.

    state["prior"]:   history prior from baseline.remaining_given (p10, p50, p90; optional cap =
                      its 98th percentile). Required: this function never reads history itself.
    waits_forecast:   waits.forecast() dict or None; uses ["wait"] and ["tail"].
    own_steps, pace:  tool calls of the agent's own work left and seconds per call, or None.

    1. Component sum C, done one after another: the open waits, then either own work
       (own_steps x pace, range STEP_SPREAD) or, without a step count, waits.py's parent tail.
       Own work is pulled in to the history cap if it alone would exceed it (flagged).
    2. No component (nothing open, no count): the prior is the answer.
    3. Otherwise blend in log space: ln mid = w ln C_mid + (1 - w) ln P_mid, w = W_STEPS with a
       step count, else W_WAITS. Lower and upper log half-widths blend the same way and are
       scaled by WIDTH. Every constant is a module-level name; see the comments there."""
    w_steps = W_STEPS if w_steps is None else w_steps
    w_waits = W_WAITS if w_waits is None else w_waits
    width = WIDTH if width is None else width
    use_tail = USE_TAIL if use_tail is None else use_tail
    prior = state["prior"]
    P = (max(float(prior["p10"]), 1.0), max(float(prior["p50"]), FLOOR_S), max(float(prior["p90"]), FLOOR_S))
    cap = float(prior.get("cap") or P[2] * CAP_FROM_P90)
    parts = {"prior": {"low_s": P[0], "mid_s": P[1], "high_s": P[2], "cap_s": cap, "n": prior.get("n")},
             "flags": []}
    wf = waits_forecast or {}
    wait = _trip(wf.get("wait"))
    if wait and wait[1] <= 0 and wait[2] <= 0:
        wait = None
    comps = []
    if wait:
        comps.append(wait)
        parts["wait"] = {"low_s": wait[0], "mid_s": wait[1], "high_s": wait[2]}
    if own_steps is not None:
        own_mid = max(0.0, float(own_steps)) * float(pace or DEFAULT_PACE)
        if own_mid > cap:
            parts["flags"].append(f"the step count implies {fmt_dur(own_mid)} of own work, more than "
                                  f"{fmt_dur(cap)}, which history almost never sees at this point; "
                                  f"pulled in to {fmt_dur(cap)}")
            own_mid = cap
        own = (own_mid * STEP_SPREAD[0], own_mid, own_mid * STEP_SPREAD[1])
        comps.append(own)
        parts["own"] = {"low_s": own[0], "mid_s": own[1], "high_s": own[2],
                        "steps": own_steps, "pace_s": pace}
        w = w_steps
    else:
        tail = _trip(wf.get("tail")) if wait and use_tail else None
        if tail:
            comps.append(tail)
            parts["tail"] = {"low_s": tail[0], "mid_s": tail[1], "high_s": tail[2]}
        w = w_waits
        if w <= 0:
            comps = []        # the fitted default: without a count, waits do not move the number
    C = _seq(comps)
    if C is None:
        low, mid, high = P
        parts["method"] = "history"
    else:
        c = (max(C[0], 1.0), max(C[1], FLOOR_S), max(C[2], FLOOR_S))
        parts["component"] = {"low_s": C[0], "mid_s": C[1], "high_s": C[2], "weight": w}
        lm = w * math.log(c[1]) + (1 - w) * math.log(P[1])
        down = w * math.log(c[1] / min(c[0], c[1])) + (1 - w) * math.log(P[1] / min(P[0], P[1]))
        up = w * math.log(max(c[2], c[1]) / c[1]) + (1 - w) * math.log(max(P[2], P[1]) / P[1])
        mid = math.exp(lm)
        low, high = mid * math.exp(-width * down), mid * math.exp(width * up)
        parts["method"] = "steps+waits+history" if own_steps is not None and wait else \
            "steps+history" if own_steps is not None else "waits+history"
    mid = max(mid, FLOOR_S)
    low, high = min(low, mid), max(high, mid)
    return low, mid, high, parts


# ---------------------------------------------------------------- evidence the agent names

def ci_wait(ci, now):
    """GitHub Actions as a wait: '' = the workflow running for this checkout, or a job URL."""
    import job_history
    try:
        if ci.startswith("https://"):
            h = job_history.ci_history(ci, home(), now)
        else:
            h = job_history.checkout_workflow_history(ci or None, now)
    except job_history.WorkflowSelectionError as error:
        return None, f"GitHub Actions: {error}"
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
        return None, f"GitHub Actions timing unavailable ({type(error).__name__})"
    if h.state != "running":
        return None, f"GitHub Actions is {h.state}"
    if len(h.remaining) < 3:
        return None, f"GitHub Actions: only {len(h.remaining)} earlier successful runs lasted this long"
    p50 = statistics.median(h.remaining)
    if h.step and not h.late and h.elapsed_median is not None:
        p50 = h.elapsed_median
    return dict(kind="ci", label="GitHub Actions" + (f" step {h.step}" if h.step else ""), model=None,
                started=now - h.elapsed, elapsed_s=h.elapsed, progress=None, p50_s=p50,
                low_s=min(p50, quantile(h.remaining, .1)), high_s=max(p50, quantile(h.remaining, .9)),
                n=len(h.remaining), basis=f"{len(h.remaining)} earlier successful runs still going at "
                f"{fmt_dur(h.elapsed)}", id="ci", source="github", background=True), None


def progress_wait(done, total, seconds, now):
    import waits as W
    est = W.progress_estimate({"done": done, "total": total, "unit": "units"}, seconds)
    if not est:
        return None
    return dict(kind="batch", label="measured progress", model=None, started=now - seconds,
                elapsed_s=seconds, progress={"done": done, "total": total}, p50_s=est["p50_s"],
                low_s=est["low_s"], high_s=est["high_s"], n=est["n"], basis=est["basis"],
                id="progress", source="agent", background=True)


# ---------------------------------------------------------------- the front door

def _prior(state, at=None):
    """History prior at this elapsed time; with `at`, only tasks that had finished by then."""
    import baseline as B
    if at is None:
        base = B.default_baseline()
        tasks = None
    else:
        tasks = [t for t in B.tasks_from_records(list(B.load_cache().values()) or B.build(), at)
                 if t["t0"] + t["wall"] < at]
        base = B.Baseline(tasks)
    r = base.remaining_given(state)
    (cap,), _, _ = base.quantiles(state, (CAP_Q,))
    r["cap"] = round(cap, 1)
    return r, base, tasks


def _history_tasks(at=None):
    import baseline as B
    tasks = B.tasks_from_records(list(B.load_cache().values()), at or time.time())
    return [t for t in tasks if at is None or t["t0"] + t["wall"] < at]


def parse_at(v):
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        ts = parse_ts(v)
        if ts is None:
            sys.exit(f"can't read --at {v!r}; use an ISO time or epoch seconds")
        return ts


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


def forecast_now(args, now=None):
    """Everything `now` knows, as one dict (also what gets logged). Never refuses."""
    import waits as W
    at = parse_at(getattr(args, "at", None))
    now = at or now or time.time()
    path, kind, how = find_transcript(args.transcript, args.session, args.agent, now=now)
    if path and not os.path.exists(path):
        path, kind, how = None, None, f"transcript not found: {path}"
    rec = dict(id=uuid.uuid4().hex[:12], v=2, created=now, transcript=path, kind=kind, found_by=how,
               cwd=os.getcwd(), note=args.note, steps=args.steps, flags=[])
    if args.agent_min is not None:
        rec["agent_s"] = args.agent_min * 60          # private scoring baseline, never printed
    if path:
        st = session_state(path, kind, now)
    else:
        started = parse_started(args.started, now)
        st = {"elapsed_s": (now - started) if started else 0.0, "harness": args.agent, "calls": None,
              "task": None, "started": started, "ask_ts": None, "turn_ended_at": None,
              "last_ts": None, "model": None}
        rec["flags"].append("no transcript readable: elapsed time is self-reported" if started else
                            "no transcript readable and no --started: answering from history alone")
    task = st.get("task")
    rec.update(task_t0=st.get("started"), elapsed_s=round(st["elapsed_s"], 1), calls=st.get("calls"),
               model=st.get("model"))
    prior_state = {k: v for k, v in st.items() if k not in ("task",)}
    prior, base, hist_tasks = _prior(prior_state, at)
    st["prior"] = prior
    rec["prior"] = {k: prior[k] for k in ("p10", "p50", "p80", "p90", "cap", "n")}

    # waits: what the transcript shows is open, plus measured progress and CI the agent names
    history = W.load_history(now=now, budget_s=0.0 if at else 2.0)
    wlist = []
    if path:
        try:
            wlist = W.open_waits(path, kind, now, history, remember=not at)
        except (OSError, ValueError, KeyError, TypeError) as error:
            rec["flags"].append(f"could not read open waits ({type(error).__name__})")
    for pat in args.ignore_wait or []:
        wlist = [w for w in wlist if pat.lower() not in (w.get("label") or "").lower()]
    if args.progress:
        pw = progress_wait(*args.progress, now)
        if pw:
            wlist.insert(0, pw)
    if args.ci is not None:
        cw, why = ci_wait(args.ci, now)
        if cw:
            wlist.insert(0, cw)
        elif why:
            rec["flags"].append(why)
    live = [w for w in wlist if w["kind"] != "stale"]
    fc = W.forecast(path or "", kind or "claude", now, history, waits_=wlist) if wlist else \
        {"waits": [], "wait": None, "tail": None, "total": None}
    rec["waits"] = [{k: w.get(k) for k in ("kind", "label", "model", "elapsed_s", "p50_s", "low_s",
                                           "high_s", "n", "basis")} for w in wlist]
    rec["wait"], rec["tail"] = fc.get("wait"), fc.get("tail")

    # what state is the task in?
    ended_before_ask = st.get("turn_ended_at") is not None
    if args.blocked:
        rec.update(outcome="blocked", est_s=None, low_s=None, high_s=None, method="blocked on the user")
    elif ended_before_ask and not live and not args.steps:
        took = (st["turn_ended_at"] - st["started"]) if st.get("started") else None
        rec.update(outcome="done", est_s=0.0, low_s=0.0, high_s=0.0, method="already done",
                   finished=st["turn_ended_at"], took_s=took)
    else:
        rec["outcome"] = "estimate"
        pace, psrc = None, None
        if args.steps is not None:
            if task:
                pace = own_pace(task, st["elapsed_s"])[0]
                psrc = "measured on this task"
            if pace is None:
                pace, psrc = history_pace(hist_tasks or _history_tasks(at), st.get("harness"), st.get("model"))
        rec.update(pace_s=round(pace, 1) if pace else None, pace_source=psrc)
        low, mid, high, parts = combine(st, fc if live else None, args.steps, pace)
        rec.update(est_s=round(mid, 1), low_s=round(low, 1), high_s=round(high, 1),
                   method=parts["method"], parts=parts)
        rec["flags"] += parts["flags"]
        # Alternatives logged for scoring each source on its own later.
        nl, nm, nh, _ = combine(st, fc if live else None, None, None)
        rec["alt"] = {"history": [prior["p10"], prior["p50"], prior["p90"]],
                      "history+waits": [round(nl, 1), round(nm, 1), round(nh, 1)]}
        if args.steps is not None:
            own = args.steps * pace
            rec["alt"]["steps_only"] = [round(own * STEP_SPREAD[0], 1), round(own, 1),
                                        round(own * STEP_SPREAD[1], 1)]
    return rec, st, how


def confidence(rec):
    if rec.get("est_s") is None or not rec.get("low_s"):
        return "high" if rec.get("outcome") == "done" else "n/a"
    ratio = rec["high_s"] / max(rec["low_s"], 1.0)
    level = "high" if ratio <= 3 else "medium" if ratio <= 10 else "low"
    if level == "high" and (rec.get("prior") or {}).get("n", 99) < 10 and rec.get("pace_source") != "measured on this task":
        level = "medium"
    return level


def render(rec, st, how, script="~/.eta/eta.py"):
    now = rec["created"]
    L = []
    so_far = (f"{fmt_dur(st['elapsed_s'])} on this task" if st.get("started") else "start time unknown")
    if st.get("calls") is not None:
        so_far += f", {st['calls']} tool calls"
    if rec["outcome"] == "done":
        took = f" after {fmt_dur(rec['took_s'])}" if rec.get("took_s") is not None else ""
        L.append(f"ETA: already done. The task finished at {fmt_clock(rec['finished'], now)}{took}, "
                 f"and nothing it started is still running.")
        L.append("Reply with that and what was delivered. No need to keep working.")
        return "\n".join(L)
    if rec["outcome"] == "blocked":
        L.append(f"ETA: waiting on you. Needed: {one_line(rec['blocked'] if rec.get('blocked') else '', 160)}")
        L.append(f"Done so far: {so_far}.")
        L.append("Reply with exactly what you need from the user.")
        return "\n".join(L)
    est, low, high = rec["est_s"], rec["low_s"], rec["high_s"]
    L.append(f"ETA: about {fmt_dur(est)}, done around {fmt_clock(now + est, now)} "
             f"(likely {fmt_dur(low)} to {fmt_dur(high)}, {fmt_clock(now + low, now)} to {fmt_clock(now + high, now)})")
    L.append(f"Done so far: {so_far}.")
    left = []
    if rec.get("note"):
        left.append(one_line(rec["note"], 120))
    if rec.get("steps") is not None:
        left.append(f"{rec['steps']} more tool calls of own work at about {rec['pace_s']:.0f}s each "
                    f"({rec['pace_source']})")
    for w in rec.get("waits") or []:
        if w["kind"] == "stale":
            continue
        what = w["label"] + (f" ({w['model']})" if w.get("model") else "")
        left.append(f"waiting on {w['kind']} {what}: running {fmt_dur(w['elapsed_s'] or 0)}, "
                    f"about {fmt_dur(w['p50_s'])} left")
    if not left:
        left.append("not stated")
    L.append("Left: " + "; ".join(left) + ".")
    parts = rec.get("parts") or {}
    pr = parts.get("prior") or rec["prior"]
    basis = []
    if parts.get("own"):
        basis.append(f"own work {fmt_dur(parts['own']['mid_s'])} from the step count")
    if parts.get("wait"):
        basis.append(f"open waits about {fmt_dur(parts['wait']['mid_s'])} from how similar ones ran before")
    if parts.get("tail"):
        basis.append(f"about {fmt_dur(parts['tail']['mid_s'])} of wrap-up after them, typical here")
    n = rec["prior"].get("n")
    basis.append(f"{n} past tasks here that ran this long took about {fmt_dur(pr['mid_s'] if 'mid_s' in pr else pr['p50'])} more"
                 if n else "almost no past tasks ran this long, so the history range is wide")
    if parts.get("component"):
        basis.append(f"blended {round(100 * parts['component']['weight'])}% measured parts, "
                     f"{100 - round(100 * parts['component']['weight'])}% history")
    L.append("Basis: " + "; ".join(basis) + ".")
    conf = confidence(rec)
    thin = " (thin basis)" if (n or 0) < 10 and rec.get("pace_source") != "measured on this task" else ""
    L.append(f"Confidence: {conf}{thin}.")
    for f in rec.get("flags") or []:
        L.append(f"Note: {f}.")
    if how and "GUESS" in how:
        L.append(f"Note: transcript found by {how}.")
    return "\n".join(L)


def cmd_now(args):
    for v in (args.steps, args.agent_min):
        if v is not None and (not math.isfinite(v) or v < 0):
            sys.exit("--steps and --agent-min must be finite and not negative")
    if args.progress and (not all(math.isfinite(x) and x >= 0 for x in args.progress)
                          or args.progress[0] > args.progress[1] or args.progress[1] <= 0):
        sys.exit("--progress needs DONE TOTAL SECONDS with 0 <= DONE <= TOTAL")
    if args.steps is not None:
        args.steps = int(round(args.steps))
    rec, st, how = forecast_now(args)
    if args.blocked:
        rec["blocked"] = args.blocked
    record = not args.no_record and args.at is None
    if record:
        try:
            resolve_pending(rec["created"])
        except Exception:    # scoring old asks must never stop an answer
            pass
        append_jsonl(pred_path(), rec)
    if args.json:
        out = {k: v for k, v in rec.items() if k != "agent_s"}
        out["recorded"] = record
        out["text"] = render(rec, st, how)
        print(json.dumps(out, indent=2, default=str))
        return
    print(render(rec, st, how))
    if rec["outcome"] == "estimate":
        print("")
        print("NEXT: send these lines to the user now, in the same message as your next tool call, "
              "and keep working. /eta is a check-in, not a stop.")
    if record and rec["outcome"] == "estimate":
        print(f"At verified completion run: python3 ~/.eta/eta.py done --id {rec['id']}")


def cmd_estimate(args):
    """Old interface kept so an older SKILL.md still gets an answer: maps onto `now`."""
    if args.steps is None and args.calls is not None:
        args.steps = args.calls
    if args.progress is None and args.batch:
        args.progress = args.batch
    if args.ci is None and args.ci_job:
        args.ci = args.ci_job
    elif args.ci is None and args.checkout_ci is not None:
        args.ci = args.checkout_ci
    if args.note is None and args.reference:
        args.note = args.reference
    return cmd_now(args)


# ---------------------------------------------------------------- scoring

@contextmanager
def finish_lock():
    """Serialize simultaneous close notifications. The OS releases locks on exit."""
    os.makedirs(home(), exist_ok=True)
    with open(os.path.join(home(), "done.lock"), "a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _scoreable(p):
    return p.get("est_s") is not None and p.get("outcome", "estimate") == "estimate"


def resolve_pending(now=None):
    """Score open predictions from their transcripts: the task's end is the agent's last action
    before the next real prompt (baseline.py's task rules). Explicit `done` always wins."""
    import baseline as B
    now = now or time.time()
    done = {r["id"] for r in load_jsonl(res_path())}
    by_file = {}
    for p in load_jsonl(pred_path()):
        if p.get("id") in done or not p.get("transcript") or not _scoreable(p) or now - p["created"] < 60:
            continue
        if p.get("agent_s") is not None and p.get("v") is None:
            continue          # old paired forecasts only close by explicit `done`
        by_file.setdefault((p["transcript"], p.get("kind") or sniff_kind(p["transcript"])), []).append(p)
    for (path, kind), preds in by_file.items():
        if not os.path.exists(path):
            for p in preds:
                if now - p["created"] > 2 * 86400:
                    append_jsonl(res_path(), dict(id=p["id"], resolved=now, status="lost", by="transcript"))
            continue
        try:
            tasks = B.mine_file(path, kind, now)["tasks"]
        except Exception:     # one unreadable transcript must not stop scoring the rest
            continue
        for p in preds:
            t0 = p.get("task_t0")
            match = [t for t in tasks if (abs(t["t0"] - t0) < 1e-3 if t0 else t["t0"] <= p["created"])]
            if not match:
                continue
            t = match[-1] if not t0 else match[0]
            if t["end"] == "live":
                continue
            status = "ok" if t["end"] == "done" else t["end"]
            finish = t["t0"] + t["wall"]
            actual = max(0.0, finish - p["created"])
            if status == "ok" and actual > 12 * 3600:
                status = "abandoned"
            append_jsonl(res_path(), dict(id=p["id"], resolved=now, status=status, by="transcript",
                                         finish=finish, actual_s=actual))


def resolutions():
    """Latest resolution per prediction; an explicit `done` beats a transcript reading."""
    out = {}
    for r in load_jsonl(res_path()):
        old = out.get(r.get("id"))
        if old is None or r.get("by") == "done" or old.get("by") != "done":
            out[r.get("id")] = r
    return out


def cmd_done(args):
    """The task is verified finished now: close its open predictions with the exact time."""
    now = time.time()
    preds = {p["id"]: p for p in load_jsonl(pred_path()) if "id" in p}
    if args.id:
        unknown = set(args.id) - preds.keys()
        if unknown:
            sys.exit("unknown prediction id: " + ", ".join(sorted(unknown)))
        targets = list(dict.fromkeys(args.id))
    else:
        path, _, _ = find_transcript(args.transcript, args.session, args.agent, now=now)
        cwd = os.getcwd()
        targets = [p["id"] for p in preds.values() if _scoreable(p) and now - p["created"] < 12 * 3600
                   and ((path and p.get("transcript") == path)
                        or (not path and not p.get("transcript") and p.get("cwd") == cwd))]
    with finish_lock():
        closed = {r["id"] for r in load_jsonl(res_path()) if r.get("by") == "done"}
        n = 0
        for pid in targets:
            if pid in closed:
                print(f"{pid} already closed")
                continue
            p = preds[pid]
            actual = now - p["created"]
            append_jsonl(res_path(), dict(id=pid, resolved=now, finish=now, actual_s=actual, status="ok",
                                         by="done", ratio=actual / p["raw_s"] if p.get("raw_s") else None))
            n += 1
            said = f"said {fmt_dur(p['est_s'])}, " if p.get("est_s") is not None else ""
            print(f"{pid} marked done: {said}took {fmt_dur(actual)}")
    if not targets:
        print("nothing to close (no open ETA for this session in the last 12 h).")


def _miss(pred, actual):
    return abs(math.log(max(pred, 30.0) / max(actual, 30.0)))


def _summary(rows, label):
    """rows: [(low|None, mid|None, high|None, actual)] -> one line of metrics."""
    n = len(rows)
    ans = [r for r in rows if r[1] is not None]
    if not n:
        return f"{label}: nothing scored yet"
    txt = f"{label}: answered {len(ans)}/{n} ({100 * len(ans) / n:.0f}%)"
    if ans:
        misses = [_miss(r[1], r[3]) for r in ans]
        w2 = sum(m <= math.log(2) for m in misses)
        txt += (f"; within 2x {w2}/{len(ans)} ({100 * w2 / len(ans):.0f}%); median miss "
                f"{math.exp(statistics.median(misses)):.1f}x")
        ranged = [r for r in ans if r[0] is not None and r[2] is not None]
        if ranged:
            hit = sum(r[0] <= r[3] <= r[2] for r in ranged)
            txt += f"; inside the range {hit}/{len(ranged)} ({100 * hit / len(ranged):.0f}%)"
    return txt


def cmd_stats(args):
    try:
        resolve_pending()
    except Exception as error:   # stats still prints what is already scored
        print(f"(could not score open asks: {type(error).__name__})")
    preds = load_jsonl(pred_path())
    res = resolutions()
    print(f"ETA TRACK RECORD  ({len(preds)} asks logged in {home()})")
    answerable = [p for p in preds if p.get("outcome") in (None, "estimate", "done")]
    answered = [p for p in answerable if p.get("est_s") is not None]
    print(f"answer rate: {len(answered)}/{len(answerable)} asks got a number"
          + (f" ({100 * len(answered) / len(answerable):.0f}%)" if answerable else ""))
    floor = (args.min_remaining_min or 0) * 60
    scored = []
    for p in preds:
        r = res.get(p.get("id"))
        if not r or r.get("status") != "ok" or r.get("actual_s") is None or p.get("outcome") in ("done", "blocked"):
            continue
        if r["actual_s"] < floor:
            continue
        scored.append((p, r["actual_s"]))
    if floor:
        print(f"only asks where at least {args.min_remaining_min:g} min actually remained")
    print(f"scored (finish known): {len(scored)}")
    print("  " + _summary([(p.get("low_s"), p.get("est_s"), p.get("high_s"), a) for p, a in scored], "skill"))
    guess = [(None, p["agent_s"], None, a) for p, a in scored if p.get("agent_s") is not None]
    if guess:
        print("  " + _summary(guess, "agent's private guess"))
        both = [(p, a) for p, a in scored if p.get("agent_s") is not None and p.get("est_s") is not None]
        if both:
            closer = sum(_miss(p["est_s"], a) < _miss(p["agent_s"], a) for p, a in both)
            print(f"  skill closer than the agent's guess on {closer}/{len(both)}")
    for key in ("history", "history+waits", "steps_only"):
        rows = [(*p["alt"][key], a) for p, a in scored if (p.get("alt") or {}).get(key)]
        if rows:
            print("  " + _summary(rows, f"part: {key}"))
    open_n = sum(1 for p in preds if p.get("id") not in res and _scoreable(p))
    if open_n:
        print(f"waiting to be scored: {open_n}")
    print()
    print(f"{'when':<16} {'said':>8} {'range':>17} {'actual':>8}  hit  note")
    for p in preds[-args.limit:]:
        r = res.get(p.get("id"))
        when = datetime.fromtimestamp(p["created"]).strftime("%m-%d %I:%M%p").lower()
        actual = fmt_dur(r["actual_s"]) if r and r.get("status") == "ok" and r.get("actual_s") is not None \
            else (r.get("status") if r else "open")
        note = one_line(p.get("note") or p.get("task") or "", 48)
        if p.get("outcome") in ("done", "blocked"):
            print(f"{when:<16} {p['outcome']:>8} {'':>17} {'':>8}       {note}")
            continue
        if p.get("est_s") is None:
            print(f"{when:<16} {'no ETA':>8} {'':>17} {actual:>8}       {note}")
            continue
        hit = ""
        if r and r.get("status") == "ok" and r.get("actual_s") is not None and p.get("low_s") is not None:
            hit = "yes" if p["low_s"] <= r["actual_s"] <= p["high_s"] else "no"
        rng = f"{fmt_dur(p['low_s'])}-{fmt_dur(p['high_s'])}" if p.get("low_s") is not None else ""
        print(f"{when:<16} {fmt_dur(p['est_s']):>8} {rng:>17} {actual:>8}  {hit:<4} {note}")


# ---------------------------------------------------------------- CLI

def _now_args(p, legacy=False):
    p.add_argument("--steps", type=float, help="tool calls of your OWN work still to do (reads, edits, runs, "
                   "fix-and-rerun cycles, checks, commits); 0 if nothing is left but waiting")
    p.add_argument("--note", help="what is left, a few plain words")
    p.add_argument("--session", help="session id (Claude Code: ${CLAUDE_SESSION_ID})")
    p.add_argument("--blocked", metavar="WHAT", help="the task cannot go on until the user answers this")
    p.add_argument("--progress", type=float, nargs=3, metavar=("DONE", "TOTAL", "SECONDS"),
                   help="measured progress of a job the transcript cannot show: units done, total, seconds so far")
    p.add_argument("--ci", nargs="?", const="", metavar="JOB_URL",
                   help="the task waits on GitHub Actions: this checkout's running workflow, or a job URL")
    p.add_argument("--ignore-wait", action="append", metavar="TEXT",
                   help="an open wait (matched by label) that the deliverable does not need")
    p.add_argument("--started", help="no transcript: when the task began (14:05, 2:05pm, 25m, ISO)")
    p.add_argument("--agent-min", type=float, help=argparse.SUPPRESS)   # private scoring baseline
    p.add_argument("--transcript", help="explicit transcript path")
    p.add_argument("--agent", choices=["claude", "codex", "pi"], help="limit discovery to one harness")
    p.add_argument("--at", help="answer as of this past moment (ISO or epoch); not logged")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-record", action="store_true", help="don't log this prediction")
    if legacy:
        S = argparse.SUPPRESS
        p.add_argument("--calls", type=float, help=S)
        p.add_argument("--batch", type=float, nargs=3, help=S)
        p.add_argument("--ci-job", help=S)
        p.add_argument("--checkout-ci", nargs="?", const="", help=S)
        p.add_argument("--reference", help=S)
        for flag in ("--model", "--task-id", "--event", "--command", "--remaining-command",
                     "--minutes", "--wait-min", "--tail-min", "--elapsed-s", "--from", "--calls-done"):
            p.add_argument(flag, action="append", help=S)
        p.add_argument("--measured", action="store_true", help=S)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eta", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("now", help="the ETA right now, reply-ready (always answers)")
    _now_args(p)
    p.set_defaults(fn=cmd_now)
    p = sub.add_parser("estimate", help=argparse.SUPPRESS)
    _now_args(p, legacy=True)
    p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("done", help="the task is verified finished: close its open ETAs")
    p.add_argument("--id", nargs="+", help="prediction ids printed by `now`")
    p.add_argument("--session")
    p.add_argument("--transcript")
    p.add_argument("--agent", choices=["claude", "codex", "pi"])
    p.set_defaults(fn=cmd_done)
    p = sub.add_parser("stats", help="past ETAs vs actual finish times")
    p.add_argument("--min-remaining-min", type=float, help="only asks where at least this much remained")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(fn=cmd_stats)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    # Applies to new ledgers, locks and caches, including under ETA_HOME.
    os.umask(0o077)
    secure_history(home(), private_dir=os.path.abspath(home()) == os.path.expanduser("~/.eta"))
    main()
