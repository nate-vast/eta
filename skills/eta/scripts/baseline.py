#!/usr/bin/env python3
"""History prior for /eta: how much longer do tasks like this one usually run?

Mines every local agent transcript (Claude Code, Codex, pi) into "tasks": one per
real human prompt, ending at the agent's last action before the next real human
prompt. Caches a compact, text-free timeline per task in ~/.eta/turns.jsonl and
answers "given the task has run this long and looks like this right now, how much
longer will it take?" from tasks that survived to the same point.

    python3 baseline.py build                       refresh the cache, print counts
    python3 baseline.py query --elapsed-min 20      ask the prior

Integrators: call remaining_given(state). See its docstring for the state shape.

Task rules (decided from what is on disk, see verify/REPLAY.md for numbers):
- Prompts that do not start a task: /eta and other ETA asks, background or
  delegate notifications (including T3's "Delegated task <id> ... completed" notice to the
  parent), T3 inter-agent messages, compaction and builtin slash commands (/compact,
  /model, ...), "[Request interrupted ...]" markers. Claude Code background completions
  that exist only as queue-operation / queued_command records count as notifications.
- Scheduled-task prompts and T3 side chats do start a task but are tagged by origin;
  the default prior is fitted on human-origin tasks only.
- An ETA ask that arrives after the agent already ended its turn closes the task; the
  agent's answer to it is not task time. (On this machine most /eta asks reach the
  transcript only after the turn ended, see REPLAY.md.)
- A notification that wakes the agent continues the current task when it arrives
  within RESUME_S of the agent's last action, or while a background job or
  delegate it launched is still outstanding. Otherwise the task ended at the
  agent's earlier stop and the late wake-up work is dropped.
- end kinds: done (the agent ended its turn), interrupted (user interrupt or
  Codex turn_aborted), steered (the next human prompt arrived mid-turn), cut (the
  transcript just stops mid-turn), live (the newest task of a file written in the
  last STALE_S). Only "done" tasks with at least one tool call are fitted.
  Interrupted, steered and cut tasks are censored and excluded; this biases the
  prior short, see REPLAY.md.
- Zero-tool turns (pure chat) are kept in the cache but never fitted.
- Claude sub-agent files (<project>/<session>/subagents/*.jsonl) are not tasks:
  their time is already inside the parent's Agent call. Codex subagent sessions
  (session_meta source.subagent, including the auto-review "guardian") are skipped
  for the same reason. Forked or resumed Claude sessions copy earlier records into
  a new file; tasks are de-duplicated by harness plus prompt timestamp.
- Waits on the user (AskUserQuestion, user-approved permission prompts) stay in
  wall time, since that is what the asker experiences, and are recorded as uw.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import eta as _eta  # noqa: E402  (reuse prompt cleaning, ETA detection, timestamps, pi paths)

SCHEMA = 8
RESUME_S = 30 * 60          # a wake-up later than this, with nothing outstanding, starts no work
STALE_S = 20 * 60           # newest task of a file touched this recently may still be running
GAP_CAP = _eta.GAP_CAP      # active time counts each quiet gap as at most this
MAX_TASK_S = 24 * 3600      # longer "tasks" are left in the cache but not fitted
K_SHRINK = 25               # pseudo-count when blending a narrow pool into its parent
LINDY_WIDEN = 1.5           # log-spread multiplier for the remaining-vs-elapsed fallback
FIT_ORIGINS = ("human",)
DEFAULT_FEATURES = ()   # replay found no feature that beats elapsed-only survival, see REPLAY.md
ALL_FEATURES = ("harness", "family", "running", "deleg", "calls", "origin")

GRID = [0, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800,
        2400, 3600, 5400, 7200, 10800, 14400, 21600, 28800, 43200, 86400]

BUILTIN_COMMANDS = {
    "compact", "model", "clear", "context", "cost", "config", "help", "login", "logout",
    "status", "resume", "effort", "fast", "permissions", "mcp", "memory", "exit", "quit",
    "doctor", "hooks", "agents", "theme", "vim", "terminal-setup", "usage", "add-dir",
    "export", "rewind", "statusline", "output-style", "privacy-settings", "upgrade",
    "release-notes", "bashes", "ide", "todos", "plugin", "keybindings", "rename",
}
NOTIFY_PREFIXES = ("<task-notification>", "[T3 Code: the agent in", "Background command completed",
                   "Background command failed", "[T3 Code: background")

# tool classes stored in the timeline: d delegate, w wait-on-something, a ask user, s shell, o other
_DELEGATE = re.compile(r"^(Agent|Task)$|delegate_task|spawn_agent|followup_task|t3_thread_launch")
_WAIT = re.compile(r"background_wait|^wait$|^wait_agent$|^Monitor$|thread_wait|^TaskOutput$|"
                   r"^BashOutput$|t3_thread_wait|wait_for")
_ASK = re.compile(r"^AskUserQuestion$|request_user_input|^ExitPlanMode$")
# launched work that reports back later through a notification
_ASYNC = re.compile(r"delegate_task|spawn_agent|followup_task|t3_thread_launch|t3_thread_send|"
                    r"schedule_task")
_SHELL = re.compile(r"^(Bash|exec|exec_command|shell|local_shell|bash)$|_exec$")


def tool_class(name: str) -> str:
    name = name or ""
    if _DELEGATE.search(name):
        return "d"
    if _WAIT.search(name):
        return "w"
    if _ASK.search(name):
        return "a"
    if _SHELL.search(name):
        return "s"
    return "o"


def model_family(model) -> str | None:
    m = (model or "").lower()
    if not m or m.startswith("<"):
        return None
    for fam in ("opus", "sonnet", "haiku", "fable"):
        if fam in m:
            return fam
    hit = re.match(r"(gpt-[\d.]+|o\d|gemini-[\d.]+|qwen[\d.]*|llama[\d.]*|deepseek)", m)
    if hit:
        return hit.group(1)
    return m.split("/")[-1][:24]


def classify_prompt(text: str, notify: bool = False) -> str | None:
    """human | eta | notify | scheduled | side | delegated | meta | interrupt | None (empty)."""
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("[Request interrupted"):
        return "interrupt"
    if notify or t.startswith(NOTIFY_PREFIXES):
        return "notify"
    if t.startswith("[T3 Code: scheduled task"):
        return "scheduled"
    if t.startswith("[T3 Code:"):
        return "notify"
    if t.startswith("You are in a side chat"):
        return "side"
    if t.startswith("Delegated task"):
        # T3 posts this to the PARENT when a delegate finishes ("Delegated task <id> (title)
        # completed ..."); it continues the parent's task. A child's own task arrives as a
        # "[T3 Code: the agent in ...]" message instead.
        return "notify"
    if t.startswith(_eta.SYSTEM_PREFIXES):
        return "meta"
    m = re.match(r"^/([\w:-]+)\s*$", t)
    if m and m.group(1).lower() in BUILTIN_COMMANDS:
        return "meta"
    if _eta.is_eta_prompt(t):
        return "eta"
    return "human"


# ---------------------------------------------------------------- paths and private files

def eta_home() -> str:
    return _eta.home()


def cache_path() -> str:
    return os.path.join(eta_home(), "turns.jsonl")


def _private_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o700, exist_ok=True)


def _write_private(path, lines):
    """Atomically replace `path` with `lines`; the file is created 0600."""
    _private_dir(os.path.dirname(path))
    tmp = f"{path}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            for line in lines:
                f.write(line + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def source_files():
    """(path, harness) for every transcript we know how to read. Sub-agent files excluded."""
    out = [(p, "claude") for p in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))]
    out += [(p, "codex") for p in glob.glob(os.path.expanduser(
        "~/.codex/sessions/[0-9][0-9][0-9][0-9]/*/*/rollout-*.jsonl"))]
    out += [(p, "pi") for p in glob.glob(os.path.join(_eta.pi_sessions_dir(), "*", "*.jsonl"))]
    return out


# ---------------------------------------------------------------- parsers
# Normalized events, in file order:
#   ("P", ts, cls)                  prompt, cls from classify_prompt
#   ("A", ts, ended, model)         agent output; ended=True when it closed the turn
#   ("T", ts, id, tool_cls, bg)     tool call started (bg: launched in the background)
#   ("D", ts, id, user_wait)        tool result (user_wait: a person approved it)
#   ("X", ts)                       interrupt / aborted turn

_CLAUDE_REC = re.compile(rb'"type":\s?"(?:user|assistant)"')
_TASK_ID = re.compile(r"<task-id>([^<]*)</task-id>")


def _queued_claude(d):
    """Text of a prompt Claude Code queued mid-turn instead of writing a user record:
    a queue-operation enqueue (background completions) or a queued_command attachment
    (completions, and prompts typed while the agent was busy). None for anything else."""
    if d.get("type") == "queue-operation":
        return d.get("content") if d.get("operation") == "enqueue" and isinstance(d.get("content"), str) else None
    att = d.get("attachment") if isinstance(d.get("attachment"), dict) else {}
    if att.get("type") != "queued_command":
        return None
    p = att.get("prompt")
    return _eta._clean_prompt(p) if isinstance(p, (str, list)) else None


def parse_claude(path):
    events, meta = [], {"cwd": None, "skip": None}
    notified = set()   # background task ids already closed, so a queued copy is not counted twice
    with open(path, "rb") as f:
        for raw in f:
            if not _CLAUDE_REC.search(raw[:200]) and (b'"queue-operation"' in raw[:80] or b'"queued_command"' in raw):
                # Background completions often exist only as these records; mid-turn prompts too.
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                text = _queued_claude(d) if isinstance(d, dict) else None
                ts = _eta.parse_ts(d.get("timestamp")) if text else None
                if ts is None:
                    continue
                if "<task-notification>" in text:
                    ids = set(_TASK_ID.findall(text))
                    if ids and ids <= notified:
                        continue
                    notified |= ids
                    events.append(("P", ts, "notify"))
                elif d.get("type") != "queue-operation" and classify_prompt(text) in ("eta", "notify"):
                    # A prompt injected mid-turn steers the current task; only ETA asks and
                    # notices matter for timing.
                    events.append(("P", ts, classify_prompt(text)))
                continue
            if not _CLAUDE_REC.search(raw):
                continue
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            typ = d.get("type")
            if typ not in ("user", "assistant") or d.get("isSidechain"):
                continue
            ts = _eta.parse_ts(d.get("timestamp"))
            if ts is None:
                continue
            if meta["cwd"] is None and d.get("cwd"):
                meta["cwd"] = d["cwd"]
            msg = d.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if typ == "assistant":
                model = msg.get("model") if isinstance(msg, dict) else None
                stop = msg.get("stop_reason") if isinstance(msg, dict) else None
                events.append(("A", ts, stop in ("end_turn", "stop_sequence"), model))
                for b in content if isinstance(content, list) else []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                        events.append(("T", ts, b.get("id"), tool_class(b.get("name")),
                                       bool(inp.get("run_in_background"))
                                       or bool(_ASYNC.search(b.get("name") or ""))))
                continue
            if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                pd = d.get("permissionDecision")
                uw = isinstance(pd, dict) and str(pd.get("source", "")).startswith("user")
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_result":
                        events.append(("D", ts, b.get("tool_use_id"), uw))
                    elif b.get("type") == "text" and str(b.get("text", "")).startswith("[Request interrupted"):
                        events.append(("X", ts))
                continue
            if d.get("isMeta") or d.get("isCompactSummary"):
                continue
            origin = d.get("origin") if isinstance(d.get("origin"), dict) else {}
            notify = d.get("turnOrigin") in ("task_notification", "system") or \
                origin.get("kind") == "task-notification"
            text = _eta._clean_prompt(content)
            if "<task-notification>" in text[:200]:
                ids = set(_TASK_ID.findall(text))
                if ids and ids <= notified:
                    continue
                notified |= ids
            cls = classify_prompt(text, notify)
            if cls == "interrupt":
                events.append(("X", ts))
            elif cls:
                events.append(("P", ts, cls))
    return events, meta


_CODEX_SKIP = re.compile(rb'"type":\s?"(?:token_count|token_usage_record|world_state)"')


def parse_codex(path):
    events, meta = [], {"cwd": None, "skip": None}
    model = None
    with open(path, "rb") as f:
        for i, raw in enumerate(f):
            if _CODEX_SKIP.search(raw, 0, 300):
                continue
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            typ, p = d.get("type"), d.get("payload") or {}
            if typ == "session_meta":
                src = p.get("source")
                if isinstance(src, dict) and "subagent" in src:
                    meta["skip"] = "codex subagent session"
                    return [], meta
                meta["cwd"] = meta["cwd"] or p.get("cwd")
                continue
            ts = _eta.parse_ts(d.get("timestamp"))
            if ts is None:
                continue
            if typ == "turn_context":
                model = p.get("model") or model
            elif typ == "response_item":
                pt = p.get("type")
                if pt in ("function_call", "custom_tool_call", "local_shell_call"):
                    events.append(("T", ts, p.get("call_id") or p.get("id"),
                                   tool_class(p.get("name") or ("exec" if pt == "local_shell_call" else "")),
                                   bool(_ASYNC.search(p.get("name") or ""))))
                elif pt in ("function_call_output", "custom_tool_call_output"):
                    events.append(("D", ts, p.get("call_id"), False))
            elif typ == "event_msg":
                pt = p.get("type")
                if pt == "item_completed":
                    it = p.get("item") or {}
                    k = it.get("type")
                    if k == "UserMessage":
                        text = _eta._clean_prompt([{"type": "text", "text": b.get("text", "")}
                                                   for b in it.get("content") or [] if isinstance(b, dict)])
                        cls = classify_prompt(text)
                        if cls == "interrupt":
                            events.append(("X", ts))
                        elif cls:
                            events.append(("P", ts, cls))
                    elif k in ("AgentMessage", "Reasoning"):
                        events.append(("A", ts, False, model))
                elif pt == "task_complete":
                    events.append(("A", ts, True, model))
                elif pt == "turn_aborted":
                    events.append(("X", ts))
    return events, meta


def parse_pi(path):
    """pi is small: reuse eta.read_pi for the event stream, plus one pass for cwd/model/stop."""
    meta = {"cwd": None, "skip": None}
    stops = {}
    model = None
    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") == "session":
                meta["cwd"] = d.get("cwd")
            elif d.get("type") == "model_change":
                model = d.get("modelId") or d.get("model") or model
            elif d.get("type") == "message":
                msg = d.get("message") or {}
                if msg.get("role") == "assistant":
                    ts = _eta.parse_ts(d.get("timestamp"))
                    stops[ts] = (msg.get("stopReason"), msg.get("model") or model)
    events = []
    for e in _eta.read_pi(path):
        if e["kind"] == "prompt":
            cls = classify_prompt(e["text"])
            if cls == "interrupt":
                events.append(("X", e["ts"]))
            elif cls:
                events.append(("P", e["ts"], cls))
        elif e["kind"] == "agent":
            stop, m = stops.get(e["ts"], (None, model))
            if stop == "aborted":
                events.append(("X", e["ts"]))
            events.append(("A", e["ts"], stop in ("stop", "end_turn", "length"), m))
        elif e["kind"] == "tool":
            events.append(("T", e["ts"], e["id"], tool_class(e["name"]), False))
        elif e["kind"] == "tool_done":
            events.append(("D", e["ts"], e["id"], False))
    return events, meta


PARSERS = {"claude": parse_claude, "codex": parse_codex, "pi": parse_pi}


# ---------------------------------------------------------------- task splitting

class _Task:
    def __init__(self, ts, origin):
        self.t0, self.origin, self.last = ts, origin, ts
        self.calls, self.open, self.bg, self.acts = [], {}, [], []
        self.model = None
        self.ended = self.interrupted = self.frozen = False
        self.uw = 0.0
        self.eta = []

    def outstanding(self):
        return any(e is None for _, e in self.bg) or any(
            self.calls[i][2] in "dw" for i in self.open.values())

    def act(self, ts):
        self.last = max(self.last, ts)
        self.acts.append(ts)


def _finish(t, harness, how):
    rel = lambda x: round(x - t.t0, 1)  # noqa: E731
    calls = [[rel(s), rel(e) if e is not None else -1, c] for s, e, c in t.calls]
    open_fg = any(e is None for s, e, c in t.calls)
    if t.interrupted:
        kind = "interrupted"
    elif t.ended and not open_fg:
        kind = "done"
    elif how == "live":
        kind = "live"
    elif how == "next":
        kind = "steered"
    else:
        kind = "cut"
    acts = sorted(t.acts)
    active = 0.0
    for a, b in zip([t.t0] + acts, acts):
        active += min(b - a, GAP_CAP)
    return {
        "k": f"{harness}:{t.t0:.3f}", "h": harness, "m": t.model, "o": t.origin,
        "t0": round(t.t0, 3), "wall": rel(t.last), "act": round(active, 1), "uw": round(t.uw, 1),
        "end": kind, "n": len(t.calls), "eta": len(t.eta), "q": [rel(x) for x in t.eta], "drop": int(t.frozen),
        "c": calls, "b": [[rel(s), rel(e) if e is not None else -1] for s, e in t.bg],
        "a": [int(x - t.t0) for x in acts[-400:]] if len(acts) <= 400 else
             [int(x - t.t0) for x in acts[::max(1, len(acts) // 400)]],
    }


def split_tasks(events, harness, live=False):
    """Turn a normalized event stream into task dicts. `live`: the file is still being written."""
    events = sorted(events, key=lambda e: e[1])   # stable: keeps file order on equal times
    tasks, cur = [], None

    def close(how):
        if cur is not None and cur.acts:
            tasks.append(_finish(cur, harness, how))

    for ev in events:
        kind, ts = ev[0], ev[1]
        if kind == "P":
            cls = ev[2]
            if cls in ("human", "scheduled", "side", "delegated"):
                mid_turn = cur is not None and not cur.frozen and not cur.ended
                close("next" if mid_turn else "end")
                cur = _Task(ts, cls)
            elif cls == "eta" and cur is not None and not cur.frozen:
                cur.eta.append(ts)
                if cur.ended and not cur.outstanding():
                    cur.frozen = True          # asked after the work ended: the answer is not task time
            elif cls == "notify" and cur is not None and not cur.frozen:
                if cur.ended and ts - cur.last > RESUME_S and not cur.outstanding():
                    cur.frozen = True          # task already over; later wake-up work is dropped
                else:
                    for j in cur.bg:
                        if j[1] is None:
                            j[1] = ts
                            break
            continue
        if cur is None or cur.frozen:
            continue
        if kind == "X":
            cur.interrupted = True
            continue
        if kind == "A":
            cur.act(ts)
            cur.ended = bool(ev[2])
            if ev[3] and not str(ev[3]).startswith("<"):
                cur.model = cur.model or ev[3]
        elif kind == "T":
            cur.act(ts)
            cur.ended = False
            tid, cls, bg = ev[2], ev[3], ev[4]
            cur.calls.append([ts, None, cls])
            if tid is not None:
                cur.open[tid] = len(cur.calls) - 1
            if bg:
                cur.bg.append([ts, None])
        elif kind == "D":
            i = cur.open.pop(ev[2], None)
            if i is None:
                continue
            cur.act(ts)
            call = cur.calls[i]
            call[1] = ts
            if ev[3] or call[2] == "a":
                cur.uw += ts - call[0]
    close("live" if live else "end")
    return tasks


def mine_file(path, harness, now=None):
    now = now or time.time()
    try:
        events, meta = PARSERS[harness](path)
    except (OSError, UnicodeDecodeError) as exc:
        return {"skip": f"unreadable: {type(exc).__name__}", "tasks": []}
    if meta.get("skip"):
        return {"skip": meta["skip"], "tasks": []}
    live = now - os.path.getmtime(path) < STALE_S
    tasks = split_tasks(events, harness, live=live)
    sess = os.path.splitext(os.path.basename(path))[0]
    for t in tasks:
        t["cwd"], t["s"] = meta.get("cwd"), sess
    return {"skip": None, "tasks": tasks}


# ---------------------------------------------------------------- cache

def load_cache(path=None):
    path = path or cache_path()
    out = {}
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("v") == SCHEMA:
                    out[r["f"]] = r
    except OSError:
        pass
    return out


def build(files=None, path=None, now=None, verbose=False):
    """Refresh the cache incrementally (file path + mtime + size). Returns the records."""
    now = now or time.time()
    path = path or cache_path()
    old = load_cache(path)
    files = source_files() if files is None else files
    recs, parsed = [], 0
    for f, h in files:
        try:
            st = os.stat(f)
        except OSError:
            continue
        r = old.get(f)
        if r and r["mt"] == st.st_mtime and r["sz"] == st.st_size and r["h"] == h:
            recs.append(r)
            continue
        parsed += 1
        res = mine_file(f, h, now)
        recs.append({"v": SCHEMA, "f": f, "h": h, "mt": st.st_mtime, "sz": st.st_size, **res})
    _write_private(path, (json.dumps(r, separators=(",", ":")) for r in recs))
    if verbose:
        print(f"parsed {parsed} changed files, reused {len(recs) - parsed}", file=sys.stderr)
    return recs


def tasks_from_records(recs, now=None):
    """Flatten, de-duplicate forks/resumes, and expire stale 'live' tasks."""
    now = now or time.time()
    best = {}
    for r in recs:
        for t in r.get("tasks") or []:
            if t["end"] == "live" and now - r["mt"] > STALE_S:
                t = dict(t, end="cut")
            k = t["k"]
            if k not in best or (t["n"], t["wall"]) > (best[k]["n"], best[k]["wall"]):
                best[k] = dict(t, f=r.get("f"))   # f: the transcript the kept copy came from
    return sorted(best.values(), key=lambda t: t["t0"])


def fit_eligible(t, origins=FIT_ORIGINS) -> bool:
    return (t["end"] == "done" and t["n"] >= 1 and t["o"] in origins
            and 5 <= t["wall"] <= MAX_TASK_S)


# ---------------------------------------------------------------- state at time t

def calls_bucket(n):
    if n is None:
        return None
    for i, hi in enumerate((0, 3, 10, 30, 100)):
        if n <= hi:
            return i
    return 5


def running_bucket(running_s, running_cls=None):
    if running_s is None:
        return None
    if running_s <= 0:
        return "none"
    if running_cls in ("d", "w"):
        return "deleg"
    return "short" if running_s < 60 else "long"


def state_at(task, t):
    """Reconstruct what a live observer would see `t` seconds into `task`."""
    calls = running = deleg = 0
    running_s, running_cls = 0.0, None
    end = task["wall"]
    for s, e, c in task["c"]:
        if s > t:
            break
        calls += 1
        e = end + 1e9 if e == -1 else e
        if e > t:
            if c in "dw":
                deleg += 1
            if t - s > running_s:
                running_s, running_cls = t - s, c
            running += 1
    for s, e in task.get("b") or []:
        if s <= t and (e == -1 or e > t):
            deleg += 1
    last = max([a for a in task.get("a") or [] if a <= t] or [0])
    return {"elapsed_s": float(t), "harness": task["h"], "model": task.get("m"),
            "origin": task.get("o"), "calls": calls, "running_s": running_s,
            "running_cls": running_cls, "delegates_open": deleg, "quiet_s": t - last}


def features(state):
    """Cheap discrete features of a state. None means unknown and is skipped when matching."""
    d = state.get("delegates_open")
    return {
        "harness": state.get("harness"),
        "family": model_family(state.get("model")) if state.get("model") else state.get("family"),
        "running": running_bucket(state.get("running_s"), state.get("running_cls")),
        "deleg": None if d is None else int(d > 0),
        "calls": calls_bucket(state.get("calls")),
        "origin": state.get("origin") or "human",
    }


# ---------------------------------------------------------------- estimator

def _snap(e):
    return GRID[max(0, bisect.bisect_right(GRID, e) - 1)]


class Baseline:
    """Survival prior: remaining time among past tasks that were still running at this
    elapsed time, narrowed by matching features and shrunk toward the broader pool."""

    def __init__(self, tasks, feats=DEFAULT_FEATURES, k=K_SHRINK, origins=FIT_ORIGINS):
        self.feats, self.k = tuple(feats), k
        fit = [t for t in tasks if fit_eligible(t, origins)]
        self.n_fit = len(fit)
        self.table = {}
        for t in fit:
            dur = t["wall"]
            for g in GRID:
                if g >= dur:
                    break
                fv = features(state_at(t, g))
                key = ()
                self.table.setdefault((g, key), []).append(dur)
                for name in self.feats:
                    key = key + (fv[name],)
                    self.table.setdefault((g, key), []).append(dur)
        for v in self.table.values():
            v.sort()
        # Lindy fallback: remaining/elapsed ratios at the deepest grid point with a decent pool
        self.ratios, self.ratio_at = [], None
        for g in reversed(GRID[1:]):
            pool = self.table.get((g, ()), [])
            if len(pool) >= 30:
                raw = sorted((d - g) / g for d in pool)
                med = raw[len(raw) // 2]
                # widened by half again in log terms: this part only speaks when evidence is thin
                self.ratios = sorted(med * (r / med) ** LINDY_WIDEN for r in raw)
                self.ratio_at = g
                break
        if not self.ratios:   # no history at all: remaining between ~1/20 and ~20x elapsed
            self.ratios = [0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0]

    def _components(self, e, fv):
        g = _snap(e)
        comps, key = [], ()
        base = self.table.get((g, key), [])
        comps.append((base, bisect.bisect_right(base, e)))
        for name in self.feats:
            if fv.get(name) is None:
                break
            key = key + (fv[name],)
            pool = self.table.get((g, key), [])
            lo = bisect.bisect_right(pool, e)
            if len(pool) - lo == 0:
                break
            comps.append((pool, lo))
        return comps

    def _cdf_parts(self, e, fv):
        """[(weight, cdf_fn)] for the shrinkage mixture."""
        comps = self._components(e, fv)
        n_base = len(comps[0][0]) - comps[0][1]
        el = max(e, 30.0)
        ratios = self.ratios

        def lindy(x):
            return bisect.bisect_right(ratios, x / el) / len(ratios)

        def pool_cdf(pool, lo):
            n = len(pool) - lo
            return lambda x: (bisect.bisect_right(pool, e + x) - lo) / n

        w_base = n_base / (n_base + self.k) if n_base else 0.0
        parts = [(1 - w_base, lindy)]
        if n_base:
            parts.append((w_base, pool_cdf(*comps[0])))
        for pool, lo in comps[1:]:
            n = len(pool) - lo
            w = n / (n + self.k)
            parts = [(pw * (1 - w), f) for pw, f in parts] + [(w, pool_cdf(pool, lo))]
        return parts, comps

    def quantiles(self, state, qs=(0.1, 0.5, 0.8, 0.9)):
        e = float(state.get("elapsed_s") or 0.0)
        fv = features(state)
        parts, comps = self._cdf_parts(e, fv)

        def cdf(x):
            return sum(w * f(x) for w, f in parts)

        out = []
        hi_cap = max(10 * 86400, 50 * e)
        for q in qs:
            lo, hi = 0.0, hi_cap
            if cdf(hi) < q:
                out.append(hi)
                continue
            for _ in range(60):
                mid = (lo + hi) / 2
                if cdf(mid) >= q:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < 0.5:
                    break
            out.append(hi)
        used = [name for name, _ in zip(self.feats, comps[1:])]
        n = len(comps[-1][0]) - comps[-1][1]
        return out, n, used

    def remaining_given(self, state):
        (p10, p50, p80, p90), n, used = self.quantiles(state)
        e = float(state.get("elapsed_s") or 0.0)
        pool, lo = self._components(e, features(state))[0]
        n_base = len(pool) - lo
        basis = (f"{n} past tasks still running at {_eta.fmt_dur(e)}"
                 + (f" with the same {', '.join(used)}" if used else "")
                 + f" (of {self.n_fit} finished tasks on this machine)")
        if n < 10:
            basis += "; thin evidence, range widened toward the general pattern"
        r = lambda x: round(x, 1)  # noqa: E731
        return {"p10": r(p10), "p50": r(p50), "p80": r(p80), "p90": r(p90), "n": n,
                "n_survivors": n_base, "basis": basis}

    def estimate(self, state):
        """Replay hook: (low, mid, high) = (p10, p50, p90)."""
        (p10, p50, p90), _, _ = self.quantiles(state, (0.1, 0.5, 0.9))
        return p10, p50, p90


_CACHED = {}


def default_baseline(refresh_after=3600, now=None):
    """Baseline over this machine's history; refreshes the cache when older than refresh_after."""
    now = now or time.time()
    path = cache_path()
    try:
        age = now - os.path.getmtime(path)
    except OSError:
        age = None
    recs = None
    if age is not None and (refresh_after is None or age <= refresh_after):
        recs = list(load_cache(path).values())
    if not recs:   # missing, stale, or written by an older schema
        recs = build(now=now)
    key = (path, os.path.getmtime(path))
    if key not in _CACHED:
        _CACHED.clear()
        _CACHED[key] = Baseline(tasks_from_records(recs, now))
    return _CACHED[key]


def remaining_given(state, baseline=None):
    """Remaining wall seconds for the current task, from this machine's own history.

    state: {"elapsed_s": float (required, wall seconds since the task's human prompt),
            "harness": "claude"|"codex"|"pi"|None,
            "model": str|None,
            "running_s": float|None   (how long the oldest open tool call has run; 0 = none),
            "running_cls": "d"|"w"|"s"|"a"|"o"|None  (delegate, wait, shell, ask, other),
            "delegates_open": int|None, "calls": int|None}
    Missing keys are fine; only matched features narrow the pool.
    Returns {"p10","p50","p80","p90" (seconds remaining), "n", "n_survivors", "basis"}.
    Always answers: with little matching history the range widens toward the broader
    pool and finally toward a remaining-proportional-to-elapsed rule.
    """
    return (baseline or default_baseline()).remaining_given(state)


def state_from_transcript(path, harness=None, now=None):
    """State of the newest task in a live transcript, in the shape remaining_given expects."""
    now = now or time.time()
    harness = harness or _eta.sniff_kind(path)
    events, meta = PARSERS[harness](path)
    tasks = split_tasks(events, harness, live=True)
    if not tasks:
        return {"elapsed_s": 0.0, "harness": harness}
    t = tasks[-1]
    st = state_at(t, max(0.0, now - t["t0"]))
    st["started"] = t["t0"]
    return st


# ---------------------------------------------------------------- CLI

def _summary(recs, now):
    from collections import Counter
    files = Counter((r["h"], r.get("skip") or "ok") for r in recs)
    tasks = tasks_from_records(recs, now)
    raw = sum(len(r.get("tasks") or []) for r in recs)
    by = Counter((t["h"], t["o"], t["end"], "tools" if t["n"] else "chat") for t in tasks)
    print(f"cache: {cache_path()}")
    for (h, s), n in sorted(files.items()):
        print(f"  files {h:<6} {s:<28} {n}")
    print(f"tasks: {len(tasks)} after de-duplicating {raw - len(tasks)} copied by forks/resumes")
    for h in sorted({k[0] for k in by}):
        rows = {k[1:]: v for k, v in by.items() if k[0] == h}
        fit = sum(1 for t in tasks if t["h"] == h and fit_eligible(t))
        print(f"  {h}: {sum(rows.values())} tasks, {fit} fit-eligible (human, done, >=1 tool call)")
        for (o, e, c), n in sorted(rows.items(), key=lambda kv: -kv[1]):
            print(f"      {o:<10} {e:<12} {c:<5} {n}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="baseline", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="refresh ~/.eta/turns.jsonl and print counts")
    q = sub.add_parser("query", help="remaining time for a task at a given elapsed time")
    q.add_argument("--elapsed-min", type=float, required=True)
    q.add_argument("--harness", choices=["claude", "codex", "pi"])
    q.add_argument("--model")
    q.add_argument("--running-s", type=float, help="seconds the current tool call has been running (0 = none)")
    q.add_argument("--running-cls", choices=list("dwsao"))
    q.add_argument("--delegates-open", type=int)
    q.add_argument("--calls", type=int)
    q.add_argument("--transcript", help="read the state from a live transcript instead")
    q.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    now = time.time()
    if args.cmd == "build":
        t = time.time()
        recs = build(now=now, verbose=True)
        _summary(recs, now)
        print(f"built in {time.time() - t:.1f}s")
        return
    if args.transcript:
        state = state_from_transcript(args.transcript, now=now)
    else:
        state = {"elapsed_s": args.elapsed_min * 60, "harness": args.harness, "model": args.model,
                 "running_s": args.running_s, "running_cls": args.running_cls,
                 "delegates_open": args.delegates_open, "calls": args.calls}
    r = remaining_given(state)
    if args.json:
        print(json.dumps({"state": state, **r}, indent=2))
        return
    f = _eta.fmt_dur
    print(f"remaining: ~{f(r['p50'])} (80% range {f(r['p10'])} to {f(r['p90'])}; p80 {f(r['p80'])})")
    print(f"basis: {r['basis']}")


if __name__ == "__main__":
    os.umask(0o077)
    main()
