#!/usr/bin/env python3
"""What a live coding-agent session is waiting on, and how long until the deliverable is ready.

The parent's tool-call pace says nothing about time spent waiting on something
external. This module finds the open waits in a session (delegated agents,
running shell commands, batches and downloads with countable progress) and
estimates each from what the waited-on thing has done so far and what similar
things took before. It also measures how long parents historically kept working
after their last delegate returned (the "parent tail"), so a caller can add
integrate-and-verify time.

Public API (all durations in seconds):
    open_waits(transcript_path, kind=None, now=None, history=None) -> [wait, ...]
    combine(waits, mode="parallel") -> (low, p50, high)
    parent_tail(harness=None, n_delegates=None, wait_s=None, kind="delegate",
                now=None, history=None) -> {p50_s, low_s, high_s, n, basis}
    forecast(transcript_path, kind=None, now=None) -> waits + combined + tail + total
    load_history(now=None, budget_s=2.0) -> History

CLI:
    python3 waits.py show [--session ID | --transcript PATH] [--json]
    python3 waits.py mine            # build or refresh the history cache fully
    python3 waits.py summary         # history sizes and distributions
    python3 waits.py backtest        # replay real sessions cut during waits

History is mined from ~/.claude/projects, ~/.codex/sessions (and archived_sessions),
pi sessions and the T3 state database, read only. The cache lives in
$ETA_HOME/waits-cache.json (mode 0600) and holds only timings, model names and
command hashes, never prompt or transcript text. Python 3.9+, stdlib only.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # reuse the shared helpers when available
    import eta as _eta
except Exception:  # pragma: no cover - eta.py always sits next to this file
    _eta = None

CACHE_VERSION = 8
STALE = 20 * 60            # an idle gap this long ends a stretch of parent work
ORPHAN = 24 * 3600         # an open delegate older than this is treated as abandoned
ORPHAN_COMMAND = 6 * 3600  # same for a background command with no completion notice
MIN_LEVEL_N = 8            # conditional samples needed before trusting a history level
MIN_ANY_N = 3              # below this even the broadest level falls back to elapsed-based rules
LONG_COMMAND = 300         # commands at least this long count as "long waits" for tails
MAX_DELEGATE = 12 * 3600   # longer "runs" are resumed sessions, not one delegate run
SYSTEM_PREFIXES = ("<task-notification>", "This session is being continued", "<local-command",
                   "Caveat: The messages below", "Delegated task ", "[Image", "<command-message>",
                   "Base directory for this skill", "[Request interrupted", "<system-reminder>")
DELEGATE_TOOLS = ("Agent", "Task")
T3_DELEGATE_TOOLS = ("mcp__t3-code__delegate_task", "delegate_task")
SHELL_TOOLS = ("Bash", "bash")


# ---------------------------------------------------------------- small helpers

def parse_ts(value):
    if _eta is not None:
        return _eta.parse_ts(value)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt_dur(s):
    if s is None:
        return "?"
    s = max(0, int(round(s)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{round(s / 60)} min"
    h, m = divmod(round(s / 60), 60)
    return f"{h}h {m:02d}m"


def eta_home():
    return os.path.expanduser(os.environ.get("ETA_HOME", "~/.eta"))


def cache_path():
    return os.path.join(eta_home(), "waits-cache.json")


def progress_path():
    return os.path.join(eta_home(), "waits-progress.json")


def short(text, n=60):
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def command_hash(command):
    norm = re.sub(r"\s+", " ", str(command or "")).strip()
    return hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()[:12] if norm else None


ALIASES = ("opus", "sonnet", "haiku", "fable")


def model_family(model):
    """claude-opus-5-5 -> claude-opus, gpt-6.2-astra -> gpt-astra, opus -> claude-opus."""
    if not model:
        return None
    m = str(model).lower().split("[")[0]
    if m in ALIASES:
        return "claude-" + m
    m = re.sub(r"(?<=[a-z])-?\d+(?:[.-]\d+)*", "", m)
    m = re.sub(r"-\d{6,8}$", "", m)
    m = re.sub(r"--+", "-", m).strip("-")
    return m or None


def _write_private(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(data)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _tail_text(path, nbytes=65536):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _block_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") in ("text", "input_text"):
                    out.append(b.get("text") or "")
                elif isinstance(b.get("content"), (str, list)):
                    out.append(_block_text(b["content"]))
            elif isinstance(b, str):
                out.append(b)
        return "\n".join(out)
    return ""


def _genuine_prompt(text):
    t = re.sub(r"<system-reminder>.*?</system-reminder>", "", text or "", flags=re.S).strip()
    if not t or t.startswith(SYSTEM_PREFIXES):
        # A slash command typed by the user is a genuine prompt.
        return bool(re.match(r"\s*<command-message>|\s*<command-name>", text or ""))
    return True


def _is_eta(text):
    if _eta is not None:
        m = re.search(r"<command-name>/?(.*?)</command-name>", text or "", re.S)
        if m:
            return m.group(1).strip().split(":")[-1] == "eta"
        return _eta.is_eta_prompt(text)
    return bool(re.match(r"\s*(/eta|\$eta|/skill:eta)\b", text or ""))


# ---------------------------------------------------------------- timelines

class Span:
    """One thing the parent waited on: a delegate or a shell command."""
    __slots__ = ("id", "what", "via", "label", "model", "start", "end", "bg", "ref",
                 "output", "status", "chash", "timeout", "extra")

    def __init__(self, id, what, via, start, label="", model=None, bg=False):
        self.id, self.what, self.via, self.start = id, what, via, start
        self.label, self.model, self.bg = label, model, bg
        self.end = self.ref = self.output = self.status = self.chash = self.timeout = None
        self.extra = {}

    def __repr__(self):
        return f"Span({self.what}/{self.via} {self.label!r} {self.start}-{self.end})"


class Timeline:
    def __init__(self, harness, path):
        self.harness, self.path = harness, path
        self.spans = []
        self.prompts = []        # (ts, is_eta) genuine user prompts
        self.activity = []       # parent assistant/tool timestamps
        self.calls = 0           # tool calls made by this agent
        self.session = None
        self.cwd = None
        self.last_ts = None

    def open_spans(self):
        return [s for s in self.spans if s.end is None]


def _loads(line):
    try:
        return json.loads(line)
    except ValueError:
        return None


T3_DONE_RE = re.compile(r"Delegated task (node:delegated-task:[^\s()]+)")
T3_TASKID_RE = re.compile(r'"taskId"\s*:\s*"(node:delegated-task:[^"]+)"')
OUTPUT_PATH_RE = re.compile(r"Output is being written to: (\S+?\.output)")


def _tag(block, name):
    m = re.search(rf"<{name}>([^<]*)</{name}>", block)
    return m.group(1).strip() if m else None


def _close_notified(text, ts, by_task, by_tool):
    for block in text.split("<task-notification>")[1:]:
        task_id, tool_id, status = _tag(block, "task-id"), _tag(block, "tool-use-id"), _tag(block, "status")
        s = by_task.get(task_id) or by_tool.get(tool_id)
        if s is not None and s.end is None and status:
            s.end, s.status = ts, status
            s.output = s.output or _tag(block, "output-file")


ROLE_RE = re.compile(rb'"type":\s?"(?:user|assistant)"')


def scan_claude(path, until=None):
    """Parse a Claude Code parent transcript into a Timeline, ignoring records after `until`."""
    tl = Timeline("claude", path)
    by_tool = {}          # tool_use id -> Span
    by_task = {}          # background task id / agent id -> Span
    by_t3 = {}            # T3 task id -> Span
    with open(path, "rb") as f:
        for raw in f:
            if b'"isSidechain":true' in raw[:400]:
                continue
            if not ROLE_RE.search(raw):
                # Completion notices that arrive mid-turn are queued, not written as user records.
                if b"<task-notification>" in raw and (b'"queue-operation"' in raw[:60] or b'"queued_command"' in raw):
                    d = _loads(raw)
                    ts = parse_ts((d or {}).get("timestamp"))
                    if not d or ts is None or (until is not None and ts > until):
                        continue
                    if d.get("type") == "queue-operation" and d.get("operation") != "enqueue":
                        continue
                    att = d.get("attachment") if isinstance(d.get("attachment"), dict) else {}
                    text = d.get("content") if isinstance(d.get("content"), str) else str(att.get("prompt") or "")
                    _close_notified(text, ts, by_task, by_tool)
                continue
            d = _loads(raw)
            if not isinstance(d, dict) or d.get("isSidechain") or d.get("type") not in ("user", "assistant"):
                continue
            ts = parse_ts(d.get("timestamp"))
            if ts is None or (until is not None and ts > until):
                continue
            tl.last_ts = ts
            tl.session = tl.session or d.get("sessionId")
            tl.cwd = tl.cwd or d.get("cwd")
            msg = d.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if d["type"] == "assistant":
                tl.activity.append(ts)
                for b in content if isinstance(content, list) else []:
                    if not isinstance(b, dict) or b.get("type") != "tool_use":
                        continue
                    tl.calls += 1
                    name, inp, tid = b.get("name") or "", b.get("input") or {}, b.get("id")
                    if not isinstance(inp, dict):
                        inp = {}
                    if name in DELEGATE_TOOLS:
                        s = Span(tid, "delegate", "agent", ts, short(inp.get("description") or "agent"),
                                 inp.get("model"), bool(inp.get("run_in_background")))
                    elif name in T3_DELEGATE_TOOLS or name.endswith("__delegate_task"):
                        target = inp.get("target") if isinstance(inp.get("target"), dict) else {}
                        s = Span(tid, "delegate", "t3", ts, short(inp.get("title") or "delegate"),
                                 target.get("model"), True)
                        s.extra["provider"] = target.get("providerInstanceId")
                    elif name in SHELL_TOOLS:
                        cmd = inp.get("command") or ""
                        s = Span(tid, "command", "shell", ts, short(inp.get("description") or cmd),
                                 None, bool(inp.get("run_in_background")))
                        s.chash, s.timeout = command_hash(cmd), inp.get("timeout")
                        s.extra["command"] = cmd
                    else:
                        continue
                    tl.spans.append(s)
                    by_tool[tid] = s
                continue
            # user records
            result = d.get("toolUseResult")
            blocks = content if isinstance(content, list) else []
            results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
            if results:
                tl.activity.append(ts)
                for b in results:
                    s = by_tool.get(b.get("tool_use_id"))
                    if s is None or s.end is not None:
                        continue
                    info = result if isinstance(result, dict) else {}
                    text = _block_text(b.get("content"))
                    if s.via == "agent":
                        s.model = info.get("resolvedModel") or s.model
                        if info.get("status") == "async_launched" or info.get("isAsync"):
                            s.bg, s.ref = True, info.get("agentId")
                            s.output = info.get("outputFile")
                            if s.ref:
                                by_task[s.ref] = s
                        else:
                            s.end, s.status = ts, ("error" if b.get("is_error") else info.get("status") or "completed")
                            s.extra["calls"] = info.get("totalToolUseCount")
                    elif s.via == "t3":
                        m = T3_TASKID_RE.search(text.replace('\\"', '"'))
                        if m and not b.get("is_error"):
                            s.ref = m.group(1)
                            by_t3[s.ref] = s
                        else:
                            s.end, s.status = ts, "error"
                    else:  # shell
                        bg_id = info.get("backgroundTaskId")
                        if bg_id:
                            s.bg, s.ref = True, bg_id
                            m = OUTPUT_PATH_RE.search(text)
                            s.output = m.group(1) if m else None
                            by_task[bg_id] = s
                        else:
                            s.end = ts
                            s.status = "interrupted" if info.get("interrupted") else (
                                "error" if b.get("is_error") else "completed")
                continue
            text = _block_text(content)
            if d.get("isMeta") or d.get("isCompactSummary"):
                continue
            if "<task-notification>" in text[:200]:
                _close_notified(text, ts, by_task, by_tool)
                continue
            m = T3_DONE_RE.search(text[:400])
            if m:
                s = by_t3.get(m.group(1))
                if s is not None and s.end is None:
                    s.end, s.status = ts, "completed"
                continue
            if d.get("turnOrigin") in ("task_notification", "system"):
                continue
            if _genuine_prompt(text):
                tl.prompts.append((ts, _is_eta(text)))
    return tl


CELL_RUNNING_RE = re.compile(r"Script running with cell ID (\S+)")
CODEX_CMD_RE = re.compile(r'cmd\s*:\s*(?:"((?:[^"\\]|\\.)*)"|`((?:[^`\\]|\\.)*)`|'
                          r"'((?:[^'\\]|\\.)*)')", re.S)


def scan_codex(path, until=None):
    tl = Timeline("codex", path)
    agents = {}       # agent_thread_id -> open Span
    models = {}       # agent_thread_id -> model
    by_t3 = {}
    exec_calls = {}   # call_id -> Span (exec cell)
    cells = {}        # cell id -> Span
    waits = {}        # wait call_id -> cell id
    with open(path, "rb") as f:
        for raw in f:
            d = _loads(raw)
            if not isinstance(d, dict):
                continue
            ts = parse_ts(d.get("timestamp"))
            if ts is None or (until is not None and ts > until):
                continue
            p = d.get("payload") or {}
            typ = d.get("type")
            if typ == "session_meta":
                tl.cwd = tl.cwd or p.get("cwd")
                tl.session = tl.session or p.get("id")
                continue
            tl.last_ts = ts
            ptype = p.get("type") if isinstance(p, dict) else None
            if typ == "event_msg" and ptype == "item_completed":
                it = p.get("item") or {}
                k = it.get("type")
                if k == "UserMessage":
                    text = _block_text(it.get("content"))
                    m = T3_DONE_RE.search(text[:400])
                    if m:
                        s = by_t3.get(m.group(1))
                        if s is not None and s.end is None:
                            s.end, s.status = ts, "completed"
                        continue
                    if _genuine_prompt(text):
                        tl.prompts.append((ts, _is_eta(text)))
                    continue
                tl.activity.append(ts)
                if k == "SubAgentActivity":
                    aid, kind = it.get("agent_thread_id"), it.get("kind")
                    if it.get("model") and it.get("model") != "None":
                        models[aid] = it.get("model")
                    if kind in ("started", "interacted"):
                        if aid in agents and agents[aid].end is None:
                            continue  # already running; a nudge does not start a new run
                        s = Span(it.get("id"), "delegate", "codex", ts,
                                 short((it.get("agent_path") or "agent").split("/")[-1]),
                                 models.get(aid), True)
                        s.ref = aid
                        s.extra["followup"] = kind == "interacted"
                        tl.spans.append(s)
                        agents[aid] = s
                    elif kind == "completed":
                        s = agents.get(aid)
                        if s is not None and s.end is None:
                            s.end, s.status = ts, "completed"
                elif k == "McpToolCall" and str(it.get("tool", "")).endswith("delegate_task"):
                    dur = it.get("duration") or {}
                    start = ts - (dur.get("secs", 0) + dur.get("nanos", 0) / 1e9) if isinstance(dur, dict) else ts
                    args = it.get("arguments") if isinstance(it.get("arguments"), dict) else {}
                    target = args.get("target") if isinstance(args.get("target"), dict) else {}
                    s = Span(it.get("id"), "delegate", "t3", start, short(args.get("title") or "delegate"),
                             target.get("model"), True)
                    m = T3_TASKID_RE.search(json.dumps(it.get("result")).replace('\\"', '"'))
                    if m:
                        s.ref = m.group(1)
                        by_t3[s.ref] = s
                    else:
                        s.end, s.status = ts, "error"
                    tl.spans.append(s)
                elif k == "CommandExecution":
                    dur = it.get("duration") or {}
                    secs = (dur.get("secs", 0) + dur.get("nanos", 0) / 1e9) if isinstance(dur, dict) else 0
                    cmd = it.get("command")
                    text = cmd[-1] if isinstance(cmd, list) and cmd else str(cmd or "")
                    s = Span(it.get("id"), "command", "shell", ts - secs, short(text))
                    s.end, s.chash = ts, command_hash(text)
                    s.status = "completed" if it.get("exit_code") == 0 else "error"
                    s.extra["history_only"] = True
                    tl.spans.append(s)
            elif typ == "response_item" and ptype in ("custom_tool_call", "function_call"):
                tl.activity.append(ts)
                tl.calls += 1
                name = p.get("name")
                if ptype == "custom_tool_call" and name == "exec":
                    m = CODEX_CMD_RE.search(p.get("input") or "")
                    label = next((g for g in m.groups() if g is not None), "") if m else ""
                    label = label.replace("\\\\", "\\") or "exec cell"
                    s = Span(p.get("call_id"), "command", "shell", ts, short(label))
                    s.chash = command_hash(label) if m else None
                    s.extra["cell"] = True
                    s.extra["command"] = label if m else ""
                    exec_calls[p.get("call_id")] = s
                elif ptype == "function_call" and name == "wait":
                    try:
                        waits[p.get("call_id")] = json.loads(p.get("arguments") or "{}").get("cell_id")
                    except ValueError:
                        pass
            elif typ == "response_item" and ptype in ("custom_tool_call_output", "function_call_output"):
                text = _block_text(p.get("output")) if not isinstance(p.get("output"), str) else p.get("output")
                cid = p.get("call_id")
                if cid in exec_calls:
                    s = exec_calls.pop(cid)
                    m = CELL_RUNNING_RE.search(text[:300])
                    if m:
                        cells[m.group(1)] = s
                        s.ref = m.group(1)
                        tl.spans.append(s)
                elif cid in waits:
                    cell = waits.pop(cid)
                    s = cells.get(cell)
                    if s is not None and not CELL_RUNNING_RE.search(text[:300]):
                        s.end, s.status = ts, "completed"
                        cells.pop(cell, None)
    # exec calls with no output yet are still running
    for s in exec_calls.values():
        tl.spans.append(s)
    return tl


def scan_pi(path, until=None):
    tl = Timeline("pi", path)
    by_tool = {}
    with open(path, "rb") as f:
        for raw in f:
            d = _loads(raw)
            if not isinstance(d, dict):
                continue
            if d.get("type") == "session":
                tl.cwd = d.get("cwd")
                tl.session = d.get("id")
                continue
            if d.get("type") != "message":
                continue
            ts = parse_ts(d.get("timestamp"))
            if ts is None or (until is not None and ts > until):
                continue
            tl.last_ts = ts
            msg = d.get("message") or {}
            role, content = msg.get("role"), msg.get("content")
            if role == "user":
                text = _block_text(content)
                if _genuine_prompt(text):
                    tl.prompts.append((ts, _is_eta(text)))
            elif role == "assistant":
                tl.activity.append(ts)
                for b in content if isinstance(content, list) else []:
                    if isinstance(b, dict) and b.get("type") == "toolCall":
                        tl.calls += 1
                    if isinstance(b, dict) and b.get("type") == "toolCall" and b.get("name") in SHELL_TOOLS:
                        args = b.get("arguments") or {}
                        cmd = args.get("command") or ""
                        s = Span(b.get("id"), "command", "shell", ts, short(cmd))
                        s.chash, s.timeout = command_hash(cmd), args.get("timeout")
                        s.extra["command"] = cmd
                        tl.spans.append(s)
                        by_tool[s.id] = s
            elif role == "toolResult":
                tl.activity.append(ts)
                s = by_tool.get(msg.get("toolCallId"))
                if s is not None and s.end is None:
                    s.end, s.status = ts, "error" if msg.get("isError") else "completed"
    return tl


def sniff_kind(path):
    if _eta is not None:
        return _eta.sniff_kind(path)
    return "codex" if "codex" in path else "pi" if "/.pi/" in path else "claude"


SCANNERS = {"claude": scan_claude, "codex": scan_codex, "pi": scan_pi}
SERVICE_RE = re.compile(
    r"\b(serve|server|http\.server|preview|vite|next dev|(npm|pnpm|yarn|bun) (run )?(dev|start)|--watch|"
    r"\bwatch\b|tail -f|ngrok|cloudflared|tunnel|ssh -[a-zA-Z]*[fNL]|caffeinate|uvicorn|gunicorn|flask run|"
    r"llama-server|mlx_lm\.server|vllm serve|ollama serve|jupyter)", re.I)


def is_service(span):
    """Long-lived helpers (servers, tunnels, watchers) that nobody waits on to finish."""
    return span.what == "command" and bool(SERVICE_RE.search(span.label + " " + (span.extra.get("command") or "")))


def fill_t3(tl, until=None, database=None):
    """T3 does not always post a completion message to the parent; take end times from its database."""
    pending = [s for s in tl.spans if s.via == "t3" and s.ref and s.end is None]
    if not pending:
        return tl
    try:
        state = t3_task_state([s.ref for s in pending], until if until is not None else float("inf"), database)
    except (sqlite3.Error, OSError):
        return tl
    for s in pending:
        info = state.get(s.ref)
        if not info:
            continue
        s.model = s.model or info.get("model")
        if info.get("end") is not None and info.get("status") in ("completed", "failed", "interrupted", "cancelled"):
            s.end, s.status = info["end"], info["status"]
            s.extra["end_from_db"] = True
    return tl


def scan(path, kind=None, until=None):
    tl = SCANNERS[kind or sniff_kind(path)](path, until)
    return fill_t3(tl, until)


# ---------------------------------------------------------------- parent tails

def tails_from_timeline(tl, live_cutoff=None):
    """After each wait (a group of overlapping delegates, or one long command) ends, how long
    did the parent keep working before the next genuine user prompt or a long idle gap?

    Returns [(kind, n_delegates, wait_s, tail_s, end_ts, is_last, task_age_s)]. is_last is 1 when no
    other wait started before the task ended. Groups whose follow-up work is not yet
    over at `live_cutoff` are left out because their tails are still growing.
    """
    out = []
    dele = sorted((s for s in tl.spans if s.what == "delegate" and s.end is not None and s.end >= s.start),
                  key=lambda s: s.start)
    groups = []
    for s in dele:
        if groups and s.start <= groups[-1][1]:
            g = groups[-1]
            groups[-1] = (g[0], max(g[1], s.end), g[2] + 1)
        else:
            groups.append((s.start, s.end, 1))
    long_cmds = [(s.start, s.end, 0) for s in tl.spans if s.what == "command" and s.end is not None
                 and s.end - s.start >= LONG_COMMAND and not s.extra.get("cell") and not is_service(s)
                 and s.status in (None, "completed")]
    acts = sorted(tl.activity)
    prompts = sorted(t for t, is_eta in tl.prompts if not is_eta)   # an /eta ask does not end the task
    import bisect
    wait_starts = sorted([s.start for s in dele] + [c[0] for c in long_cmds])
    for start, end, n in groups + long_cmds:
        # The next genuine prompt after the wait ends closes the task.
        i = bisect.bisect_right(prompts, end)
        stop = prompts[i] if i < len(prompts) else None
        # Only the last wait before the task ends has a pure "integrate and verify" tail.
        k = bisect.bisect_right(wait_starts, end)
        is_last = not (k < len(wait_starts) and (stop is None or wait_starts[k] < stop))
        j = bisect.bisect_right(acts, end)
        last = end
        while j < len(acts) and (stop is None or acts[j] < stop):
            if acts[j] - last > STALE:
                break
            last = acts[j]
            j += 1
        finished = stop is not None or (j < len(acts)) or (live_cutoff is not None and live_cutoff - last > STALE)
        if live_cutoff is None:
            finished = True
        if not finished:
            continue
        p = bisect.bisect_right(prompts, start)
        age = start - prompts[p - 1] if p else None   # how long the task had run when the wait began
        out.append(("delegate" if n else "command", n, end - start, last - end, end, int(is_last), age))
    return out


# ---------------------------------------------------------------- history mining

def _claude_files():
    return glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))


def _codex_files():
    base = os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex"))
    return (glob.glob(os.path.join(base, "sessions/*/*/*/rollout-*.jsonl"))
            + glob.glob(os.path.join(base, "archived_sessions/rollout-*.jsonl")))


def _pi_files():
    if _eta is not None:
        base = _eta.pi_sessions_dir()
    else:
        base = os.path.expanduser("~/.pi/agent/sessions")
    return glob.glob(os.path.join(base, "*", "*.jsonl"))


def _t3_db():
    return Path(os.environ.get("T3_STATE_DB", Path.home() / ".t3/userdata/statev2.sqlite"))


def _key(path):
    """Cache key for a transcript. Folder names can echo a prompt, so only a hash is stored."""
    return hashlib.sha1(path.encode("utf-8", "replace")).hexdigest()[:16]


def all_transcripts():
    return [(p, "claude") for p in _claude_files()] + [(p, "codex") for p in _codex_files()] + \
        [(p, "pi") for p in _pi_files()]


def file_records(path, kind, now):
    """Timing records from one transcript, safe to cache (no text)."""
    tl = scan(path, kind)
    live = None
    try:
        if now - os.path.getmtime(path) < STALE:
            live = now
    except OSError:
        pass
    dl, cm = [], []
    for s in tl.spans:
        if s.end is None or s.end < s.start:
            continue
        if s.what == "delegate":
            if s.via == "t3" or s.end - s.start > MAX_DELEGATE:
                continue  # T3 delegates come from the state database, which covers every harness
            dl.append([s.via, s.model, round(s.start, 3), round(s.end - s.start, 3), int(bool(s.bg)),
                       s.status or "completed"])
        elif s.what == "command" and not s.extra.get("cell") and not is_service(s):
            cm.append([round(s.end, 3), round(s.end - s.start, 3), int(bool(s.bg)), s.chash,
                       s.status or "completed"])
    tails = [[k, n, round(w, 3), round(t, 3), round(e, 3), last, None if a is None else round(a, 1)]
             for k, n, w, t, e, last, a in tails_from_timeline(tl, live)]
    return {"h": kind, "dl": dl, "cm": cm, "tl": tails}


def t3_delegate_rows(now=None, database=None):
    """Completed and running T3 delegated tasks (not continuations). Metadata only."""
    database = Path(database or _t3_db())
    if not database.is_file():
        return []
    db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=0.5)
    try:
        rows = db.execute("""SELECT subagent_id, thread_id, provider, json_extract(payload_json, '$.model'),
                status, started_at, completed_at, child_thread_id
            FROM orchestration_v2_projection_subagents
            WHERE origin='app_owned' AND subagent_id LIKE 'node:delegated-task:command%'""").fetchall()
    finally:
        db.close()
    out = []
    for sid, thread, provider, model, status, started, completed, child in rows:
        s, e = parse_ts(started), parse_ts(completed)
        if s is None:
            continue
        out.append(dict(id=sid, thread=thread, provider=provider, model=model, status=status,
                        start=s, end=e, child=child))
    return out


class History:
    """Aggregated timing history; filter by `now` to avoid looking into the future."""

    def __init__(self, delegates=(), tails=(), commands=(), files=0, complete=True):
        self.delegates = list(delegates)   # dicts: source, harness, model, start, dur, bg, status
        self.tails = list(tails)           # dicts: harness, kind, n, wait, tail, end
        self.commands = list(commands)     # dicts: harness, end, dur, bg, hash, status
        self.files = files
        self.complete = complete

    def before(self, now):
        if now is None:
            return self
        return History([d for d in self.delegates if d["start"] + d["dur"] < now],
                       [t for t in self.tails if t["end"] + t["tail"] < now],
                       [c for c in self.commands if c["end"] < now], self.files, self.complete)


def mine(budget_s=None, now=None, verbose=False):
    """Refresh the cache incrementally: newest changed files first, stop at the budget."""
    now = now or time.time()
    t0 = time.time()
    cache = _read_json(cache_path(), {})
    if cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "files": {}}
    files = cache["files"]
    todo = []
    present = set()
    for p, kind in all_transcripts():
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = _key(p)
        present.add(key)
        sig = [st.st_size, int(st.st_mtime)]
        entry = files.get(key)
        if entry is None or entry.get("sig") != sig:
            todo.append((st.st_mtime, p, kind, sig))
    for p in list(files):
        if p not in present:
            files.pop(p)
    todo.sort(reverse=True)
    done = 0
    for _, p, kind, sig in todo:
        if budget_s is not None and time.time() - t0 > budget_s:
            break
        try:
            rec = file_records(p, kind, now)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            rec = {"h": kind, "dl": [], "cm": [], "tl": [], "err": type(exc).__name__}
        rec["sig"] = sig
        files[_key(p)] = rec
        done += 1
        if verbose and done % 50 == 0:
            print(f"  mined {done}/{len(todo)} files", file=sys.stderr)
    cache["updated"] = now
    cache["pending"] = len(todo) - done
    if done or not os.path.exists(cache_path()):
        _write_private(cache_path(), json.dumps(cache, separators=(",", ":")))
    return cache


def load_history(now=None, budget_s=2.0, cache=None):
    """History for estimates. Spends at most `budget_s` refreshing the cache (None = no limit)."""
    if cache is None:
        try:
            cache = mine(budget_s=budget_s, now=time.time())
        except OSError:
            cache = _read_json(cache_path(), {"files": {}})
    delegates, tails, commands = [], [], []
    seen = set()   # forked and resumed sessions copy earlier records into a new file: count each once

    def first(key):
        if key in seen:
            return False
        seen.add(key)
        return True

    for path, rec in (cache.get("files") or {}).items():
        h = rec.get("h")
        for via, model, start, dur, bg, status in rec.get("dl", []):
            if (status in ("completed", "async_launched") or status is None) and first(("d", h, start, dur)):
                delegates.append(dict(source=via, harness=h, model=model, start=start, dur=dur, bg=bg))
        for k, n, w, t, e, last, age in rec.get("tl", []):
            if first(("t", h, k, e, w)):
                tails.append(dict(harness=h, kind=k, n=n, wait=w, tail=t, end=e, last=last, age=age))
        for e, dur, bg, hsh, status in rec.get("cm", []):
            if first(("c", h, e, dur, hsh)):
                commands.append(dict(harness=h, end=e, dur=dur, bg=bg, hash=hsh, status=status))
    try:
        for r in t3_delegate_rows():
            if r["status"] == "completed" and r["end"] and 0 <= r["end"] - r["start"] <= MAX_DELEGATE:
                delegates.append(dict(source="t3", harness=r["provider"], model=r["model"],
                                      start=r["start"], dur=r["end"] - r["start"], bg=1))
    except (sqlite3.Error, OSError):
        pass
    hist = History(delegates, tails, commands, len(cache.get("files") or {}), not cache.get("pending"))
    return hist.before(now)


# ---------------------------------------------------------------- estimators

def conditional(durations, elapsed):
    """Remaining time among past runs that were still going at `elapsed`."""
    return sorted(d - elapsed for d in durations if d > elapsed)


def summarize(samples, n_label, basis, widen=False):
    p50, low, high = quantile(samples, .5), quantile(samples, .1), quantile(samples, .9)
    if widen or len(samples) < 15:
        low, high = low * 0.7, high * 1.4
        basis += "; range widened for a thin sample"
    return dict(p50_s=p50, low_s=max(0.0, min(low, p50)), high_s=max(high, p50), n=n_label, basis=basis)


def elapsed_fallback(elapsed, what, n_total, n_cond):
    """Lindy-style guess when history has nothing that ran this long: expect about as long again."""
    base = max(elapsed, 60.0)
    return dict(p50_s=0.75 * base, low_s=0.1 * base, high_s=3.0 * base, n=n_cond,
                basis=f"only {n_cond} of {n_total} past {what} ran longer than {fmt_dur(elapsed)}; "
                      f"assumed roughly as long again (wide range)")


def estimate_delegate(model, source, elapsed, history):
    """Remaining time for a running delegate, from delegates that had already run this long."""
    fam = model_family(model)
    pool = history.delegates
    levels = [
        ("same model and harness", lambda d: model and d["model"] == model and d["source"] == source),
        ("same model", lambda d: model and d["model"] == model),
        ("same model family", lambda d: fam and model_family(d["model"]) == fam),
        ("same harness", lambda d: d["source"] == source),
        ("all delegates", lambda d: True),
    ]
    seen = set()
    for label, keep in levels:
        durs = [d["dur"] for d in pool if keep(d)]
        key = len(durs)
        if not durs or (label, key) in seen:
            continue
        seen.add((label, key))
        cond = conditional(durs, elapsed)
        if len(cond) >= MIN_LEVEL_N or (label == "all delegates" and len(cond) >= MIN_ANY_N):
            return summarize(cond, len(cond),
                             f"{len(cond)} of {len(durs)} past delegates ({label}"
                             f"{': ' + str(model) if 'model' in label and model else ''}) were still running at "
                             f"{fmt_dur(elapsed)}; remaining time after that point")
    return elapsed_fallback(elapsed, "delegates", len(pool), len(conditional([d["dur"] for d in pool], elapsed)))


def estimate_command(span, elapsed, history, harness):
    """Remaining time for a running command: same command first, then all commands of the harness."""
    same = [c["dur"] for c in history.commands if span.chash and c["hash"] == span.chash
            and c["status"] == "completed"]
    cond = conditional(same, elapsed)
    if len(cond) >= MIN_ANY_N:
        return summarize(cond, len(cond), f"{len(cond)} earlier successful runs of this exact command "
                                          f"outlasted {fmt_dur(elapsed)}", widen=len(cond) < 5)
    pool = [c["dur"] for c in history.commands if c["harness"] == harness and c["bg"] == int(bool(span.bg))
            and c["status"] == "completed"]
    which = ("background" if span.bg else "foreground") + f" {harness} commands"
    cond = conditional(pool, elapsed)
    if len(cond) < MIN_LEVEL_N:
        pool = [c["dur"] for c in history.commands if c["status"] == "completed"]
        which = "commands from all harnesses"
        cond = conditional(pool, elapsed)
    if len(cond) >= MIN_LEVEL_N:
        res = summarize(cond, len(cond), f"no repeat of this command; {len(cond)} of {len(pool)} past "
                                         f"{which} ran past {fmt_dur(elapsed)}", widen=True)
        if span.timeout:
            cap = max(0.0, span.timeout / 1000.0 - elapsed)
            if not span.bg and cap > 0:
                res["high_s"] = min(res["high_s"], cap)
                res["p50_s"] = min(res["p50_s"], res["high_s"])
                res["low_s"] = min(res["low_s"], res["p50_s"])
                res["basis"] += f"; capped at the tool timeout ({fmt_dur(cap)} left)"
        return res
    return elapsed_fallback(elapsed, which, len(pool), len(cond))


# ---------------------------------------------------------------- progress parsing

_UNITS = {"": 1, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
          "Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3, "Ti": 1024 ** 4}


def _num(value, unit=""):
    unit = (unit or "").replace("B", "").replace("b", "")
    return float(value) * _UNITS.get(unit, _UNITS.get(unit[:1], 1))


def _clock(text):
    """'01:23' or '1:02:03' or '0:00:45' -> seconds; '?' and '--:--' -> None."""
    if not text or "?" in text or "-" in text:
        return None
    parts = [float(x) for x in text.strip().split(":") if x != ""]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


TQDM_RE = re.compile(
    r"(?P<pct>\d{1,3})%\|[^|\n]*\|\s*(?P<done>[\d.]+)(?P<du>[kMGT]?i?B?)?\s*/\s*(?P<total>[\d.]+)(?P<tu>[kMGT]?i?B?)?"
    r"\s*\[(?P<el>[\d:]+)<(?P<rem>[\d:?-]+)")
BYTES_RE = re.compile(r"(?P<done>\d+(?:\.\d+)?)\s*(?P<du>[kKMGT]i?B)\s*/\s*(?P<total>\d+(?:\.\d+)?)\s*(?P<tu>[kKMGT]i?B)")
KOFN_RE = re.compile(
    r"(?:(?P<word>[A-Za-z_]+)s?\s*[:#]?\s*)?\[?\s*(?P<done>\d+)\s*(?:/|\bof\b)\s*(?P<total>\d+)\s*\]?"
    r"(?!\s*(?:[kMGT]i?B|ms|s\b|%|\d))")
CURL_RE = re.compile(
    r"^\s*(?P<pct>\d{1,3})\s+(?P<total>[\d.]+)(?P<tu>[kMGT]?)\s+\d{1,3}\s+(?P<done>[\d.]+)(?P<du>[kMGT]?)\s+\d{1,3}\s+"
    r"[\d.]+[kMGT]?\s+(?P<speed>[\d.]+)(?P<su>[kMGT]?)\s+[\d.]+[kMGT]?\s+[\d:-]+\s+(?P<el>[\d:-]+)\s+(?P<left>[\d:-]+)",
    re.M)
WGET_RE = re.compile(r"\s(?P<pct>\d{1,3})%\s+(?P<speed>[\d.]+)(?P<su>[KMG]?)[ =](?P<left>[\dhms.]+)?")
WGET_LEN_RE = re.compile(r"Length:\s*(\d+)")
PCT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s?%")
FFMPEG_RE = re.compile(r"frame=\s*(\d+)")
UNIT_WORDS = {"frame", "frames", "step", "steps", "item", "items", "batch", "batches", "epoch", "file",
              "files", "chunk", "chunks", "shard", "shards", "sample", "samples", "image", "images",
              "page", "pages", "task", "tasks", "row", "rows", "clip", "clips", "video", "videos",
              "prompt", "prompts", "case", "cases", "job", "jobs", "iter", "iteration", "iterations",
              "it", "doc", "docs", "record", "records", "part", "parts", "layer", "layers", "test", "tests",
              "query", "queries", "request", "requests", "url", "urls", "model", "models", "trial", "trials",
              "scene", "scenes", "segment", "segments", "processed", "done", "completed", "fetching",
              "downloading", "processing", "rendering", "progress"}


def parse_progress(text):
    """Find the latest progress reading in command output. Tolerant and generic.

    Returns None or a dict: done, total, unit ('bytes'|'items'|'percent'), kind (parser name),
    plus elapsed_s / remaining_s / rate when the tool printed them, and `series`: earlier
    (elapsed_s, done) readings from the same tool for recent-rate estimates.
    """
    if not text:
        return None
    text = text.replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()][-400:]
    best = None
    tq_series = []
    for idx, ln in enumerate(lines):
        reading = None
        m = TQDM_RE.search(ln)
        if m:
            done, total = _num(m["done"], m["du"]), _num(m["total"], m["tu"])
            is_bytes = "B" in (m["du"] or "") + (m["tu"] or "") or bool(m["du"] and m["du"][0] in "MGT")
            el, rem = _clock(m["el"]), _clock(m["rem"])
            reading = dict(done=done, total=total, unit="bytes" if is_bytes else "items", kind="tqdm",
                           elapsed_s=el, remaining_s=rem)
            if el is not None:
                tq_series.append((el, done, total))
        if reading is None:
            m = CURL_RE.search(ln)
            if m:
                reading = dict(done=_num(m["done"], m["du"]), total=_num(m["total"], m["tu"]), unit="bytes",
                               kind="curl", rate=_num(m["speed"], m["su"]),
                               elapsed_s=_clock(m["el"]), remaining_s=_clock(m["left"]))
        if reading is None:
            m = BYTES_RE.search(ln)
            if m:
                reading = dict(done=_num(m["done"], m["du"]), total=_num(m["total"], m["tu"]),
                               unit="bytes", kind="bytes")
        if reading is None:
            m = WGET_RE.search(ln)
            if m and re.search(r"\.{5,}", ln):
                reading = dict(done=float(m["pct"]), total=100.0, unit="percent", kind="wget",
                               rate=_num(m["speed"], m["su"]))
        if reading is None:
            for m in KOFN_RE.finditer(ln):
                done, total = int(m["done"]), int(m["total"])
                word = (m["word"] or "").lower()
                if total <= 0 or done > total or total == 1:
                    continue
                # Without a unit word, accept bracketed [k/N] or a bare k/N at the start of a line.
                bare = ln.strip()
                if not (word in UNIT_WORDS or word.rstrip("s") in UNIT_WORDS
                        or re.match(r"^\[?\s*\d+\s*/\s*\d+\s*\]?", bare) or "[" in m.group(0)):
                    continue
                if re.search(r"\d{4}-\d{2}-\d{2}|\d+/\d+/\d+", ln[max(0, m.start() - 3): m.end() + 3]):
                    continue  # dates
                reading = dict(done=float(done), total=float(total), unit="items", kind="k/N",
                               label=word if word in UNIT_WORDS else None)
        if reading is None:
            m = FFMPEG_RE.search(ln)
            if m:
                reading = dict(done=float(m.group(1)), total=None, unit="items", kind="ffmpeg", label="frame")
        if reading is None and not re.search(r"\b(cpu|mem|gpu|util|load|battery|disk|accuracy|acc|loss)\b", ln, re.I):
            ms = PCT_RE.findall(ln)
            if ms and len(ms) <= 2:
                v = float(ms[-1])
                if 0 <= v <= 100:
                    reading = dict(done=v, total=100.0, unit="percent", kind="percent")
        if reading is not None:
            reading["line_index"] = idx
            best = reading
    if best is None:
        return None
    if best["kind"] == "tqdm":
        same_total = [(e, d) for e, d, t in tq_series if t == best["total"]]
        best["series"] = same_total[-30:]
    if best["kind"] == "wget":
        m = None
        for m in WGET_LEN_RE.finditer(text):
            pass
        if m:
            total = float(m.group(1))
            best.update(done=best["done"] / 100 * total, total=total, unit="bytes")
    best["lines_after"] = len(lines) - 1 - best.pop("line_index")
    return best


def progress_estimate(prog, elapsed, prev=None, now=None):
    """Remaining time from measured progress. `prev` is an earlier (ts, done) for the same job."""
    done, total = prog.get("done"), prog.get("total")
    if total is None or done is None or total <= 0:
        return None
    left = max(0.0, total - done)
    if left == 0:
        return dict(p50_s=0.0, low_s=0.0, high_s=60.0, n=1, basis=f"progress shows {done:g}/{total:g}: finishing")
    rates, notes = [], []
    series = prog.get("series") or []
    if len(series) >= 3:
        tail = series[-max(3, len(series) // 4):]
        (e0, d0), (e1, d1) = tail[0], tail[-1]
        if e1 > e0 and d1 > d0:
            rates.append(("recent", (d1 - d0) / (e1 - e0)))
    if prev and now and now > prev[0] and done > prev[1]:
        rates.append(("since last check", (done - prev[1]) / (now - prev[0])))
    if prog.get("rate"):
        r = prog["rate"]
        if prog.get("unit") == "percent" and prog.get("kind") == "wget":
            r = None
        if r:
            rates.append(("reported", r))
    run_el = prog.get("elapsed_s") or elapsed
    if run_el and done > 0:
        rates.append(("average", done / run_el))
    rem_tool = prog.get("remaining_s")
    if not rates and rem_tool is None:
        return None
    ests = [left / r for _, r in rates if r > 0]
    if rem_tool is not None:
        ests.append(rem_tool)
    primary = None
    for label, r in rates:
        if label in ("recent", "since last check") and r > 0:
            primary = (label, left / r)
            break
    if primary is None:
        primary = ("tool estimate", rem_tool) if rem_tool is not None else (rates[0][0], left / rates[0][1])
    p50 = primary[1]
    frac = done / total
    spread_lo, spread_hi = (0.8, 1.35) if frac >= 0.2 else (0.6, 2.0)
    low = min([p50 * spread_lo] + [e for e in ests if e < p50])
    high = max([p50 * spread_hi] + [e for e in ests if e > p50])
    unit = prog.get("label") or prog.get("unit")
    shown = (f"{done / 1e9:.2f}/{total / 1e9:.2f} GB" if prog.get("unit") == "bytes"
             else f"{done:g}/{total:g} {unit}")
    basis = f"{shown} done ({frac:.0%}); {primary[0]} rate -> {fmt_dur(p50)} left"
    if len(ests) > 1:
        basis += f"; other rates give {fmt_dur(min(ests))} to {fmt_dur(max(ests))}"
    return dict(p50_s=p50, low_s=max(0.0, low), high_s=high, n=len(ests), basis=basis)


def _remember_progress(key, now, done):
    """Keep the last progress reading per job so the next ask can measure a recent rate."""
    path = progress_path()
    state = _read_json(path, {})
    prev = state.get(key)
    state[key] = [now, done]
    cutoff = now - 6 * 3600
    state = {k: v for k, v in state.items() if v[0] >= cutoff}
    try:
        _write_private(path, json.dumps(state))
    except OSError:
        pass
    return tuple(prev) if prev and prev[0] < now else None


REDIRECT_RE = re.compile(r"(?:\d?>>?|&>|\btee(?:\s+-a)?)\s*(['\"]?)([^\s|;&'\"<>]+)\1")


def output_sources(span, cwd):
    """Files that may hold a running command's output: the harness output file and redirect targets."""
    paths = []
    if span.output:
        paths.append(span.output)
    cmd = span.extra.get("command") or ""
    base = cwd or os.getcwd()
    cds = re.findall(r"\bcd\s+(['\"]?)([^\s;&|'\"]+)\1", cmd)
    if cds:
        base = os.path.expanduser(cds[-1][1]) if os.path.isabs(os.path.expanduser(cds[-1][1])) else os.path.join(base, cds[-1][1])
    for _, target in REDIRECT_RE.findall(cmd):
        if target in ("/dev/null", "&1", "&2") or target.startswith("&"):
            continue
        p = os.path.expanduser(target)
        paths.append(p if os.path.isabs(p) else os.path.join(base, p))
    seen, out = set(), []
    for p in paths:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            out.append(p)
    return out


def incomplete_downloads(now, window=300):
    """Growing Hugging Face *.incomplete blobs: bytes so far and rate since the file was created."""
    found = []
    for p in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/*/blobs/*.incomplete")):
        try:
            st = os.stat(p)
        except OSError:
            continue
        if now - st.st_mtime > window:
            continue
        born = getattr(st, "st_birthtime", None) or st.st_ctime
        rate = st.st_size / max(1.0, st.st_mtime - born) if st.st_mtime > born else None
        found.append(dict(path=p, bytes=st.st_size, rate=rate,
                          repo=os.path.basename(os.path.dirname(os.path.dirname(p)))))
    return found


# ---------------------------------------------------------------- live delegate progress

def _subagent_file(transcript, agent_id=None, tool_use_id=None):
    session_dir = os.path.splitext(transcript)[0]
    sub = os.path.join(session_dir, "subagents")
    if agent_id:
        p = os.path.join(sub, f"agent-{agent_id}.jsonl")
        if os.path.exists(p):
            return p
        hits = glob.glob(os.path.join(sub, f"agent-*{glob.escape(agent_id)}.jsonl"))
        if hits:
            return hits[0]
    if tool_use_id:
        for meta in glob.glob(os.path.join(sub, "*.meta.json")):
            info = _read_json(meta, {})
            if info.get("toolUseId") == tool_use_id:
                return meta[: -len(".meta.json")] + ".jsonl"
    return None


def transcript_progress(path, kind, now):
    """Tool calls so far and last activity age of a delegate's own transcript."""
    if not path or not os.path.exists(path):
        return None
    try:
        tl = scan(path, kind, until=now) if kind != "claude-sub" else None
    except (OSError, ValueError):
        return None
    if tl is None:  # Claude subagent files are sidechains; count directly
        calls, last, model = 0, None, None
        with open(path, "rb") as f:
            for raw in f:
                d = _loads(raw)
                if not isinstance(d, dict):
                    continue
                ts = parse_ts(d.get("timestamp"))
                if ts is None or ts > now:
                    continue
                last = ts
                msg = d.get("message") or {}
                if d.get("type") == "assistant":
                    model = model or msg.get("model")
                    calls += sum(1 for b in msg.get("content") or [] if isinstance(b, dict) and b.get("type") == "tool_use")
        if last is None:
            return None
        return dict(calls=calls, last_activity_age_s=max(0.0, now - last), model=model)
    if tl.last_ts is None:
        return None
    return dict(calls=tl.calls, last_activity_age_s=max(0.0, now - tl.last_ts))


def _codex_rollout(thread_id):
    if not thread_id:
        return None
    base = os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex"))
    hits = (glob.glob(os.path.join(base, "sessions/*/*/*", f"rollout-*{glob.escape(thread_id)}.jsonl"))
            + glob.glob(os.path.join(base, "archived_sessions", f"rollout-*{glob.escape(thread_id)}.jsonl")))
    return hits[0] if hits else None


def t3_task_state(task_ids, now, database=None):
    """Status, model and child progress for T3 delegated tasks, as known at `now`."""
    database = Path(database or _t3_db())
    if not task_ids or not database.is_file():
        return {}
    out = {}
    try:
        db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=0.5)
    except sqlite3.Error:
        return {}
    try:
        for tid in task_ids:
            row = db.execute("""SELECT provider, json_extract(payload_json,'$.model'), status, started_at,
                    completed_at, child_thread_id FROM orchestration_v2_projection_subagents
                    WHERE subagent_id=?""", (tid,)).fetchone()
            if not row:
                continue
            provider, model, status, started, completed, child = row
            info = dict(provider=provider, model=model, status=status, start=parse_ts(started),
                        end=parse_ts(completed), child=child)
            if info["end"] is not None and info["end"] > now:
                info["end"], info["status"] = None, "running"
            native = db.execute("""SELECT json_extract(payload_json,'$.nativeThreadRef.nativeId')
                    FROM orchestration_v2_projection_provider_threads WHERE thread_id=?""", (child,)).fetchone()
            info["native"] = native[0] if native else None
            out[tid] = info
    except sqlite3.Error:
        pass
    finally:
        db.close()
    return out


def _claude_session_file(session_id):
    if not session_id:
        return None
    hits = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{glob.escape(session_id)}.jsonl"))
    return hits[0] if hits else None


# ---------------------------------------------------------------- the public API

SELF_RE = re.compile(r"\b(eta|waits)\.py\b")
STALE_DELEGATE = 45 * 60  # a delegate whose own transcript is quiet this long is not counted
FRESH = 15  # a foreground command younger than this is the agent's own lookup, not a wait


def _output_finished(span):
    """Claude appends '[exited with code N]' to a background task's output file when it ends."""
    if span.what != "command" or not span.output:
        return False
    return bool(re.search(r"\[exited with code -?\d+\]\s*$", _tail_text(span.output, 200)))


def _wait_dict(span, kind, now, est, progress=None, source=None):
    return dict(kind=kind, label=span.label, model=span.model, started=span.start,
                elapsed_s=max(0.0, now - span.start), progress=progress,
                p50_s=est["p50_s"], low_s=est["low_s"], high_s=est["high_s"], n=est["n"],
                basis=est["basis"], id=span.id, source=source or span.via, background=bool(span.bg))


def open_waits(transcript_path, kind=None, now=None, history=None, remember=True):
    """Everything the session is waiting on at `now`, each with an estimate of time remaining."""
    now = now or time.time()
    kind = kind or sniff_kind(transcript_path)
    if history is None:
        history = load_history(now=now)
    else:
        history = history.before(now)
    tl = scan(transcript_path, kind, until=now)
    waits = []
    began = task_start(tl)
    spans = [s for s in tl.open_spans() if not s.extra.get("history_only") and not is_service(s)
             and now - s.start < (ORPHAN if s.what == "delegate" else ORPHAN_COMMAND)
             and not (s.what == "command" and (SELF_RE.search(s.extra.get("command") or s.label)
                                              or (not s.bg and now - s.start < FRESH)))
             and not _output_finished(s)]
    t3_ids = [s.ref for s in spans if s.via == "t3" and s.ref]
    t3 = {}
    try:
        t3 = t3_task_state(t3_ids, now)
    except (sqlite3.Error, OSError):
        pass
    for s in spans:
        elapsed = max(0.0, now - s.start)
        if s.what == "delegate":
            prog = None
            if s.via == "agent":
                sub = _subagent_file(transcript_path, s.ref, s.id)
                if sub:
                    meta = _read_json(sub[: -len(".jsonl")] + ".meta.json", {})
                    prog = transcript_progress(sub, "claude-sub", now)
                    if not s.model or s.model in ALIASES:
                        s.model = meta.get("model") or (prog or {}).get("model") or s.model
                    if prog:
                        prog.pop("model", None)
                    if meta.get("finished") and prog and prog["last_activity_age_s"] > 30:
                        continue  # finished; the notification has not reached the transcript yet
            elif s.via == "t3" and s.ref in t3:
                info = t3[s.ref]
                if info["end"] is not None:
                    continue
                s.model = s.model or info["model"]
                s.start = info["start"] or s.start
                elapsed = max(0.0, now - s.start)
                native = info.get("native")
                path = _claude_session_file(native) if info["provider"] == "claudeAgent" else _codex_rollout(native)
                if path:
                    prog = transcript_progress(path, "claude" if info["provider"] == "claudeAgent" else "codex", now)
            elif s.via == "codex":
                prog = transcript_progress(_codex_rollout(s.ref), "codex", now)
            est = estimate_delegate(s.model, s.via, elapsed, history)
            quiet = (prog or {}).get("last_activity_age_s", 0)
            wkind = "delegate"
            if quiet > STALE_DELEGATE:
                wkind = "stale"   # no sign of life: likely finished without a notice, or stuck
                est["basis"] = (f"its transcript has been quiet for {fmt_dur(quiet)}; likely finished without a "
                                f"completion notice or stuck, so it is left out of the combined wait")
            elif quiet > 15 * 60:
                est["basis"] += f"; its transcript has been quiet for {fmt_dur(quiet)}"
            waits.append(_wait_dict(s, wkind, now, est, prog))
            continue
        # running command: progress from its output if any, else history
        prog_est, prog = None, None
        for src in output_sources(s, tl.cwd):
            prog = parse_progress(_tail_text(src))
            if prog:
                prev = _remember_progress(f"{transcript_path}:{s.id}", now, prog["done"]) if remember else None
                prog_est = progress_estimate(prog, elapsed, prev, now)
                prog["source"] = src
                break
        if prog_est:
            kind_name = "download" if prog.get("unit") == "bytes" else "batch"
            shown = {k: v for k, v in prog.items() if k in ("done", "total", "unit", "kind", "label", "source")}
            waits.append(_wait_dict(s, kind_name, now, prog_est, shown))
            continue
        est = estimate_command(s, elapsed, history, kind)
        if prog:
            est["basis"] += f"; output shows {prog.get('done'):g} {prog.get('label') or prog.get('unit')} but no total"
        if re.search(r"\b(hf|huggingface-cli)\s+download\b|snapshot_download", s.extra.get("command") or ""):
            grow = incomplete_downloads(now)
            if grow:
                total = sum(g["bytes"] for g in grow)
                rate = sum(g["rate"] or 0 for g in grow)
                prog = dict(bytes=total, rate=rate, unit="bytes", kind="incomplete files")
                est["basis"] += f"; {total / 1e9:.2f} GB in partial files at {rate / 1e6:.1f} MB/s, total unknown"
        waits.append(_wait_dict(s, "command", now, est, prog, "shell"))
    for w in waits:
        w["task_started"] = began
    order = {"delegate": 0, "batch": 1, "download": 2, "command": 3, "stale": 4}
    waits.sort(key=lambda w: (order.get(w["kind"], 9), -w["elapsed_s"]))
    return waits


def task_start(tl):
    """Time of the latest genuine user prompt (not an /eta ask) in a timeline."""
    real = [t for t, is_eta in tl.prompts if not is_eta]
    return real[-1] if real else None


def _cdf(x, w):
    """Approximate CDF of a wait from its 10/50/90 points, piecewise linear in log time."""
    lo, mid, hi = (max(1.0, w[k]) for k in ("low_s", "p50_s", "high_s"))
    lx = math.log(max(1.0, x))
    pts = [(math.log(lo), .1), (math.log(mid), .5), (math.log(hi), .9)]
    if lx <= pts[0][0]:
        slope = .4 / max(1e-6, pts[1][0] - pts[0][0])
        return max(0.0, .1 - (pts[0][0] - lx) * slope)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if lx <= x1:
            return y0 + (y1 - y0) * (lx - x0) / max(1e-9, x1 - x0)
    slope = .4 / max(1e-6, pts[2][0] - pts[1][0])
    return min(1.0, .9 + (lx - pts[2][0]) * slope)


def combine(waits, mode="parallel"):
    """(low, p50, high) for several waits.

    parallel: the waits run at once and the task resumes when the last one ends. Delegates
      launched together tend to be alike, so this assumes they move together and takes the
      max of each quantile. On the history this beat treating them as independent
      (see verify/WAITS.md).
    independent: parallel, but assumes independent durations (pushes the max later).
    sequential: one after another; medians add, spreads add in quadrature.
    """
    waits = [w for w in waits if w and w.get("p50_s") is not None and w.get("kind") != "stale"]
    if not waits:
        return (0.0, 0.0, 0.0)
    if len(waits) == 1:
        w = waits[0]
        return (w["low_s"], w["p50_s"], w["high_s"])
    if mode == "parallel":
        return (max(w["low_s"] for w in waits), max(w["p50_s"] for w in waits), max(w["high_s"] for w in waits))
    if mode == "sequential":
        p50 = sum(w["p50_s"] for w in waits)
        lo = p50 - math.sqrt(sum((w["p50_s"] - w["low_s"]) ** 2 for w in waits))
        hi = p50 + math.sqrt(sum((w["high_s"] - w["p50_s"]) ** 2 for w in waits))
        return (max(lo, max(w["low_s"] for w in waits)), p50, hi)

    def q(target):
        a = math.log(max(1.0, min(w["low_s"] for w in waits))) - 4
        b = math.log(max(1.0, max(w["high_s"] for w in waits))) + 4
        for _ in range(80):
            m = (a + b) / 2
            prob = 1.0
            for w in waits:
                prob *= _cdf(math.exp(m), w)
            if prob < target:
                a = m
            else:
                b = m
        return math.exp((a + b) / 2)

    p50 = max(q(.5), max(w["p50_s"] for w in waits))
    return (max(q(.1), max(w["low_s"] for w in waits)), p50, max(q(.9), p50))


def _bucket_n(n):
    return "1" if n <= 1 else "2+"


def _bucket_wait(w):
    return "<10m" if w < 600 else ">=10m"


def _bucket_age(a):
    return "<5m" if a < 300 else "5-60m" if a < 3600 else ">=1h"


def parent_tail(harness=None, n_delegates=None, wait_s=None, kind="delegate", now=None, history=None,
                final_only=False, task_age_s=None):
    """How long parents kept working after a wait like this one ended, until the task finished.

    Mined across all projects. The task ends at the next genuine user prompt (an /eta ask does
    not count), or at a 20 minute idle gap. By default (final_only=False) every wait counts,
    measured from its end to the end of the task, so later waits and integration are included;
    at ask time nobody knows whether the current wait is the last one, and on the backtest this
    predicted the total better. final_only=True counts only waits that were the last before
    the task ended (pure integrate-and-verify time, usually under a minute).

    Conditioning, in priority order, chosen from the mined tails (verify/WAITS.md):
      task_age_s  how long the task had run when the wait began (<5m, 5-60m, >=1h): the
                  strongest signal; delegate tail medians 4 min, 19 min and 35 min
      wait_s      the wait's own length, under or over 10 minutes
      n_delegates 1 or 2+ parallel delegates (any-wait tails only; no effect on final tails)
      harness     claude or codex
    The most specific combination with at least 12 examples is used; features are dropped
    from the end of that list until one qualifies.
    """
    if history is None:
        history = load_history(now=now)
    else:
        history = history.before(now)
    pool = [t for t in history.tails if t["kind"] == kind and (t.get("last") or not final_only)]
    if not pool:
        pool = [t for t in history.tails if t.get("last") or not final_only]
    feats = []
    if task_age_s is not None:
        ab = _bucket_age(task_age_s)
        feats.append((f"tasks {ab} old at the wait", lambda t, ab=ab: t.get("age") is not None
                      and _bucket_age(t["age"]) == ab))
    if wait_s is not None:
        wb = _bucket_wait(wait_s)
        feats.append((f"{wb} waits", lambda t, wb=wb: _bucket_wait(t["wait"]) == wb))
    if n_delegates and kind == "delegate" and not final_only:
        nb = _bucket_n(n_delegates)
        feats.append((f"{nb} delegates", lambda t, nb=nb: _bucket_n(t["n"]) == nb))
    if harness:
        feats.append((f"{harness}", lambda t: t["harness"] == harness))
    which = "final " if final_only else ""
    for k in range(len(feats), -1, -1):
        use = feats[:k]
        xs = [t["tail"] for t in pool if all(f(t) for _, f in use)]
        if len(xs) >= 12 or (k == 0 and xs):
            label = ", ".join(lbl for lbl, _ in use) or "all parents"
            p50, low, high = quantile(xs, .5), quantile(xs, .1), quantile(xs, .9)
            basis = (f"{len(xs)} past {which}{kind} waits ({label}): the task finished a median "
                     f"{fmt_dur(p50)} after the wait ended")
            if len(xs) < 20:
                low, high = low * 0.7, high * 1.4
                basis += "; range widened for a thin sample"
            return dict(p50_s=p50, low_s=low, high_s=high, n=len(xs), basis=basis)
    return dict(p50_s=300.0, low_s=60.0, high_s=1800.0, n=0,
                basis="no parent-tail history yet; assumed 5 min (1 to 30 min)")


def forecast(transcript_path, kind=None, now=None, history=None, mode="parallel", final_only=False, waits_=None):
    """Waits, their combination (parallel), parent tail and the total to a ready deliverable."""
    now = now or time.time()
    kind = kind or sniff_kind(transcript_path)
    if history is None:
        history = load_history(now=now)
    waits = waits_ if waits_ is not None else open_waits(transcript_path, kind, now, history, remember=False)
    low, p50, high = combine(waits, mode)
    dele = [w for w in waits if w["kind"] == "delegate"]
    live = [w for w in waits if w["kind"] != "stale"]
    if live:
        longest = max(w["elapsed_s"] + w["p50_s"] for w in live)
        first = min(w["started"] for w in live)
        began = live[0].get("task_started")
        age = max(0.0, first - began) if began is not None and began <= first else None
        tail = parent_tail(kind, len(dele) or None, longest, "delegate" if dele else "command", now, history,
                           final_only, age)
    else:
        tail = None
    total = None
    if tail:
        t_lo, t_p50, t_hi = combine([dict(low_s=low, p50_s=p50, high_s=high), tail], "sequential")
        total = dict(low_s=t_lo, p50_s=t_p50, high_s=t_hi)
    return dict(now=now, transcript=transcript_path, kind=kind, waits=waits,
                wait=dict(low_s=low, p50_s=p50, high_s=high) if live else None, tail=tail, total=total)


# ---------------------------------------------------------------- reporting

def history_summary(history):
    lines = []
    by = {}
    for d in history.delegates:
        by.setdefault((d["source"], d["model"]), []).append(d["dur"])
    lines.append(f"delegate runs: {len(history.delegates)} (files mined: {history.files}"
                 f"{'' if history.complete else ', cache still filling'})")
    for (src, model), xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {src:6} {str(model):28} n={len(xs):4}  p10 {fmt_dur(quantile(xs, .1)):>7}  "
                     f"median {fmt_dur(quantile(xs, .5)):>7}  p90 {fmt_dur(quantile(xs, .9)):>7}")
    for kind, final in (("delegate", True), ("delegate", False), ("command", True), ("command", False)):
        ts = [t for t in history.tails if t["kind"] == kind and (t.get("last") or not final)]
        if not ts:
            continue
        xs = [t["tail"] for t in ts]
        lines.append(f"parent tails after {'the final ' if final else 'any '}{kind} wait: n={len(xs)}  "
                     f"p10 {fmt_dur(quantile(xs, .1))}  "
                     f"median {fmt_dur(quantile(xs, .5))}  p90 {fmt_dur(quantile(xs, .9))}")
        for name, key in (("task age", lambda t: "?" if t.get("age") is None else _bucket_age(t["age"])),
                          ("harness", lambda t: t["harness"]), ("delegates", lambda t: _bucket_n(t["n"])),
                          ("wait length", lambda t: _bucket_wait(t["wait"]))):
            groups = {}
            for t in ts:
                groups.setdefault(key(t), []).append(t["tail"])
            parts = [f"{g}: n={len(v)} median {fmt_dur(quantile(v, .5))} (IQR {fmt_dur(quantile(v, .25))}-"
                     f"{fmt_dur(quantile(v, .75))})" for g, v in sorted(groups.items())]
            lines.append(f"  by {name}: " + "; ".join(parts))
    xs = [c["dur"] for c in history.commands]
    if xs:
        lines.append(f"shell commands: n={len(xs)}  median {fmt_dur(quantile(xs, .5))}  "
                     f"p90 {fmt_dur(quantile(xs, .9))}  over 5 min: {sum(x >= 300 for x in xs)}")
    return "\n".join(lines)


def render(fc):
    L = [f"transcript: {fc['transcript']} ({fc['kind']})"]
    if not fc["waits"]:
        L.append("open waits: none (nothing delegated or running right now)")
        return "\n".join(L)
    if fc["wait"] is None:
        L.append("open waits: none live; these look finished or stuck:")
        for w in fc["waits"]:
            L.append(f"  [{w['kind']}] {w['label']}: {w['basis']}")
        return "\n".join(L)
    L.append(f"open waits: {len(fc['waits'])}")
    for w in fc["waits"]:
        head = f"  [{w['kind']}] {w['label']}"
        if w.get("model"):
            head += f"  ({w['model']})"
        L.append(head)
        L.append(f"     running {fmt_dur(w['elapsed_s'])}; about {fmt_dur(w['p50_s'])} left "
                 f"({fmt_dur(w['low_s'])} to {fmt_dur(w['high_s'])}), n={w['n']}")
        if w.get("progress"):
            p = w["progress"]
            bits = []
            if p.get("calls") is not None:
                bits.append(f"{p['calls']} events so far")
            if p.get("last_activity_age_s") is not None:
                bits.append(f"last activity {fmt_dur(p['last_activity_age_s'])} ago")
            if p.get("total"):
                bits.append(f"{p['done']:g}/{p['total']:g} {p.get('label') or p.get('unit')}")
            if bits:
                L.append("     progress: " + ", ".join(bits))
        L.append(f"     basis: {w['basis']}")
    w = fc["wait"]
    L.append(f"all waits done (parallel): about {fmt_dur(w['p50_s'])} ({fmt_dur(w['low_s'])} to {fmt_dur(w['high_s'])})")
    t = fc["tail"]
    L.append(f"parent work after the wait: about {fmt_dur(t['p50_s'])} ({fmt_dur(t['low_s'])} to "
             f"{fmt_dur(t['high_s'])}); {t['basis']}")
    tot = fc["total"]
    L.append(f"deliverable ready in: about {fmt_dur(tot['p50_s'])} ({fmt_dur(tot['low_s'])} to {fmt_dur(tot['high_s'])})")
    return "\n".join(L)


# ---------------------------------------------------------------- backtest

def backtest(history=None, max_cases=60, fractions=(0.25, 0.5, 0.75), out=print, verbose=True):
    """Cut real sessions during delegate waits and long commands; compare with what happened.

    For a cut at time c inside a wait group [s, e] whose task ended at f (last parent activity
    before the next genuine prompt), the truth is: wait remaining e - c and total remaining
    f - c. History is filtered to records that ended before c, so nothing leaks from the future.
    Cases are spread evenly over the whole history.
    """
    history = history or load_history(budget_s=None)
    cache = _read_json(cache_path(), {})
    cands = []
    paths = {_key(p): p for p, _ in all_transcripts()}
    for key, rec in (cache.get("files") or {}).items():
        path = paths.get(key)
        if not path:
            continue
        for k, n, w, t, e, last, age in rec.get("tl", []):
            if w >= 120:
                cands.append((e, path, rec.get("h"), k, n, w, t))
    cands.sort()
    if len(cands) > max_cases:
        step = len(cands) / max_cases
        cands = [cands[int(i * step)] for i in range(max_cases)]
    variants = [("parallel", False), ("parallel", True), ("independent", False), ("independent", True)]
    score = {v: dict(n=0, hit_w=0, hit_t=0, err_w=[], err_t=[]) for v in variants}
    rows = []
    if verbose:
        out(f"{'session':10} {'h':6} {'kind':8} {'n':>2} {'wait':>7} {'cut':>4} | {'wait est':>8} {'actual':>7} "
            f"{'in':>3} | {'total est':>9} {'range':>17} {'actual':>7} {'in':>3}")
    for e, path, h, k, n, w, t in cands:
        for frac in fractions:
            cut = e - w + frac * w
            try:
                ws = open_waits(path, h, now=cut, history=history, remember=False)
            except Exception as exc:  # report and continue
                out(f"error on {os.path.basename(path)}: {type(exc).__name__}")
                continue
            wait_true, total_true = e - cut, e - cut + t
            sid = os.path.basename(path)[:8] if not os.path.basename(path).startswith("rollout") \
                else os.path.basename(path)[-14:-6]
            if not [x for x in ws if x["kind"] != "stale"]:
                if verbose:
                    out(f"{sid:10} {h:6} {k:8} {n:>2} {fmt_dur(w):>7} {frac:>4} | no open wait detected")
                continue
            for mode, final in variants:
                fc = forecast(path, h, now=cut, history=history, mode=mode, final_only=final, waits_=ws)
                sc = score[(mode, final)]
                sc["n"] += 1
                sc["hit_w"] += fc["wait"]["low_s"] <= wait_true <= fc["wait"]["high_s"]
                sc["hit_t"] += fc["total"]["low_s"] <= total_true <= fc["total"]["high_s"]
                sc["err_w"].append(abs(math.log(max(fc["wait"]["p50_s"], 30) / max(wait_true, 30))))
                sc["err_t"].append(abs(math.log(max(fc["total"]["p50_s"], 30) / max(total_true, 30))))
                if (mode, final) == variants[0]:
                    rows.append(dict(session=sid, harness=h, kind=k, n=n, wait=w, cut=frac, fc=fc,
                                     wait_true=wait_true, total_true=total_true))
                    if verbose:
                        wt, tt = fc["wait"], fc["total"]
                        out(f"{sid:10} {h:6} {k:8} {n:>2} {fmt_dur(w):>7} {frac:>4} | {fmt_dur(wt['p50_s']):>8} "
                            f"{fmt_dur(wait_true):>7} {'yes' if wt['low_s'] <= wait_true <= wt['high_s'] else 'no':>3} | "
                            f"{fmt_dur(tt['p50_s']):>9} {fmt_dur(tt['low_s']) + '-' + fmt_dur(tt['high_s']):>17} "
                            f"{fmt_dur(total_true):>7} "
                            f"{'yes' if tt['low_s'] <= total_true <= tt['high_s'] else 'no':>3}")
    for (mode, final), sc in score.items():
        if sc["n"]:
            out(f"{mode:11} tails={'final' if final else 'any  '}: cuts {sc['n']}; wait in range "
                f"{sc['hit_w']}/{sc['n']}, median factor off x{math.exp(quantile(sc['err_w'], .5)):.2f}; "
                f"total in range {sc['hit_t']}/{sc['n']}, median factor off x{math.exp(quantile(sc['err_t'], .5)):.2f}")
    return rows, score


# ---------------------------------------------------------------- CLI

def _find(args):
    if _eta is not None:
        path, kind, how = _eta.find_transcript(args.transcript, args.session, args.agent)
        return path, kind, how
    if args.transcript:
        return args.transcript, sniff_kind(args.transcript), "given"
    return None, None, "eta.py unavailable; pass --transcript"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="waits", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show", help="what the live session is waiting on, with estimates")
    p.add_argument("--session")
    p.add_argument("--transcript")
    p.add_argument("--agent", choices=["claude", "codex", "pi"])
    p.add_argument("--at", help="pretend the time is this ISO timestamp (replay)")
    p.add_argument("--json", action="store_true")
    sub.add_parser("mine", help="build or refresh the history cache fully")
    sub.add_parser("summary", help="history sizes and distributions")
    p2 = sub.add_parser("backtest", help="replay real sessions cut during waits")
    p2.add_argument("--max", type=int, default=60)
    p2.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    os.umask(0o077)
    if args.cmd == "mine":
        cache = mine(budget_s=None, verbose=True)
        print(f"cache: {cache_path()} ({len(cache['files'])} files)")
        print(history_summary(load_history(budget_s=0, cache=cache)))
        return
    if args.cmd == "summary":
        print(history_summary(load_history(budget_s=0)))
        return
    if args.cmd == "backtest":
        backtest(max_cases=args.max, verbose=not args.quiet)
        return
    path, kind, how = _find(args)
    if not path:
        sys.exit(f"no transcript found ({how})")
    now = parse_ts(args.at) if args.at else time.time()
    fc = forecast(path, kind, now)
    if args.json:
        print(json.dumps(fc, indent=2, default=str))
    else:
        print(render(fc))


if __name__ == "__main__":
    main()
