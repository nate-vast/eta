"""Read completed shell commands and GitHub Actions jobs as duration observations."""
import json
import os
import re
import sqlite3
import subprocess
from statistics import median
from typing import NamedTuple, Optional
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlencode, urlsplit


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def remaining(durations, elapsed):
    return sorted(seconds - elapsed for seconds in durations if seconds > elapsed)


def directory(value):
    if isinstance(value, str) and value.startswith("file://"):
        uri = urlsplit(value)
        if uri.netloc not in ("", "localhost"):
            return None
        value = unquote(uri.path)
    return os.path.realpath(value) if value and os.path.isabs(value) else None


def command_text(argv):
    if (isinstance(argv, list) and len(argv) == 3
            and os.path.basename(str(argv[0])) in ("bash", "sh", "zsh", "fish")
            and argv[1] in ("-lc", "-c")):
        return argv[2]
    return None


def shell_samples(path, kind, cwd, command, before):
    pending = {}
    pi_cwd = None
    with open(path, errors="replace") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                if kind == "codex":
                    payload = row.get("payload") or {}
                    if row.get("type") != "event_msg" or payload.get("type") != "item_completed":
                        continue
                    item = payload.get("item") or {}
                    if (item.get("type") != "CommandExecution" or item.get("status") != "completed"
                            or item.get("exit_code") != 0 or directory(item.get("cwd")) != cwd
                            or command_text(item.get("command")) != command):
                        continue
                    duration = item.get("duration") or {}
                    seconds = duration["secs"] + duration["nanos"] / 1e9
                    end = epoch(row["timestamp"])
                    if seconds > 0 and end <= before:
                        yield (end, item["id"]), seconds
                    continue

                if kind == "pi":
                    if row.get("type") == "session":
                        pi_cwd = directory(row.get("cwd"))
                        pending.clear()
                    if pi_cwd != cwd or row.get("type") != "message":
                        continue
                msg = row.get("message") or {}
                content = msg.get("content")
                blocks = content if isinstance(content, list) else []
                if kind == "claude" and (row.get("isSidechain") or directory(row.get("cwd")) != cwd):
                    continue
                ts = epoch(row["timestamp"])
                if ts > before:
                    continue
                if msg.get("role") == "assistant":
                    for block in blocks:
                        args = block.get("input") or block.get("arguments") or {}
                        if (block.get("type") in ("tool_use", "toolCall")
                                and block.get("name") in ("Bash", "bash")
                                and args.get("command") == command and not args.get("run_in_background")):
                            pending[block["id"]] = ts
                elif kind == "pi" and msg.get("role") == "toolResult":
                    start = pending.pop(msg.get("toolCallId"), None)
                    if start is not None and msg.get("isError") is False and ts > start:
                        yield (ts, msg["toolCallId"]), ts - start
                elif kind == "claude":
                    info = row.get("toolUseResult")
                    if not isinstance(info, dict) or info.get("interrupted") or info.get("backgroundTaskId"):
                        continue
                    for block in blocks:
                        start = pending.pop(block.get("tool_use_id"), None)
                        if start is not None and not block.get("is_error") and ts > start:
                            yield (ts, block["tool_use_id"]), ts - start
            except (KeyError, TypeError, ValueError, AttributeError):
                continue


def local_durations(candidates, cwd, command, before):
    samples = {}
    for path, kind in candidates:
        try:
            samples.update(shell_samples(path, kind, cwd, command, before))
        except OSError:
            continue
    return [samples[key] for key in sorted(samples)[-40:]]


def project_files(kind, cwd):
    if kind == "claude":
        slug = re.sub(r"[^A-Za-z0-9]", "-", cwd)
        return Path.home().joinpath(".claude/projects", slug).glob("*.jsonl")
    if kind == "pi":
        base = Path(os.environ.get("PI_CODING_AGENT_SESSION_DIR",
                    Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home()/".pi/agent"))/"sessions"))
        files = base.glob("*/*.jsonl")
    elif kind == "codex":
        files = Path(os.environ.get("CODEX_HOME", Path.home()/".codex")).glob("sessions/*/*/*/rollout-*.jsonl")
    else:
        return []
    matches = []
    for path in files:
        try:
            with path.open(errors="replace") as f:
                meta = json.loads(f.readline())
            if directory((meta.get("payload") or meta).get("cwd")) == cwd:
                matches.append(path)
        except (ValueError, OSError, AttributeError):
            continue
    return matches


def recent_project_files(kind, cwd):
    files = []
    for path in project_files(kind, cwd):
        try:
            files.append((path.stat().st_mtime, path))
        except OSError:
            continue
    return [path for _, path in sorted(files, reverse=True)[:40]]


def gh(endpoint):
    proc = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True)
    if proc.returncode:
        raise ValueError("GitHub job history unavailable")
    return json.loads(proc.stdout)


class CIHistory(NamedTuple):
    remaining: list
    elapsed: float
    state: str
    step: Optional[str] = None
    method: str = "job history"
    late: bool = False
    elapsed_median: Optional[float] = None


def ci_samples(history, elapsed, active):
    """Project from a matching observed step when at least five past jobs qualify.

    Step timings are offsets from job start. Only the current step's identity and
    start are observed in the target job.
    """
    baseline = remaining([duration for duration, _ in history], elapsed)
    if active:
        number, name, started = active
        age = elapsed - started
        samples = []
        fractions = []
        for duration, steps in history:
            for n, label, begin, end in steps:
                if n == number and label == name and 0 <= age < end - begin and 0 <= begin < end <= duration:
                    fractions.append(begin / duration)
                    samples.append(duration - begin - age)
                    break
        if len(samples) >= 5:
            return CIHistory(sorted(samples), elapsed, "running", name, "CI step history",
                             median(fractions) >= .8, median(baseline) if len(baseline) >= 3 else None)
    return CIHistory(baseline, elapsed, "running")


def completed_steps(job):
    start = job.get("started_at")
    if not start:
        return []
    return [(s["number"], s["name"], epoch(s["started_at"]) - epoch(start),
             epoch(s["completed_at"]) - epoch(start))
            for s in job.get("steps", []) if s.get("started_at") and s.get("completed_at")
            and s.get("conclusion") == "success"]


def attempt_jobs(endpoint):
    page = 1
    jobs = []
    while True:
        batch = gh(f"{endpoint}?per_page=100&page={page}")["jobs"]
        jobs.extend((j["name"], j["conclusion"], j["started_at"], j["completed_at"], completed_steps(j))
                    for j in batch)
        if len(batch) < 100:
            return jobs
        page += 1


def ci_history(url, cache_dir, now):
    match = re.fullmatch(r"https://github\.com/([\w.-]+/[\w.-]+)/actions/runs/(\d+)/job/(\d+)/?", url)
    if not match:
        raise ValueError("expected a GitHub Actions job URL")
    repo, run_id, job_id = match.groups()
    api = f"repos/{repo}/actions"
    job = gh(f"{api}/jobs/{job_id}")
    if str(job["run_id"]) != run_id:
        raise ValueError("job and workflow run differ")
    if job["status"] != "in_progress" or not job.get("started_at"):
        return CIHistory([], 0, job.get("conclusion") or job["status"])
    run = gh(f"{api}/runs/{run_id}")
    start = epoch(job["started_at"])
    params = urlencode(dict(branch=run["head_branch"], event=run["event"], status="completed", per_page=25))
    runs = gh(f"{api}/workflows/{run['workflow_id']}/runs?{params}")["workflow_runs"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(os.path.join(cache_dir, "ci-history.sqlite"), isolation_level=None)
    observed = []
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS step_attempts (key TEXT PRIMARY KEY, data TEXT NOT NULL)")
            for old in runs:
                if str(old["id"]) == run_id or epoch(old["created_at"]) >= start:
                    continue
                endpoint = f"{api}/runs/{old['id']}/attempts/{old['run_attempt']}/jobs"
                cached = db.execute("SELECT data FROM step_attempts WHERE key=?", (endpoint,)).fetchone()
                if cached:
                    jobs = json.loads(cached[0])
                else:
                    jobs = attempt_jobs(endpoint)
                    db.execute("INSERT OR IGNORE INTO step_attempts VALUES (?,?)", (endpoint,json.dumps(jobs)))
                for name, status, began, ended, steps in jobs:
                    if name == job["name"] and status == "success" and ended and began and epoch(ended) < start:
                        observed.append((epoch(ended) - epoch(began), steps))
    finally:
        db.close()
    # A first lookup can take several API calls. Check for a finish during the lookup.
    latest = gh(f"{api}/jobs/{job_id}")
    if latest["status"] != "in_progress":
        return CIHistory([], 0, latest.get("conclusion") or latest["status"])
    active = next(((s["number"], s["name"], epoch(s["started_at"]) - start)
                   for s in job.get("steps", []) if s.get("status") == "in_progress"
                   and s.get("started_at") and start <= epoch(s["started_at"]) <= now), None)
    elapsed = max(0, now - start)
    return ci_samples(observed, elapsed, active)
