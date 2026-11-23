---
name: eta
description: Give an accurate, calibrated ETA for the task you are working on right now, measured from this session's real pace instead of guessed. Use when the user types /eta, or asks for an ETA, how long is left, how much longer, or when the current work will be done. Also `/eta stats` for the track record.
argument-hint: "[stats | which task]"
---

# /eta

**The user is asking how long the current task will take. Answer that first, before doing any more work on the task.**

Your very first action must be this tool call:

```bash
python3 ~/.eta/eta.py measure --session "${CLAUDE_SESSION_ID}"
```

Then, in order:
1. Run `estimate` as a second tool call (section 3). Its output starts with `ETA RESULT`.
2. Reply to the user with the numbers `estimate` printed. Never reply with a number you made up.
3. If the task isn't finished, keep working in this same turn. In most harnesses a message with no tool call ends your turn, so put the ETA reply and your next task tool call in the same message: the ETA text first, then the call.

Why: agents guess wall-clock time badly (we tend to think in human-developer hours). So measure your pace from this session, count the work left in tool calls, and let the calibration log correct for your usual bias.

Run commands with the script at `~/.eta/eta.py`. If that path is missing, use `scripts/eta.py` next to this SKILL.md. Pass `--session "${CLAUDE_SESSION_ID}"` exactly as written (Claude Code fills it in; elsewhere it expands to nothing and the script finds the session itself).

User arguments: $ARGUMENTS

If the arguments are `stats`, run `python3 ~/.eta/eta.py stats`, show the table, and stop.

## 1. Measure

```bash
python3 ~/.eta/eta.py measure --session "${CLAUDE_SESSION_ID}"
```

- Check the `task` line. If the task really began at an earlier prompt (the user added follow-ups), rerun with `--from N` using the number from the recent-prompts list. If the user's arguments name a task, pick the prompt that matches.
- If it says **self-reported mode** (a harness whose transcript the script can't read), rerun with `--started <when this task began: 14:05, 2:05pm or 25m>` and `--calls-done <tool calls you've made on it>`, both from your own context. Keep those flags on the estimate call too.
- If it warns that it guessed the session or the transcript looks stale, say so in the reply.

## 2. Count what's left

Think through the remaining steps for this task in order. For each, count the tool calls it will really take: reads, edits, runs, test runs, a fix-and-rerun cycle for anything that can fail, verification, commits, journaling, the final summary. Be honest. Don't drop the boring steps; they are where estimates go wrong.

Separately, add up known long waits in minutes: builds, installs, model downloads, training, deploys, CI, sub-agents, anything that sits there running. Use the long operations the measure step listed as a reference.

Cross-check with the `todo list` step pace if one is shown. If the two disagree a lot, figure out why before choosing.

## 3. Estimate and log

```bash
python3 ~/.eta/eta.py estimate --session "${CLAUDE_SESSION_ID}" --calls N [--wait-min M] --note "what's left, a few words"
```

Use the same `--from`, `--started`, `--calls-done` flags as step 1. Report the numbers it prints as they are; don't shave them. If you truly can't count calls, use `--minutes M` instead of `--calls`.

Skip the estimate (nothing gets logged) when:
- the task is already done (check: is every part of the request finished?): reply "Already done." with a one-line summary;
- you are blocked waiting on the user: say what you need from them;
- there is no task in progress.

## 4. Reply

Keep it short. Fill every <...> from the `ETA RESULT` output and your own step list. Never copy numbers from these instructions:

> **ETA: ~<remaining>** (done around <finish>, likely <low> to <high>)
> Done so far: <one line>.
> Left: <steps> (about <N> tool calls<+ waits, if any>).
> Basis: <basis line from ETA RESULT, in plain words>. Confidence: <confidence>.

If you skipped `estimate` because the task is already done, reply only: **Already done.** plus a one-line summary. No ETA, no numbers.

Then, if you were in the middle of the task when /eta came in, **keep working on it**. /eta is a check-in, not a stop.

## 5. When the task finishes

Run `python3 ~/.eta/eta.py done --session "${CLAUDE_SESSION_ID}"` right before your final message for a task you gave an ETA on (use `--agent` or the same session flags if needed). It records the exact finish time so future ETAs get more accurate. In Claude Code, Codex and pi the script also works this out from the transcript, so a missed `done` there is fine.

## Rules

- Estimate agent wall-clock time, never human-developer time.
- Don't pad to look safe and don't trim to look fast. Every ETA is logged and checked against when the work really finished; the correction factor catches bias either way.
- A range plus a clock time beats a single number. Always give both.
