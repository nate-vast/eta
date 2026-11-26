# Waits: timing what a session is waiting on

`skills/eta/scripts/waits.py` finds the open waits in a live session (delegated agents, running
shell commands, batches and downloads with countable progress). It estimates each from what the
waited-on thing has done so far and from how similar things ran before, then adds the parent's
historical work after a wait ("parent tail"). Measured on 2026-11-25 against this machine's history.

```bash
python3 skills/eta/scripts/waits.py mine        # full history cache, about 18 s for 951 transcripts
python3 skills/eta/scripts/waits.py summary     # sizes and distributions
python3 skills/eta/scripts/waits.py show --session ID [--at ISO] [--json]
python3 skills/eta/scripts/waits.py backtest --max 80 [--quiet]
```

The live path (`show`, `open_waits`) takes about 0.1 s once the cache exists. Without a cache it
spends at most 2 s refreshing, newest files first, and continues on the next call.

## What is on disk (verified)

| Harness | Delegates | Completion | Child progress |
|---|---|---|---|
| Claude Code `Agent`/`Task` | `tool_use` with description, optional `model` alias (`opus`, `fable`); result `toolUseResult.resolvedModel` has the real id | foreground: the `tool_result`; background: result `status: async_launched` + `agentId`, then a `<task-notification>` with that task id | `<session>/subagents/agent-<agentId>.jsonl` plus `.meta.json` (`toolUseId`, `model`, `requestShape`, `finished`) |
| T3 `delegate_task` (any parent) | tool result text holds `taskId` (`node:delegated-task:command...`); DB `orchestration_v2_projection_subagents` holds model, start, end | a user record `Delegated task <taskId> ...`, **but not always**; the DB `completed_at` is authoritative | DB child thread `nativeThreadRef.nativeId` is the child's Claude session id or Codex thread id; its transcript is readable |
| Codex native `spawn_agent` | `item_completed` `SubAgentActivity` `kind: started` with model and `agent_thread_id`; `interacted` starts a follow-up run | `SubAgentActivity kind: completed` for the same thread | rollout `*<agent_thread_id>.jsonl` |
| Claude background `Bash` | result `toolUseResult.backgroundTaskId`, also for foreground commands moved to background at their timeout | `<task-notification>` with status completed, failed, killed or stopped. **Often only as a `queue-operation` enqueue record or a `queued_command` attachment, not a user record.** The output file ends with `[exited with code N]` | `/private/tmp/claude-<uid>/<project>/<session>/tasks/<id>.output` (tmp, cleaned) |
| Codex commands | `CommandExecution` items are logged only on completion, with exact duration | a running `exec` cell answers `Script running with cell ID X`; a later `wait` on X that does not say running ends it | none |
| pi | `toolCall` / `toolResult` pairs | `toolResult` | none |

Verified on real sessions: all Claude Code paths, T3 delegates from Claude and Codex parents,
Codex native sub-agents, Codex exec cells. Built from the format but not checked on a live
example: pi running commands (only 29 pi commands in history, no pi sub-agents), Hugging Face
`*.incomplete` download growth, wget and curl progress (curl rows match 11 real transcript
outputs; the parser test is synthetic).

## History (all projects, read only)

1095 completed delegate runs:

| Source | Model | n | p10 | median | p90 |
|---|---|---:|---:|---:|---:|
| Claude Agent | claude-opus-5-5 | 583 | 1 min | 8 min | 15 min |
| T3 | claude-opus-5-5 | 172 | 3 min | 15 min | 53 min |
| T3 | gpt-6.1-sol | 128 | 2 min | 3 min | 8 min |
| Claude Agent | claude-fable-5-2 | 60 | 4 min | 22 min | 2h 16m |
| Claude Agent | claude-haiku-5-5 | 56 | 9 s | 21 s | 58 s |
| Claude Agent | claude-fable-5-1 | 27 | 41 s | 4 min | 9 min |
| T3 | gpt-6.2-sol | 27 | 3 min | 5 min | 13 min |
| Codex native | gpt-6.2-astra | 24 | 21 s | 57 s | 3 min |
| T3 | gpt-6.2-astra | 9 | 2 min | 5 min | 1h 20m |
| T3 | claude-fable-5-2 | 5 | 3 min | 12 min | 1h 58m |

Plus 31,439 completed shell commands (473 over 5 minutes).

Estimates use runs still going at the current elapsed time (remaining given elapsed). Levels:
same model and harness, same model, same family (`claude-fable`, `gpt-astra`), same harness, all.
A level needs 8 conditional samples. With fewer than 3 at the broadest level the estimate is
"about as long again" with a 0.1x to 3x range. It never refuses.

## Parent tails

The task ends at the next genuine user prompt (an `/eta` ask, a task notification, a T3
"Delegated task" notice or an injected image line does not count) or at a 20 minute idle gap.

| Definition | n (delegate waits) | p10 | median | p90 |
|---|---:|---:|---:|---:|
| after the final wait of the task | 345 | 10 s | 32 s | 5 min |
| after any wait | 1241 | 22 s | 14 min | 2h 40m |

After the final wait, parents mostly just report. But at ask time nobody knows whether the
current wait is the last, and the "any wait" definition predicted totals better on the backtest
(160/236 vs 122/236 inside range), so it is the default.

What separates any-wait delegate tails (Mann-Whitney z, medians):

- task age when the wait began: under 5 min 4 min, 5 to 60 min 19 min, over 1 h 35 min (strongest)
- wait length: under 10 min 12 min, 10 min or more 20 min (z 3.7)
- parallel delegates: 1 16 min, 2+ 8 min (z 4.1)
- harness: Claude 14 min (n 1213), Codex 3 min (n 28) (z 3.3)

For final-wait tails, wait length (z 6.8) and harness (z 2.8) matter and the delegate count does
not (z 0.1). `parent_tail` uses task age, wait length, delegate count and harness in that order,
dropping features from the end until a bucket has 12 examples.

## Backtest

`waits.py backtest --max 80`: 80 wait groups spread over the whole history, each cut at 25, 50
and 75 percent of the wait: 240 cuts from 38 sessions (165 delegate, 75 command), of which 236
had a live wait to score. History is filtered to records that ended before the cut.

| Variant | Wait inside range | Wait median factor off | Total inside range | Total median factor off |
|---|---:|---:|---:|---:|
| parallel (comonotonic), any-wait tails (default) | 197/236 | x1.86 | 160/236 | x2.85 |
| parallel, final-wait tails | 197/236 | x1.86 | 122/236 | x4.05 |
| independent parallel, any-wait tails | 183/236 | x1.87 | 151/236 | x2.78 |
| independent parallel, final-wait tails | 183/236 | x1.87 | 112/236 | x3.99 |

Baselines on the same cuts: the unconditional model median minus elapsed is off by x8.04 on
the wait (conditional estimate closer on 185/236). A fixed 25 minute total is off by x2.49,
**better than this module's total (x2.85)**. Rank correlation of predicted vs actual is 0.35 for
the wait, 0.25 for the tail and 0.20 for the total. The wait estimate carries real signal. The
tail mostly does not. Whether the user writes again in 1 minute or the agent keeps going for an
hour is bimodal and only weakly predictable from these features.

Sample cuts at 50 percent (default variant):

| Session | Kind | Delegates | Wait | Wait est | Actual | In | Total est | Range | Actual | In |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---|
| 48ba6539 | command | 0 | 15 min | 5 min | 7 min | yes | 10 min | 4 min-40 min | 27 min | yes |
| 6217ab48 | command | 0 | 26 min | 4 min | 13 min | yes | 14 min | 4 min-2h 42m | 38 min | yes |
| ebd02f10 | delegate | 4 | 4 min | 46 s | 2 min | yes | 3 min | 54 s-24 min | 4 min | yes |
| 50e42d1d | delegate | 1 | 11 min | 3 min | 5 min | yes | 5 min | 2 min-15 min | 5 min | yes |
| 15474431 | delegate | 1 | 3 min | 44 s | 2 min | yes | 3 min | 1 min-12 min | 60 min | no |
| dcae4d2e | delegate | 1 | 13 min | 2 min | 6 min | yes | 4 min | 2 min-16 min | 51 min | no |
| 692d17cf | delegate | 1 | 10 min | 19 min | 5 min | yes | 19 min | 3 min-2h 20m | 6 min | yes |
| e413538e | delegate | 2 | 13 min | 7 min | 7 min | yes | 18 min | 7 min-1h 06m | 34 min | yes |
| 9dee29a6 | delegate | 1 | 8 min | 10 min | 4 min | no | 21 min | 9 min-1h 13m | 51 min | yes |
| 963e9159 | delegate | 1 | 22 min | 3 min | 11 min | no | 14 min | 3 min-1h 06m | 1h 13m | no |
| 8947752e | delegate | 1 | 16 min | 6 min | 8 min | yes | 17 min | 5 min-53 min | 11 min | yes |

## The four asks from ~/.eta/predictions.jsonl

What `forecast()` says at the moment each ask was made, with history cut at that moment:

| Ask | Agent guess | Wait est (range) | Actual wait | Tail est | Total est (range) | Actual to `done` |
|---|---:|---|---:|---:|---|---:|
| turkey SAM2 segmentation, 17:54 UTC | 22 min | 8 min (36 s-2h 12m) | job killed after 40 min, rewritten | 3 min | 11 min (3 min-2h 22m) | 62 min |
| museum redesign ask 1, 18:11 | 35 min | 34 min (4 min-2h 07m) | 56 min | 2 min | 36 min (7 min-2h 27m) | 91 min |
| 3 research agents, 18:30 | 25 min | 8 min (1 min-14 min) | 1.6 min | 3 min | 11 min (4 min-18 min) | 3.2 min |
| museum redesign ask 2, 18:45 | 25 min | 25 min (7 min-1h 35m) | 22 min | 2 min | 27 min (9 min-1h 59m) | 57 min |

Three of four totals fall inside the range. Research lands just below it. Closer than the agent's
own guess on research and museum ask 2, tied on museum ask 1, worse on turkey.

Notes on the actuals, from the transcripts:
- Museum: at 18:53 the user wrote that the new look was excellent and asked for demo videos for
  every exhibit. The `done` at 19:42 therefore includes that added scope, a GPU rental and two more
  delegates. The Fable delegate itself returned at 19:07:48 (T3 database). T3 never posted its
  completion notice to the parent transcript.
- Turkey: the segmentation ran with output piped through `tail`, so no progress was readable. It
  was killed after 40 minutes and a rewritten version did all 160 frames in 2 minutes. The
  remaining 40 minutes were compositing iterations. No wait model could have seen that.
- Research: three Opus agents finished in 1.8 to 2.6 minutes against an Opus Agent median of 8.

## Integrator API

```python
import waits
hist = waits.load_history(budget_s=2.0)            # once per ask
fc = waits.forecast(transcript_path, kind, now=None, history=hist)
# fc["waits"]: [{kind: delegate|batch|download|command|stale, label, model, started, elapsed_s,
#                progress, p50_s, low_s, high_s, n, basis, id, source, background}]
# fc["wait"]:  {low_s, p50_s, high_s} for all live waits together (None when nothing is open)
# fc["tail"]:  {p50_s, low_s, high_s, n, basis} parent work after the wait
# fc["total"]: {low_s, p50_s, high_s} wait then tail
```

Lower-level pieces: `open_waits(path, kind, now, history)`, `combine(waits, "parallel" |
"independent" | "sequential")`, `parent_tail(harness, n_delegates, wait_s, kind, now, history,
final_only=False, task_age_s=None)`, `parse_progress(text)`, `progress_estimate(prog, elapsed)`.
`stale` waits (delegate transcript quiet over 45 min) are reported but left out of `combine`.
Servers, tunnels and watchers are not waits. The agent's own `eta.py`/`waits.py` call and
foreground commands under 15 s old are ignored.

## Bugs and surprises in existing code and data

Verified:
- `eta.read_claude` and `eta.read_codex` treat T3 `Delegated task ...` notices as user prompts (7
  in the museum transcript, 2 in a Codex rollout). At the second museum ask, `measure()` took a
  delegate notice as the task prompt and reported 30 min elapsed instead of 43.
- Claude Code often records background completions only as `queue-operation` enqueue or
  `queued_command` attachment records. Across all Claude transcripts, 555 of 1082 enqueued
  background-task completions never appear as a user record, so reading only user records loses
  about half of them. `read_claude` ignores them.
- T3 does not always post the delegate completion notice to the parent (museum Fable delegate,
  and the 19:24 Astra delegate). Use `completed_at` in the state database.
- The Claude `Agent` tool's `model` input is an alias (`opus`, `fable`). `resolvedModel` in the
  tool result is the real id.
- The museum `done` actuals (91 and 57 min) include a scope change at 18:53, so they overstate
  the time to the deliverable that was asked about.
- `job_history.shell_samples` skips Claude commands that were moved to the background at their
  timeout, which are exactly the long ones.

Suspected:
- T3 delegate children share the parent's project folder in `~/.claude/projects`, so
  `eta.find_transcript`'s "this folder's newest session" can return a child instead of the parent
  (the museum folder holds three child sessions next to the parent).
- `read_codex` turns `SubAgentActivity` and `CollabAgentToolCall` items into zero-length tool
  events, slightly inflating call counts.
