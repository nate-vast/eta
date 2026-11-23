# eta

Type `/eta` and your coding agent tells you when the current task will be done, with a clock time, a range, and the basis for it. It works with any agent in any harness.

```
ETA: ~14 min (done around 3:42 PM, likely 9 to 24 min)
Done so far: parser rewritten, 12 of 20 tests passing.
Left: fix 8 tests, update docs, commit (about 38 tool calls + a 2 min build).
Basis: 19 s per tool call measured on this task; past ETAs ran x1.4 long, corrected. Confidence: medium.
```

Then the agent keeps working. `/eta` is a check-in, not a stop.

## Why it's accurate

Agents are bad at guessing wall-clock time. They think in human-developer hours and have no clock sense. So `/eta` never asks the model to guess minutes:

1. **Measure.** `eta.py measure` reads the session's own transcript: when this task's prompt came in, active working time (time spent waiting on you is excluded), tool calls made, seconds per call, long operations (builds, downloads), todo-list progress and step pace.
2. **Count.** The agent lists the remaining steps and counts the tool calls each needs, plus known long waits. Counting steps is something models are good at.
3. **Convert.** `eta.py estimate` turns calls into time with the measured pace, then applies a calibration factor.
4. **Calibrate.** Every ETA is logged in `~/.eta/`. Later runs check each old prediction against the transcript to see when the work really stopped (or the agent runs `eta.py done`). The median actual/predicted ratio of the last 40 becomes the correction, and the spread of those ratios sets the range. If your agent always runs 40% long, ETAs get 40% longer, automatically.

`/eta stats` shows the track record: how often the real finish landed inside the range, and the median error.

## Any agent, any harness

| Harness | How `/eta` is invoked | Pace data |
|---|---|---|
| Claude Code (CLI, desktop, T3 Code, Agent SDK) | `/eta` | transcript, automatic (exact session via `${CLAUDE_SESSION_ID}`) |
| Codex | `$eta` or `/skills` | transcript, automatic (`~/.codex/sessions`) |
| pi | `/skill:eta` | transcript, automatic (`~/.pi/agent/sessions`) |
| OpenCode, Gemini CLI, Cursor, Amp, others with Agent Skills | the tool's skill invocation | self-reported: the agent passes `--started` and `--calls-done` from its own context, and runs `eta.py done` at the end |
| Anything else | paste the body of `skills/eta/SKILL.md` as a prompt | self-reported |

**Open models** (tested 2026-11-23 in pi with local gpt-oss-20b and Qwen3-Coder-30B-A3B, see [verify/RESULTS.md](verify/RESULTS.md)): gpt-oss follows the full procedure mid-task and keeps working, though it tends to post the ETA text at the end of its turn. Qwen3-Coder-30B often ignores an `/eta` that arrives mid-turn. Re-run with `verify/rpc_eta.py <provider> <model> <label>`.

Adding a transcript reader for another harness is one function in `eta.py` that emits the shared event shapes (prompt, tool, tool_done, todos); see `read_pi` for a 30-line example.

## Install

```bash
git clone https://github.com/nate-vast/eta ~/projects/eta
~/projects/eta/install.sh
```

That links the skill into `~/.claude/skills/eta` and `~/.agents/skills/eta` (read by Codex, pi, OpenCode and others) and the script to `~/.eta/eta.py`. Options: `--copy` to copy instead of link, `--dir PATH` to add another skills folder, `--uninstall` to remove (your `~/.eta` history stays). Needs Python 3.9+, no dependencies. Start a new agent session after installing.

## CLI

```bash
python3 ~/.eta/eta.py measure  [--session ID] [--from N] [--started 14:05 --calls-done 40]
python3 ~/.eta/eta.py estimate --calls 38 [--wait-min 2] [--note "tests, docs, commit"]
python3 ~/.eta/eta.py done
python3 ~/.eta/eta.py stats
```

`--from N` picks an earlier prompt as the task start when you gave follow-ups. `--json` on measure/estimate for machine output. `ETA_HOME` moves the data folder.

## How a finish time is detected

For a prediction made at time T, the task counts as finished at the agent's last action before your next real prompt (other `/eta` prompts and background-task notifications don't count). If there is no next prompt yet, it waits until the transcript has been quiet for 20 minutes. If the agent stopped within 30 seconds of answering, that prediction is marked inconclusive and left out. `eta.py done` overrides all of this with the exact time.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Files

- `skills/eta/SKILL.md`: the skill (Agent Skills format)
- `skills/eta/scripts/eta.py`: measure, estimate, done, stats (stdlib only)
- `install.sh`: installer for every harness on the machine
- `tests/test_eta.py`: transcript readers, pace math, calibration, self-reported mode
