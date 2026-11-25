# eta

Ask `/eta` during a task to get a remaining duration, finish time, rough range and the evidence behind them. The agent posts the result and continues working. Claude Code uses `/eta`, Codex supports `$eta` and pi uses `/skill:eta`.

## How the forecast works

The agent privately records its own guess for comparison. The skill uses `--measured` for its user-facing estimate. Timing comes from measured batch throughput, previous successful command or CI runtimes, or the median remaining time on at least three comparable verified tasks. Known additional waits are included. A timed job can stand in for the whole task when the remaining work after success is also accounted for.

With too little evidence, the skill reports **“No measured ETA yet.”** Closing that check-in still records an actual duration for future forecasts. Stats reports measured forecast coverage separately. The older CLI comparison mode remains available by omitting `--measured`; the [paired benchmarks](verify/PAIRED.md) evaluated that earlier behavior.

At completion, the agent closes the specific forecast IDs after verifying the requested outcome. Giving up, starting another turn, and transcript inactivity leave paired forecasts open. `/eta stats` reports mean absolute error for both methods, wins, ties, losses, and open forecasts. The legacy call-count method has a separate track record.

Use `/eta stats long` to check whether the skill beats your agent when at least five minutes actually remained. It scores the first measured check-in per verified task, reports how often each estimate was too long, and counts tasks where the skill lacked evidence. Accuracy on long, open-ended work remains unproven.

## Install

```bash
git clone https://github.com/nate-vast/eta ~/projects/eta
~/projects/eta/install.sh
```

The installer links the shared skill and `~/.eta/eta.py`. It also supports `--copy`, `--dir PATH` and `--uninstall`. Python 3.9+ is required. Run from a consistent project directory. Set `ETA_HOME` to use an isolated history.

On POSIX, ETA restricts its default history directory to its owner and stores history files with mode 0600. A custom `ETA_HOME` keeps its existing directory permissions. The installer preserves unrecognized launcher and helper files. If upgrading an older unmarked `--copy` installation, move those files aside and run the installer again.

## CLI

```bash
python3 ~/.eta/eta.py estimate --measured --agent-min BASELINE_MINUTES --model MODEL --agent codex \
    --reference "remaining stage and scale" --task-id UNIQUE_JOB
python3 ~/.eta/eta.py done --id PREDICTION_ID
python3 ~/.eta/eta.py stats
```

Use `references --model MODEL --agent codex` to look up existing remaining-work labels for this scope. Use `--batch DONE TOTAL SECONDS` for measured progress covering all remaining variable work. `--wait-min` adds a known, nonoverlapping wait. `--json` exposes both predictions in seconds. Multiple forecasts for the same job share a task ID and can be closed together with `done --id ID1 ID2`.

For a running command use `--command "make test" --elapsed-s 120 --tail-min 0`. For GitHub Actions use `--ci-job JOB_URL --tail-min MINUTES`. Supply the time needed after success, including verification or deployment. Change the event to `started`, `retry`, `overrun` or `scope-change` when the agent observes a transition and issues another forecast for the same task. No service runs in the background. Each forecast is scored from its own issue time. See [the longer-task benchmark](verify/COMMANDS.md).

Existing `measure`, `estimate --calls`, `estimate --minutes` and session-based `done` commands still support the previous transcript workflow. Legacy estimates apply their correction to variable work and add known waits afterward. For the open-model invocation tests, see [verification results](verify/RESULTS.md).

Run the test suite with `python3 -m unittest discover -s tests -v`.
