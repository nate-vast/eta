# Command history and real-task timing

## GitHub Actions replay

We collected the latest 24 main-branch push runs of pytest’s test workflow available on November 24, 2026. We reserved the six newest run IDs and used only successful timings available before the holdout began. The two jobs were Windows Python 3.11 and Ubuntu Python 3.12. Forecasts were issued in replay at two and four minutes, conditional on the job still running. The independent agent saw the elapsed time and the same 16 prior successful durations for each job. It supplied four blind predictions before the holdout outcomes were scored.

Four held-out workflows completed both jobs, one cancelled both, and one had neither named job. This gave eight successful jobs and 16 scored checkpoints. Two checkpoints on a job share one finish. Cancelled jobs have no successful-completion label. Per-job URLs and timestamps are in [the frozen dataset](ci-history.json).

| Absolute error | Skill | Agent with the same history |
|---|---:|---:|
| Mean | 1.165 min | 1.169 min |
| Median | 0.954 min | 0.958 min |
| Worst | 2.475 min | 2.483 min |

Both methods selected the historical median. Rounding accounts for the small difference. The skill’s descriptive range covered 8 of 16 checkpoints. Run `python3 verify/score_ci.py` for the release CLI replay. An additional GitHub API integration check used an archived job with its running state simulated. It matched 20 prior successes and produced identical samples from an empty and a populated cache. Local project discovery also found five real successful executions of one exact command without any ETA records.

## Prospective whole-task forecasts

Both agents locked their own estimates before the outcome. Neither task had comparable whole-task history or a single timed command covering the remaining work. The skill returned the working agent’s estimate on each.

| Remaining task | Agent and skill | Verified remaining time | Absolute error |
|---|---:|---:|---:|
| Implement and verify the command-history feature | 7m 00s | 6m 20s | 40s |
| Diagnose local transcript timing with three harness fixtures | 4m 00s | 1m 47s | 2m 13s |

The debugging case includes the server restart. Its original prediction was closed after the resumed worker verified the fix. The implementation finish was recorded after integration and regression checks, before writing this report and pushing the branch. The trial provides no evidence of an advantage for open-ended work. These results are separate from the earlier short-task study.

## GitHub Actions step timing

Four chronological replays used disjoint groups of 30 historical pytest main-branch push workflow runs. Each used the first 20 runs for historical observations available before the cutoff and the last ten as targets. Forecasts refer to the successful job finishing, with zero downstream task time. Absolute error is measured in wall-clock seconds.

- At four minutes, the median for matching steps missed by 86.50s on average over 20 jobs, against 85.65s for elapsed-time history. Adding sibling-job speed in a separate 16-job trial missed by 63.29s against 53.06s. Both variants were discarded.
- On the third group, bounding every guess by the middle half of matching step durations improved cleanup forecasts, but the resulting ordinary-agent forecast at four minutes missed by 104.38s against 31.88s for elapsed-time history.
- The final rule accepts in-band guesses. For outside guesses it uses elapsed-time history during earlier steps (matching step median if fewer than three elapsed-time observations qualify). For steps that historically start at or after 80% of job duration by median, it uses the nearer quartile. Both require five matching step observations. The rule was fixed before scoring the final group.

Final group, 16 successful jobs checked twice. Each checkpoint type and information condition had its own blinded agent. Ordinary agents saw the current job and step timing. Informed agents also saw the historical records. Each row reports mean absolute error in seconds.

| Information available to agent | Checkpoint | Agent | Prior release | Updated skill |
|---|---|---:|---:|---:|
| Current job | Four minutes | 110.56 | 44.25 | 44.25 |
| Current job | First cleanup step | 11.00 | 33.12 | 1.69 |
| Current job | Both | 60.78 | 38.69 | 22.97 |
| Current job and history | Four minutes | 47.88 | 44.25 | 47.88 |
| Current job and history | First cleanup step | 1.50 | 33.06 | 1.50 |
| Current job and history | Both | 24.69 | 38.66 | 24.69 |

The update was closer than the ordinary agent at 28 checkpoints and further at four. It tied the informed agent at every checkpoint. The prior release falls back to the supplied guess with sparse history, so its scores can differ between conditions. These checkpoints share jobs and a workflow. They measure CI wait time and do not establish an advantage on open-ended development.

Reproduce the four trials, in order, with:

```sh
python3 verify/score_steps.py
python3 verify/score_steps.py ci-peers.json
python3 verify/score_ci_band.py ci-band.json
python3 verify/score_ci_band.py
```
