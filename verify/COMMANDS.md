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
