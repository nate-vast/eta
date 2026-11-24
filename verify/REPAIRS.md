# First failed test to verified repair

The prospective comparison pins the current merge at `08299d93159a0fd09b04e96a0d18d31ee33ad4b0`. One Codex worker using GPT-6.2 Astra handled nine different small Python defects sequentially in the same fixture repository. Each function had a written contract and executable acceptance checks. The clock stopped at the first successful verification after the fix. Diagnosis, editing, failed attempts and reruns after forecast issuance all counted.

For each task the runner timestamped the first failed check. The worker could inspect the current failure, implementation and contract, then locked its own remaining-time estimate before editing. At that instant the runner also saved the merged CLI forecast and the candidate forecast. It withheld their values from the worker. The baseline had the same guess, consistent reference label, task identity and previously verified outcomes. It could learn after each case.

The candidate uses the first-failure-to-verified-finish durations of earlier comparable repairs. It subtracts the current failure age from historical durations that exceed that age and takes the median remaining time if at least three qualify. With fewer matches it returns the merged forecast. The trial fixed the rule, three training cases, six held-out cases and promotion criterion before the first repair. Each verified holdout becomes eligible history for subsequent forecasts as it would in normal use. The primary measure is mean absolute error in seconds across all six held-out tasks. Promotion requires lower error than both the unaided agent and the merged version.

Run `python3 verify/score_repairs.py` from a clone containing the pinned merge. The script reconstructs an empty ledger, calls the original CLI against the recorded timestamps, and checks each saved forecast before computing errors. These tasks cover short, single-function repairs with local tests.

## Outcome

| Forecast | Held-out mean absolute error (6 repairs) | Including warmup (9 repairs) |
|---|---:|---:|
| Merged ETA | 0.93s | 1.85s |
| Working agent | 2.09s | 2.62s |
| First-failure history | 6.25s | 5.40s |

The candidate had qualifying history on all six held-out repairs. Every candidate estimate was too high and further from completion than both baselines. All nine cases passed the first verification after editing. On one URL-decoding repair, 3.00 seconds remained when the forecast was locked. The worker estimated 3 seconds, the merge estimated about 5.4 seconds, and the candidate estimated about 12.7 seconds. The worker had already inspected the failure before forecasting; elapsed time alone did not capture how much diagnosis was complete. The trial supplies no data on multiple unsuccessful repair attempts or longer debugging sessions. The candidate was rejected for release.

The frozen data includes the broken functions, worker fixes, unmodified acceptance checks, forecast timestamps and verified finish timestamps. The actual is finish minus forecast issuance for all three methods.
