# Paired accuracy trials

On November 24, 2026, the revised skill and the working agents issued 13 prospective paired forecasts. Mean absolute error was 3.45 seconds for the skill and 3.69 seconds for the agents. The skill finished closer six times, the agent six times, and they tied once. These are wall-clock seconds from issuing both forecasts to the successful check or explicit completion marker. The sample covers two workers, small coding tasks, one QA check-in, and controlled compression batches.

Protocol fixed before scoring. One independent working agent receives eight small Python implementation tasks, one at a time. After reading each specification, it records its own remaining seconds before implementing the function. The first passing acceptance check timestamps completion. Every task counts, including cold starts. The agent may consult its own previous timings.

The skill uses the median remaining time of prior, explicitly completed tasks in the same project, harness, model, and remaining-work category once three independent tasks exist. Until then its paired forecast uses the agent's estimate. Each prediction can see only outcomes already available at its forecast timestamp. The primary metric is mean absolute error in seconds over all eight tasks. Also report wins, ties and losses. The test measures short, repeated coding work at the same stage. General coding work and unfamiliar task types require separate measurements.

A second prospective trial uses six new specifications, the same independent worker, and an empty isolated ledger. The installed CLI logs both forecasts before coding and a successful acceptance check immediately runs the installed finish command. The worker can use earlier results and sees each skill estimate after locking its own. The same scoring rule includes every cold start.

## Revised selection rule and fresh holdout

The live median-only trial lost on mean absolute error. Freeze the following revision before opening the next holdout. With three or more eligible prior outcomes, choose among the unchanged agent estimate, an equal-weight average of the agent estimate and the historical median, and the agent estimate multiplied by the median historical actual/agent ratio. Score candidates by leave-one-task-out absolute error on prior tasks only. Ties favor the agent. Apply the selected method once to the new blind guess. Fixed waits sit outside all three calculations.

For the new prospective trial, the same worker implements eight fresh small functions. The previous six confirmed live tasks supply the initial history. Primary outcome is mean absolute error over all eight. No task is excluded because its completion time or prediction is unusual.

A separate live batch test covers three LZMA compression and decompression jobs on equal-size generated blocks. A second worker sees the completed count, total and measured elapsed time before making its own prediction. The runner freezes both forecasts, resumes the job, verifies every remaining block against the input, and closes the prediction. Both methods have access to the same progress information. Score all three jobs. This tests a measurable batch stage rather than open-ended development.

One additional batch extends the time horizon. The worker forecasts the remaining 728 of 760 blocks after verifying 32. Report it separately even if it loses.

## Recorded outcomes

| Trial | Tasks | Skill mean absolute error | Agent mean absolute error | Skill wins / ties / losses |
|---|---:|---:|---:|---:|
| Initial median, chronological replay | 8 | 0.857 s | 0.903 s | 2 / 3 / 3 |
| Initial median, live | 6 | 0.509 s | 0.369 s | 2 / 3 / 1 |
| Revised rule, fresh coding tasks | 8 | 0.465 s | 0.561 s | 4 / 0 / 4 |
| Measured progress, short batches | 3 | 0.674 s | 0.757 s | 1 / 0 / 2 |
| Measured progress, longer batch | 1 | 6.559 s | 8.759 s | 1 / 0 / 0 |
| New QA task with no comparable history | 1 | 32.511 s | 32.511 s | 0 / 1 / 0 |

The longer batch finished in 140.76 seconds. The skill said 134.20 and the agent said 132.00. On the cold QA task both said 45 seconds, and the agent verified completion in 12.49 seconds. The first two rows evaluated the discarded median-only rule. The last four rows count toward the revised skill’s results. The comparison uses full-precision point forecasts and includes every cold start. Different tasks share a worker and machine, and most took seconds. Long open-ended debugging work remains unvalidated.

Run `python3 verify/score_paired.py` to reproduce the scores from [the timestamped pairs](paired-trials.jsonl).
