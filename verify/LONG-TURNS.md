# Local Codex turn-history experiment

At five minutes into a still-running turn, take the median residual duration of up to 40 previous turns in the same directory and model that were still running at five minutes. Require five distinct earlier finishes. Replay uses completed, uncompressed local transcripts and only outcomes available before the target started. Runtime discovery caps candidate files at 40 per project; this replay searched the full available corpus, so its coverage is optimistic. A turn ends at a final response, which can precede task completion.

On the local November 25 corpus there were 27 eligible turns and two predictions. Their median remaining time was 792 seconds and mean absolute error was 810 seconds. One was too long. There were no contemporaneous agent guesses for comparison. This experiment does not justify replacing the public estimator. Run `python3 verify/score_long_turns.py` for a local aggregate without exporting transcript text.
