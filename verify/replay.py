#!/usr/bin/env python3
"""Score ETA estimators against this machine's own task history, without leakage.

    python3 verify/replay.py                    score every registered estimator, write verify/REPLAY.md
    python3 verify/replay.py --features         also run the baseline feature search
    python3 verify/replay.py --module my_est.py add estimators from another module

Method: tasks come from skills/eta/scripts/baseline.py (cache ~/.eta/turns.jsonl).
Fit on tasks that finished before a time cut, score on tasks that started after it.
For each held-out task, pick cut points, rebuild the state a live observer would
have seen at that moment, ask each estimator for (low, mid, high) remaining
seconds, and compare with the actual remaining time.

Plug-in hook: a module passed with --module may define
    register_estimators(register)          and call register(name, factory)
or a dict ESTIMATORS = {name: factory}. A factory takes the list of training
tasks (dicts from baseline.tasks_from_records) and returns fn(state) -> (low, mid, high)
in seconds, or None to decline. Use register_fn(name, fn) for a fixed function.
state["_task"] is the held-out task itself (its timeline and its transcript path "f"), for
estimators that must re-read the transcript as of the cut (t0 + elapsed_s); never read
anything in it after that moment.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "eta", "scripts"))
import baseline as B  # noqa: E402

MIN_GAIN = 0.03      # a feature must cut uniform+late median log miss by this much
FLOOR = 30.0          # seconds; predictions and actuals below this count as 30 s in log error
LATE_FROM = 600.0     # "long already" mode: asks made at least 10 minutes in
CHECKIN_EVERY = 300.0
GRID_ASKS = [60, 300, 900, 1800, 3600, 7200]
MODES = ("uniform", "late", "checkins", "grid", "real")

ESTIMATORS = {}


def register(name, factory):
    """factory(train_tasks) -> fn(state) -> (low, mid, high) | None"""
    ESTIMATORS[name] = factory


def register_fn(name, fn):
    register(name, lambda train: fn)


# ---------------------------------------------------------------- built-in estimators

def _fit_durs(train):
    return sorted(t["wall"] for t in train if B.fit_eligible(t))


def _naive(train):
    def est(state):
        e = state["elapsed_s"]
        return e / 3, e, e * 3
    return est


def _constant(train):
    d = _fit_durs(train)
    lo, mid, hi = (B._eta.quantile(d, q) for q in (0.1, 0.5, 0.9))
    return lambda state: (lo, mid, hi)


register("baseline", lambda train: B.Baseline(train).estimate)
register("remaining=elapsed", _naive)
register("constant-median", _constant)


# ---------------------------------------------------------------- cuts and scoring

def folds(tasks, n_folds=4, start=0.4):
    """Rolling origin: fit-eligible tasks sorted by start time; the last (1 - start) share is
    cut into n_folds chunks. Each chunk is scored by estimators fitted only on tasks that
    had finished before the chunk's first task started."""
    elig = sorted((t for t in tasks if B.fit_eligible(t)), key=lambda t: t["t0"])
    edges = [int(len(elig) * (start + (1 - start) * i / n_folds)) for i in range(n_folds + 1)]
    out = []
    for a, b in zip(edges, edges[1:]):
        cut = elig[a]["t0"]
        train = [t for t in tasks if t["t0"] + t["wall"] < cut]
        out.append((train, elig[a:b], cut))
    return out


def make_cuts(test, mode, seed=0):
    """uniform: one random moment per task after its first minute (tasks over 2 min).
    late: one random moment per task after 10 minutes (tasks over 10 min).
    checkins: an ask every 5 minutes once a task passes 10 minutes, so long tasks weigh
    in proportion to their time (a user who keeps checking).
    grid: an ask at each of 1, 5, 15, 30, 60, 120 minutes the task was still running.
    real: the moments a person actually asked for an ETA in that task."""
    rng = random.Random(seed)
    out = []
    for t in test:
        d = t["wall"]
        if mode == "uniform" and d > 120:
            out.append((t, rng.uniform(60, d)))
        elif mode == "late" and d > LATE_FROM:
            out.append((t, rng.uniform(LATE_FROM, d)))
        elif mode == "checkins":
            e = LATE_FROM + rng.uniform(0, CHECKIN_EVERY)
            while e < d:
                out.append((t, e))
                e += CHECKIN_EVERY
        elif mode == "grid":
            out += [(t, float(e)) for e in GRID_ASKS if e < d]
        elif mode == "real":
            out += [(t, float(e)) for e in t.get("q") or [] if 0 < e < d]
    return out


def elapsed_bucket(e):
    for lim, name in ((120, "<2m"), (600, "2-10m"), (1800, "10-30m"), (3600, "30-60m"),
                      (3 * 3600, "1-3h")):
        if e < lim:
            return name
    return ">3h"


def length_bucket(d):
    for lim, name in ((300, "task<5m"), (1800, "task 5-30m"), (7200, "task 30m-2h")):
        if d < lim:
            return name
    return "task>2h"


def score(rows):
    """rows: [(pred or None, actual)] -> metrics dict."""
    n = len(rows)
    ans = [(p, a) for p, a in rows if p is not None]
    if not ans:
        return {"n": n, "answer": 0.0}
    errs, signed, hits = [], [], 0
    for (lo, mid, hi), a in ans:
        r = math.log(max(mid, FLOOR) / max(a, FLOOR))
        errs.append(abs(r))
        signed.append(r)
        hits += lo <= a <= hi
    errs.sort()
    return {"n": n, "answer": len(ans) / n, "med": statistics.median(errs),
            "p90": errs[int(0.9 * (len(errs) - 1))], "w2x": sum(e <= math.log(2) for e in errs) / len(errs),
            "cov": hits / len(ans), "bias": statistics.median(signed)}


def run_estimators(fold_list, mode, names):
    """Pooled predictions over all folds: {name: [(pred, actual, state, task), ...]}."""
    preds = {n: [] for n in names}
    for train, test, _ in fold_list:
        states = [(dict(B.state_at(t, e), _task=t), t["wall"] - e, t) for t, e in make_cuts(test, mode)]
        for name in names:
            fn = ESTIMATORS[name](train)
            for st, actual, t in states:
                try:
                    p = fn(st)
                except Exception:  # an estimator that crashes declines that ask
                    p = None
                preds[name].append((p, actual, st, t))
    return preds


def _err(p, a):
    return abs(math.log(max(p[1], FLOOR) / max(a, FLOOR))) if p is not None else None


def bootstrap_median(rows, reps=300, seed=1):
    """90% interval for the median miss, resampling whole tasks."""
    by_task = {}
    for p, a, st, t in rows:
        e = _err(p, a)
        if e is not None:
            by_task.setdefault(t["k"], []).append(e)
    keys = list(by_task)
    if len(keys) < 5:
        return None
    rng = random.Random(seed)
    meds = []
    for _ in range(reps):
        errs = []
        for _ in keys:
            errs += by_task[keys[rng.randrange(len(keys))]]
        meds.append(statistics.median(errs))
    meds.sort()
    return math.exp(meds[int(0.05 * reps)]), math.exp(meds[int(0.95 * reps) - 1])


def closer_than(rows, ref_rows):
    """Share of asks where this estimator's mid is closer than the reference's (ties split)."""
    win = 0.0
    n = 0
    for (p, a, _, _), (q, _, _, _) in zip(rows, ref_rows):
        e1, e2 = _err(p, a), _err(q, a)
        if e1 is None or e2 is None:
            continue
        n += 1
        win += 1.0 if e1 < e2 else 0.5 if e1 == e2 else 0.0
    return win / n if n else None


def breakouts(rows):
    groups = {}
    for p, a, st, t in rows:
        for key in ("all", "elapsed " + elapsed_bucket(st["elapsed_s"]), "harness " + t["h"],
                    length_bucket(t["wall"])):
            groups.setdefault(key, []).append((p, a))
    return {k: score(v) for k, v in groups.items()}


ORDER = ["all", "elapsed <2m", "elapsed 2-10m", "elapsed 10-30m", "elapsed 30-60m", "elapsed 1-3h",
         "elapsed >3h", "harness claude", "harness codex", "harness pi", "task<5m", "task 5-30m",
         "task 30m-2h", "task>2h"]


def fmt_row(name, m):
    if not m.get("answer"):
        return f"| {name} | {m['n']} | 0% | | | | | |"
    return (f"| {name} | {m['n']} | {m['answer']:.0%} | {math.exp(m['med']):.2f}x | "
            f"{math.exp(m['p90']):.1f}x | {m['w2x']:.0%} | {m['cov']:.0%} | {math.exp(m['bias']):.2f}x |")


HEADER = ("| group | asks | answered | median miss | p90 miss | within 2x | in 80% range | median bias |\n"
          "|---|---:|---:|---:|---:|---:|---:|---:|")


def table(title, results, by="estimator"):
    lines = [f"#### {title}", "", HEADER]
    for name, m in results:
        lines.append(fmt_row(name, m))
    return "\n".join(lines)


def feature_search(fold_list, modes=("uniform", "late", "checkins")):
    """Forward selection over baseline features. Criterion: sum over the modes of the
    pooled median log miss; a feature must lower it by MIN_GAIN to be kept."""
    cuts = [(train, {m: make_cuts(test, m) for m in modes}) for train, test, _ in fold_list]

    def evaluate(feats, k=B.K_SHRINK):
        rows = {m: [] for m in modes}
        for train, by_mode in cuts:
            b = B.Baseline(train, feats=feats, k=k)
            for m, cs in by_mode.items():
                rows[m] += [(b.estimate(B.state_at(t, e)), t["wall"] - e) for t, e in cs]
        return {m: score(r) for m, r in rows.items()}

    def crit(r):
        return sum(r[m]["med"] for m in modes)

    base = evaluate(())
    rows = [("(survival only)", base)]
    for f in B.ALL_FEATURES:
        rows.append((f, evaluate((f,))))
    chosen, best = [], crit(base)
    remaining = list(B.ALL_FEATURES)
    while remaining:
        cand = sorted(((crit(r), f, r) for f, r in ((f, evaluate(tuple(chosen + [f]))) for f in remaining)),
                      key=lambda c: c[0])
        if cand[0][0] > best - MIN_GAIN:
            break
        best = cand[0][0]
        chosen.append(cand[0][1])
        remaining.remove(cand[0][1])
        rows.append(("forward: " + "+".join(chosen), cand[0][2]))
    for k in (5, 100):
        rows.append((f"survival only, K={k}", evaluate((), k)))
    return rows, chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--module", action="append", default=[], help="python file that registers estimators")
    ap.add_argument("--features", action="store_true", help="run the baseline feature search")
    ap.add_argument("--no-build", action="store_true", help="use the cache as is")
    ap.add_argument("--out", default=os.path.join(HERE, "REPLAY.md"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    for path in args.module:
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "register_estimators"):
            mod.register_estimators(register)
        for name, fac in getattr(mod, "ESTIMATORS", {}).items():
            register(name, fac)

    t_start = now = time.time()
    recs = list(B.load_cache().values()) if args.no_build else B.build(now=now)
    tasks = B.tasks_from_records(recs, now)
    fl = folds(tasks, args.folds)
    day = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d")  # noqa: E731
    fold_txt = "; ".join(f"fold {i + 1}: {sum(B.fit_eligible(t) for t in tr)} train / {len(te)} test from {day(c)}"
                         for i, (tr, te, c) in enumerate(fl))
    print(fold_txt)
    names = list(ESTIMATORS)
    ref = "baseline"
    md = []
    for mode in MODES:
        preds = run_estimators(fl, mode, names)
        res = {n: breakouts(rows) for n, rows in preds.items()}
        n_asks = len(preds[names[0]])
        n_tasks = len({t["k"] for _, _, _, t in preds[names[0]]})
        print(f"\n== {mode} asks ({n_asks} asks, {n_tasks} tasks)")
        print(f"{'estimator':<20} {'answered':>8} {'median':>7} {'(90% CI)':>13} {'p90':>7} {'w2x':>5} "
              f"{'cov80':>6} {'bias':>6} {'closer than ' + ref:>20}")
        lines = [f"### {mode} asks ({n_asks} asks over {n_tasks} held-out tasks)", "",
                 "| estimator | answered | median miss (90% CI) | p90 miss | within 2x | in 80% range | "
                 f"median bias | closer than {ref} |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for n in names:
            m = res[n]["all"]
            ci = bootstrap_median(preds[n])
            cl = closer_than(preds[n], preds[ref]) if ref in preds and n != ref else None
            ci_txt = f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "n/a"
            cl_txt = f"{cl:.0%}" if cl is not None else "-"
            print(f"{n:<20} {m['answer']:>8.0%} {math.exp(m['med']):>6.2f}x {ci_txt:>13} "
                  f"{math.exp(m['p90']):>6.1f}x {m['w2x']:>5.0%} {m['cov']:>6.0%} "
                  f"{math.exp(m['bias']):>5.2f}x {cl_txt:>20}")
            lines.append(f"| {n} | {m['answer']:.0%} | {math.exp(m['med']):.2f}x ({ci_txt}) | "
                         f"{math.exp(m['p90']):.1f}x | {m['w2x']:.0%} | {m['cov']:.0%} | "
                         f"{math.exp(m['bias']):.2f}x | {cl_txt} |")
        md.append("\n".join(lines) + "\n")
        for n in names:
            md.append(table(f"{mode}: {n} by group", [(g, res[n][g]) for g in ORDER if g in res[n]]) + "\n")
    feat_md = ""
    if args.features:
        rows, chosen = feature_search(fl)
        print("\n== baseline feature search (median miss uniform / late / checkins; 80% range coverage)")
        lines = ["| features | uniform miss | late miss | checkins miss | uniform in range | late in range | "
                 "checkins in range |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, r in rows:
            u, l_, c = r["uniform"], r["late"], r["checkins"]
            print(f"{name:<34} {math.exp(u['med']):.2f}x / {math.exp(l_['med']):.2f}x / {math.exp(c['med']):.2f}x"
                  f"   cov {u['cov']:.0%} / {l_['cov']:.0%} / {c['cov']:.0%}")
            lines.append(f"| {name} | {math.exp(u['med']):.2f}x | {math.exp(l_['med']):.2f}x | "
                         f"{math.exp(c['med']):.2f}x | {u['cov']:.0%} | {l_['cov']:.0%} | {c['cov']:.0%} |")
        lines.append("")
        lines.append("Forward selection kept: " + ("+".join(chosen) if chosen else
                                                   "nothing (no feature beat survival only by the margin)") + ".")
        print(lines[-1])
        feat_md = "\n".join(lines)
    print(f"\nreplay took {time.time() - t_start:.1f}s")
    if not args.no_write:
        write_md(args.out, tasks, fold_txt, md, feat_md)
        print(f"wrote {args.out}")


def corpus_md(tasks):
    from collections import Counter
    c = Counter()
    for t in tasks:
        tag = ("excluded: pure chat (no tool calls)" if t["n"] == 0 else
               "fit: human, done, >=1 tool call" if B.fit_eligible(t) else
               f"excluded: origin {t['o']}" if t["o"] not in B.FIT_ORIGINS else
               f"excluded: {t['end']}" if t["end"] != "done" else "excluded: under 5 s or over 24 h")
        c[(t["h"], tag)] += 1
    lines = ["| harness | category | tasks |", "|---|---|---:|"]
    for (h, tag), n in sorted(c.items(), key=lambda kv: (kv[0][0], -kv[1])):
        lines.append(f"| {h} | {tag} | {n} |")
    fit = [t["wall"] for t in tasks if B.fit_eligible(t)]
    f = B._eta.fmt_dur
    qs = [f(B._eta.quantile(fit, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.99)]
    lines.append("")
    lines.append(f"Fitted task length: p25 {qs[0]}, median {qs[1]}, p75 {qs[2]}, p90 {qs[3]}, p99 {qs[4]}. "
                 f"Over 10 min: {sum(w > 600 for w in fit)}; over 1 h: {sum(w > 3600 for w in fit)}.")
    return "\n".join(lines)


def write_md(path, tasks, fold_txt, md, feat_md):
    body = f"""# History-prior replay

Generated by `python3 verify/replay.py --features` on {datetime.now().strftime('%Y-%m-%d')}.
Numbers only; no transcript text. Rerun to refresh. Reading and weaknesses are at the end.

## Corpus

{corpus_md(tasks)}

## Method

- Tasks come from `skills/eta/scripts/baseline.py` (rules in its docstring).
- Rolling-origin holdout, no leakage: fit-eligible tasks sorted by start time, the newest
  60% cut into {len(md) and fold_txt.count('fold')} chunks; each chunk is scored by estimators fitted only on tasks
  that had finished before the chunk began. {fold_txt}.
- Ask modes. **uniform**: one random moment per task after its first minute (tasks over
  2 min). **late**: one random moment per task after 10 minutes (tasks over 10 min);
  people ask /eta when things feel slow. **checkins**: an ask every 5 minutes once a
  task passes 10 minutes, so long tasks weigh by their duration. **grid**: one ask at
  each of 1, 5, 15, 30, 60 and 120 minutes the task was still running.
- At each ask the state is rebuilt from the cached timeline (elapsed, calls so far,
  open tool and its age, open delegates or background jobs) and each estimator returns
  (low, mid, high) remaining seconds.
- Miss = |ln(mid / actual remaining)| shown as a factor: 1.00x is perfect, 2.00x is off by
  double or half. Predictions and actuals under {FLOOR:.0f} s count as {FLOOR:.0f} s. The 90%
  interval on the median miss resamples whole tasks. "in 80% range" = share of actuals
  inside (low, high), stated by every estimator as its 10th to 90th percentile, so 80%
  is calibrated. Bias = median of mid / actual (above 1 = predicts too long). "closer
  than baseline" = share of asks where that estimator's mid beat the baseline's.

## Results

{chr(10).join(md)}
## Baseline feature search

Same folds. Criterion: summed median log miss over the uniform, late and checkins modes;
a feature must lower it by {MIN_GAIN} to be kept.

{feat_md}

{NOTES}
"""
    with open(path, "w") as f:
        f.write(body)


NOTES = """## Reading (from the 2026-11-25 run; numbers above are authoritative if they drift)

- History alone gives an honest range, not a sharp point. The baseline's stated 80%
  range held 76% to 86% of actual outcomes across the four synthetic ask modes. Its
  midpoint misses by a median 2.4x to 3.2x and is within 2x on only 32% to 39% of asks.
  Remaining time on this machine is extremely spread: at 1 minute elapsed, the
  remaining time of tasks still running goes from about 12 s (p10) to 22 min (p90).
- Against the naive rules: remaining=elapsed has a smaller median miss on early asks
  (uniform 2.54x vs 2.90x, grid 2.73x vs 3.16x), but its fixed 3x band covers only 39%
  to 57% of outcomes and it is 4.2x too long on late asks, where the baseline is closer
  on 84% of asks. The constant median is unusable once a task has run 10 minutes
  (bias 0.05x to 0.19x).
- Weakest spots: tasks that turn out long, and early asks. In the grid mode, asks
  inside tasks that ran 30 min to 2 h miss by 5.4x (range coverage 48%), and tasks over
  2 h by 17x. Nothing at minute 5 tells survival that this task is a two-hour one.
- Bias depends on how people ask. One ask per task at a random moment (uniform, late)
  falls nearer the end, so the baseline reads about 2.3x too long there. Asks spread
  over the task's time (checkins, grid) read 0.8x, slightly short. The survival median
  is the right answer for asks made at random wall-clock moments.
- Features: none earned a place. Harness, model family, tool running and its age,
  delegates or background jobs open, calls so far, and prompt origin each moved the
  median miss by under 0.15x in either direction, and none passed the forward-selection
  margin. The default is elapsed-only survival, shrunk toward a widened
  remaining-proportional-to-elapsed fallback when few past tasks ran that long.
  Codex rows are small (15 to 170 asks) and look different (bias up to 3.8x long),
  so harness is the first feature to retest as Codex history grows.
- For the "agent said 3 days, finished in 30 minutes" complaint: of 2,965 finished,
  tool-using tasks here, the median ran 50 s, p90 11 min, p99 54 min, 6 ran over 2 h
  and the longest 9.1 h. At 10 minutes elapsed the baseline's p90 remaining is about
  43 min. An agent claiming days for a running task is far outside this machine's
  history, so p90 or p99 is a defensible sanity cap for the integrator.

## Weaknesses

- Censoring. Interrupted, steered and cut tasks (163) are dropped instead of being
  treated as "ran at least this long". They lean long, so the prior probably runs
  short at large elapsed times. Delegated, side-chat and scheduled tasks (339) are
  excluded by origin.
- Task boundaries are heuristics. Work that resumes on a notification more than 30
  minutes after the agent stopped, with no tracked launch outstanding, is dropped (6
  tasks). Background work the agent never hears back about is not part of the task.
- Real /eta asks cannot be scored yet. Of 40 recorded ETA asks, 27 reached the
  transcript only after the agent had already ended its turn, so the transcript does
  not show when the person actually asked (suspected: the T3 harness holds messages
  until the turn ends). The 8 that land inside a fit-eligible task are too few to read.
- Few long tasks: 316 fitted tasks over 10 minutes and 27 over 1 hour. Ranges past an
  hour lean on the widened fallback.
- One machine, about five weeks (2026-10-21 to 2026-11-25), mostly one person using
  T3 with Claude Code. The task mix shifts: tasks in later folds ran longer at 5 to 30
  minutes elapsed than the training data said.
- The metric floors both prediction and actual at 30 s, which flatters every
  estimator on very short tasks.
- Agent self-estimates are not in transcripts, so this replay cannot compare history
  with the agent's own guess. That needs the paired predictions ledger in ~/.eta.
"""


if __name__ == "__main__":
    main()
