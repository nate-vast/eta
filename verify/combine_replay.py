#!/usr/bin/env python3
"""Score eta.combine() with verify/replay.py's folds and ask modes, and fit its constants.

    python3 verify/combine_replay.py            replay table (writes verify/COMBINE.md)
    python3 verify/combine_replay.py --fit      also grid-search W_WAITS, USE_TAIL and WIDTH

Also usable as a plug-in: python3 verify/replay.py --module verify/combine_replay.py

At each ask the held-out task's own transcript is re-read as of the cut with waits.forecast()
(waits history filtered to runs that had ended by then) and the prior is fitted only on tasks
that finished before the fold began. Known small leaks, all in waits.py's live probes: a Claude
sub-agent's .meta.json "finished" flag and background output files are read as they are now.

The agent's step count cannot be replayed without a model (verify/llm_replay.py does that).
The "oracle steps" rows use the true number of tool calls the task still made, so they show
the best a perfect count could do, and "steps x2 noise" multiplies it by a random factor
with a log-sd of ln 2.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "eta", "scripts"))
sys.path.insert(0, HERE)
import eta as E  # noqa: E402
import baseline as B  # noqa: E402
import waits as W  # noqa: E402
import replay as R  # noqa: E402

_HIST = None
_FC = {}


def history():
    global _HIST
    if _HIST is None:
        W.mine(budget_s=None)
        _HIST = W.load_history(budget_s=0.0)
    return _HIST


def waits_at(state):
    """waits.forecast() for the held-out task as of the cut, or None when nothing was open."""
    t = state.get("_task")
    if not t or not t.get("f"):
        return None
    e = state["elapsed_s"]
    key = (t["k"], round(e, 3))
    if key not in _FC:
        fc = None
        if state.get("delegates_open") or (state.get("running_s") or 0) >= W.FRESH:
            try:
                fc = W.forecast(t["f"], t["h"], t["t0"] + e, history())
            except Exception:   # an unreadable transcript is scored as "nothing open"
                fc = None
            if fc and not fc.get("wait"):
                fc = None
        _FC[key] = fc
    return _FC[key]


def prior_fn(train):
    base = B.Baseline(train)

    def prior(state):
        (p10, p50, p90, cap), n, _ = base.quantiles(state, (0.1, 0.5, 0.9, E.CAP_Q))
        return {"p10": p10, "p50": p50, "p90": p90, "cap": cap, "n": n}
    return prior, train


def oracle_steps(state):
    t = state["_task"]
    return sum(1 for s, _, _ in t["c"] if s > state["elapsed_s"])


def make(kind, **kw):
    def factory(train):
        prior, tasks = prior_fn(train)
        hp = {}

        def pace_for(state):
            p, _, _ = E.own_pace(state["_task"], state["elapsed_s"])
            if p is None:
                h = state["harness"]
                if h not in hp:
                    hp[h] = E.history_pace(tasks, h)[0]
                p = hp[h]
            return p

        rng = random.Random(7)

        def est(state):
            st = dict(state, prior=prior(state))
            fc = waits_at(state)
            steps = pace = None
            if kind.startswith("oracle"):
                steps, pace = oracle_steps(state), pace_for(state)
                if kind == "oracle-noisy":
                    steps = steps * math.exp(rng.gauss(0, math.log(2)))
            lo, mid, hi, _ = E.combine(st, fc, steps, pace, **kw)
            return lo, mid, hi
        return est
    return factory


ESTIMATORS = {
    "shipped, no step count": make("nosteps"),
    "history+waits+tail (w .5)": make("nosteps", use_tail=True, w_waits=0.5, width=1.0),
    "history+waits, no tail (w .5)": make("nosteps", use_tail=False, w_waits=0.5, width=1.0),
    "waits+tail, no history": make("nosteps", use_tail=True, w_waits=1.0, width=1.0),
    "oracle steps (W_STEPS)": make("oracle"),
    "oracle steps, no history": make("oracle", w_steps=1.0),
    "oracle steps x2 noise": make("oracle-noisy"),
}


def register_estimators(register):
    for name, fac in ESTIMATORS.items():
        register(name, fac)


# ---------------------------------------------------------------- fitting

def collect(fold_list, modes):
    """Per ask: (state with prior, waits forecast, oracle steps, noisy steps, pace, actual, mode)."""
    rows = []
    rng = random.Random(3)
    for train, test, _ in fold_list:
        prior, tasks = prior_fn(train)
        hp = {}
        for mode in modes:
            for t, e in R.make_cuts(test, mode):
                st = dict(B.state_at(t, e), _task=t)
                p, _, _ = E.own_pace(t, e)
                if p is None:
                    hp.setdefault(t["h"], E.history_pace(tasks, t["h"])[0])
                    p = hp[t["h"]]
                n = oracle_steps(st)
                rows.append((dict(st, prior=prior(st)), waits_at(st), n, n * math.exp(rng.gauss(0, math.log(2))),
                             p, t["wall"] - e, mode))
    return rows


def evaluate(rows, steps=None, **kw):
    """steps: None (no count), "oracle" or "noisy". kw goes to eta.combine."""
    scored = []
    for st, fc, n, noisy, pace, actual, _ in rows:
        s = n if steps == "oracle" else noisy if steps == "noisy" else None
        lo, mid, hi, _ = E.combine(st, fc, s, pace if s is not None else None, **kw)
        scored.append(((lo, mid, hi), actual))
    return R.score(scored)


def _line(label, m):
    return (f"| {label} | {math.exp(m['med']):.2f}x | {m['w2x']:.0%} | {m['cov']:.0%} | "
            f"{math.exp(m['bias']):.2f}x |")


def fit(rows, out):
    """1. W_WAITS and USE_TAIL: pooled median log miss on asks with an open wait (no count).
    2. WIDTH: 80% coverage of the step-count blend, using the true remaining call count with
       random 2x noise as a stand-in for an agent's count (an assumption, not a measurement).
    3. For W_STEPS, how oracle and noisy counts do at each weight (it stays a chosen number)."""
    fit_modes = ("uniform", "late", "checkins")
    wr = [r for r in rows if r[1] is not None and r[6] in fit_modes]
    hdr = ["| setting | median miss | within 2x | in 80% range | median bias |", "|---|---:|---:|---:|---:|"]
    out.append(f"\n### 1. Waits without a step count ({len(wr)} asks with an open wait, uniform+late+checkins)\n")
    out += hdr
    grid = []
    for tail in (True, False):
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            m = evaluate(wr, w_waits=w, use_tail=tail, width=1.0)
            grid.append((m["med"], w, tail))
            out.append(_line(f"w_waits {w:.2f}, {'with' if tail else 'no'} tail", m))
    _, w, tail = min(grid)
    out.append(f"\nChosen by lowest median miss: W_WAITS = {w}, USE_TAIL = {tail}.")
    sr = [r for r in rows if r[6] in fit_modes]
    out.append(f"\n### 2. Range width for the step-count blend ({len(sr)} asks, noisy oracle counts, W_STEPS = {E.W_STEPS})\n")
    out += hdr
    widths = []
    for k in range(8, 21):
        m = evaluate(sr, "noisy", width=k / 10)
        widths.append((abs(m["cov"] - 0.80), k / 10))
        out.append(_line(f"width {k / 10:.1f}", m))
    _, width = min(widths)
    out.append(f"\nChosen: WIDTH = {width}.")
    out.append(f"\n### 3. Step-count weight ({len(sr)} asks, WIDTH = {width})\n")
    out += hdr
    out.append(_line("history alone (baseline)", evaluate(sr)))
    for ws in (0.25, 0.5, 0.75, 1.0):
        out.append(_line(f"oracle count, w_steps {ws:.2f}", evaluate(sr, "oracle", w_steps=ws, width=width)))
        out.append(_line(f"oracle count x2 noise, w_steps {ws:.2f}", evaluate(sr, "noisy", w_steps=ws, width=width)))
    print("\n".join(out))
    return w, tail, width


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    now = time.time()
    recs = list(B.load_cache().values()) or B.build(now=now)
    tasks = B.tasks_from_records(recs, now)
    fl = R.folds(tasks)
    history()
    fit_lines = []
    if args.fit:
        rows = collect(fl, ("uniform", "late", "checkins"))
        fit(rows, fit_lines)
    R.ESTIMATORS.clear()
    R.register("baseline", lambda train: B.Baseline(train).estimate)
    register_estimators(R.register)
    lines = []
    for mode in R.MODES:
        preds = R.run_estimators(fl, mode, list(R.ESTIMATORS))
        ref = preds["baseline"]
        n_wait = sum(1 for _, _, st, _ in ref if waits_at(st) is not None)
        lines.append(f"\n### {mode} asks ({len(ref)} asks, {n_wait} with an open wait)\n")
        lines.append("| estimator | asks | median miss | within 2x | in 80% range | median bias | "
                     "closer than baseline | open-wait asks: miss | open-wait asks: in range |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, rows in preds.items():
            m = R.score([(p, a) for p, a, _, _ in rows])
            wrows = [(p, a) for p, a, st, _ in rows if waits_at(st) is not None]
            wm = R.score(wrows) if wrows else {"answer": 0}
            cl = R.closer_than(rows, ref) if name != "baseline" else None
            wtxt = (f"{math.exp(wm['med']):.2f}x | {wm['cov']:.0%}" if wm.get("answer") else "- | -")
            lines.append(f"| {name} | {m['n']} | {math.exp(m['med']):.2f}x | {m['w2x']:.0%} | {m['cov']:.0%} | "
                         f"{math.exp(m['bias']):.2f}x | {'-' if cl is None else f'{cl:.0%}'} | {wtxt} |")
        print("\n".join(lines[-len(preds) - 3:]))
    print(f"\ntook {time.time() - t0:.0f}s")
    if not args.no_write:
        with open(os.path.join(HERE, "COMBINE_TABLES.md"), "w") as f:
            f.write("# Combiner replay tables\n\nGenerated by `python3 verify/combine_replay.py --fit`. "
                    "Numbers only. See COMBINE.md for the reading.\n\n## Fit\n" + "\n".join(fit_lines)
                    + "\n\n## Replay by ask mode\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
