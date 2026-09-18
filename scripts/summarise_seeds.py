#!/usr/bin/env python3
"""Summarise a multi-seed forecaster study against DMD and persistence.

Reads ``forecast_results_<tag>_s<seed>.json`` files written by
``train_forecaster.py`` (one per training seed) plus the ``dmd_results.json``
written by ``run_dmd.py`` in the same directory, and prints a Markdown table
of mean +/- sample std over seeds: error at fixed horizons, time-averaged
error, and how many seeds beat persistence in every Stokes regime.

Example
-------
    python scripts/summarise_seeds.py -d figures/seeds --tags cond gru ode
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--dir", default="figures/seeds")
    p.add_argument("--tags", nargs="+", default=["cond", "gru", "ode"])
    return p.parse_args()


def fmt(values):
    v = np.asarray(values, dtype=float)
    if v.size == 1:
        return f"`{v[0]:.2e}`"
    return f"`{v.mean():.2e}` ± `{v.std(ddof=1):.1e}`"


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # the table uses ± and –
    d = pathlib.Path(args.dir)
    rows = []
    marks = None
    for tag in args.tags:
        files = sorted(d.glob(f"forecast_results_{tag}_s*.json"),
                       key=lambda f: int(re.search(r"_s(\d+)\.json$", f.name).group(1)))
        if not files:
            raise SystemExit(f"no results for tag {tag!r} in {d}")
        rs = [json.loads(f.read_text()) for f in files]
        marks = marks or list(rs[0]["mean_forecast_rmse_at_t"])
        wins = sum(all(v["forecast_final"] < v["persistence_final"]
                       for v in r["per_stokes"].values()) for r in rs)
        rows.append((f"{rs[0]['model']} (`{tag}`), {len(rs)} seeds",
                     [[r["mean_forecast_rmse_at_t"][m] for r in rs] for m in marks],
                     [r["mean_forecast_rmse_time_avg"] for r in rs],
                     f"{wins}/{len(rs)}"))

    dmd = json.loads((d / "dmd_results.json").read_text())
    per = dmd["per_stokes"]
    dmd_wins = all(v["forecast_final"] < v["persistence_final"]
                   for v in per["per_stokes"].values())
    rows.append((f"DMD, per-Stokes (r={dmd['rank']})",
                 [[per["mean_forecast_rmse_at_t"][m]] for m in marks],
                 [per["mean_forecast_rmse_time_avg"]],
                 "yes" if dmd_wins else "no"))
    rows.append(("persistence",
                 [[dmd["mean_persistence_rmse_at_t"][m]] for m in marks],
                 [dmd["mean_persistence_rmse_time_avg"]], "–"))

    head = ["model"] + [f"t = {m}" for m in marks] + ["time-avg", "beats persistence at every St"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, at_t, tavg, wins in rows:
        print("| " + " | ".join([name] + [fmt(v) for v in at_t] + [fmt(tavg), wins]) + " |")


if __name__ == "__main__":
    main()
