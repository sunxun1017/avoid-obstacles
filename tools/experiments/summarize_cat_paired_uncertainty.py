"""Summarize conditional paired effects for the archived CAT diagnostics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 20261113
REPLICATES = 10_000


def paired_effect(left, right, field, *, seed_key):
    def indexed(rows):
        records = {(row[seed_key], row["slot"]): row for row in rows}
        if len(records) != len(rows):
            raise ValueError("Duplicate seed/slot within a comparison arm")
        return records

    a, b = indexed(left), indexed(right)
    if not a or a.keys() != b.keys():
        raise ValueError("Expected the same initial seed/slot pairs")
    deltas = defaultdict(list)
    for seed, slot in sorted(a):
        x, y = a[seed, slot].get(field), b[seed, slot].get(field)
        if x not in (None, "") and y not in (None, ""):
            deltas[seed].append(float(y) - float(x))
    if not deltas:
        return dict(pairs=0, median=None, interval_95=None, by_seed={})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = [
        rng.choice(values, size=(REPLICATES, len(values)), replace=True)
        for values in deltas.values()
    ]
    estimate = np.median(np.concatenate(samples, axis=1), axis=1)
    return dict(
        pairs=sum(map(len, deltas.values())),
        median=float(np.median(np.concatenate(list(deltas.values())))),
        interval_95=np.quantile(estimate, [0.025, 0.975]).tolist(),
        by_seed={k: dict(pairs=len(v), median=float(np.median(v))) for k, v in deltas.items()},
    )


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = args.outputs
    route_rows = read_csv(root / "stance-route-v2/episode-metrics.csv")
    stance_rows = [
        row for batch in ("stance-factor-v1", "stance-factor-seed11")
        for row in read_csv(root / batch / "episode-metrics.csv")
    ]
    result = dict(
        bootstrap_seed=BOOTSTRAP_SEED, replicates=REPLICATES,
        interpretation="Percentile intervals from paired slot resampling within fixed evaluation seeds. "
                       "They do not measure variation over independently trained models.",
        landing={}, stance={},
    )
    for route in ("hurdle-near", "hurdle-far"):
        a, b = (
            [row for row in route_rows if row["route"] == route and row["label"] == label]
            for label in ("balance50", "balance100")
        )
        result["landing"][route] = {
            field: paired_effect(a, b, field, seed_key="evaluation_seed")
            for field in ("landing_peak_bw", "landing_down_m_s", "landing_knee_ankle_target_step_deg")
        }
    for label in ("balance50", "balance100"):
        a = [row for row in stance_rows if row["label"] == label and row["arm"] == "no-push"]
        result["stance"][label] = {}
        for arm in ("no-push-no-observation-noise", "no-push-no-rfi"):
            b = [row for row in stance_rows if row["label"] == label and row["arm"] == arm]
            result["stance"][label][arm] = {
                field: paired_effect(a, b, field, seed_key="seed")
                for field in ("gyro_peak", "knee_ankle_target_step_deg")
            }
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
