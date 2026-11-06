"""Measure standing-filter effects against a matched frozen-policy control."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from summarize_cat_landing_stance import episode_metrics, stats
from summarize_cat_stance_disturbance import summarize_episode

MATCHING = (
    "code_commit", "checkpoint_sha256", "config_sha256", "protocol_sha256",
    "field_cache_identity", "initial_hashes", "initial_goals", "gpu",
)
METRICS = (
    "gyro_peak", "gyro_rms", "knee_ankle_target_step_deg", "landing_peak_bw",
    "first_goal_seconds", "stationary_gyro_power_share_ge_10hz",
)


def load_cohort(directory):
    m = json.loads((directory / "manifest.json").read_text())
    episodes = [json.loads(x) for x in (directory / "episodes.jsonl").read_text().splitlines()]
    if sorted(e["slot"] for e in episodes) != list(range(m["evaluation_episodes"])):
        raise ValueError(f"Missing or duplicate slots: {directory}")
    mode = m["stationary_velocity"]
    applied = (
        json.loads((directory / "velocity-filter.json").read_text())["applied_control_ticks"]
        if mode == "ema" else [0] * len(episodes)
    )
    if m["disturbance"] != "full" or m["suppress_push"] or len(applied) != len(episodes):
        raise ValueError(f"Invalid full-disturbance comparison: {directory}")
    with np.load(directory / "traces.npz") as traces:
        control, motion, physics = (traces[k] for k in ("control", "motion", "physics"))
    rows = {}
    for e in episodes:
        slot = e["slot"]
        if not 0 <= applied[slot] <= e["steps"]:
            raise ValueError(f"Invalid filter coverage: {directory}, {slot}")
        c, a, p = control[:, slot], motion[:, slot], physics[:, slot]
        row = summarize_episode(e, c, a, m)
        row.update(
            code_commit=m["code_commit"], route=m["route"], mode=mode,
            label=m["label"], seed=m["evaluation_seed"], slot=slot, applied_ticks=applied[slot],
            landing_peak_bw=episode_metrics(p, c, m).get("landing_peak_bw"),
            first_goal_seconds=e.get("first_goal_seconds"),
        )
        rows[slot] = row
    return m, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    batches = defaultdict(dict)
    for path in sorted(args.root.glob("*/*/evaluation.json")):
        m, rows = load_cohort(path.parent)
        key = (m["route"], m["label"], m["evaluation_seed"])
        mode = m["stationary_velocity"]
        if mode in batches[key]:
            raise ValueError(f"Duplicate cohort {key}, {mode}")
        batches[key][mode] = m, rows
    if not batches:
        raise ValueError("No completed cohorts")
    cohorts, comparisons, episodes = [], [], []
    for (route, label, seed), modes in sorted(batches.items()):
        for mode, (_, items) in sorted(modes.items()):
            rows = list(items.values())
            episodes.extend(rows)
            cohorts.append(dict(
                route=route, label=label, seed=seed, mode=mode, episodes=len(rows),
                success=sum(r["success"] for r in rows),
                strict_success=sum(r["strict_success"] for r in rows),
                applied_episodes=sum(r["applied_ticks"] > 0 for r in rows),
                falls=sum(r["reason"] == "fall" for r in rows),
                prohibited_contacts=sum(r["reason"] == "prohibited_contact" for r in rows),
                metrics={k: stats([r[k] for r in rows if r.get(k) is not None]) for k in METRICS},
            ))
        if set(modes) != {"off", "ema"}:
            continue
        a, left = modes["off"]
        b, right = modes["ema"]
        if left.keys() != right.keys() or any(a[k] != b[k] for k in MATCHING):
            raise ValueError(f"Filter and control provenance do not match: {route}, {label}, {seed}")
        comparisons.append(dict(
            route=route, label=label, seed=seed, pairs=len(left),
            strict_success_delta=sum(right[i]["strict_success"] - left[i]["strict_success"] for i in left),
            delta_ema_minus_off={k: stats([
                right[i][k] - left[i][k] for i in left
                if right[i].get(k) is not None and left[i].get(k) is not None
            ]) for k in METRICS},
        ))
    fields = list(dict.fromkeys(k for row in episodes for k in row))
    with (args.root / "episode-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)
    (args.root / "comparison.json").write_text(
        json.dumps(dict(cohorts=cohorts, comparisons=comparisons), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
