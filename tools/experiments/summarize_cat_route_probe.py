"""Report CAT route comparisons while retaining per-seed and per-slot data."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from embodied_learning.evaluation.cat_route_probe import first_stop_after_landing
from summarize_cat_landing_stance import episode_metrics, stats


def route_metrics(control, motion, names, hz, landing_time):
    active = control[:, 6].astype(bool)
    c, m = control[active], motion[active]
    move = m[:, names.index("move")] > 0.5
    root_x = m[:, names.index("root_x")]
    omega = m[:, :3]
    result = {}
    if landing_time is not None:
        index = min(len(c) - 1, max(0, int(np.ceil(landing_time * hz)) - 1))
        end = index + hz // 2
        if end < len(c):
            result["post_landing_progress_0p5_m"] = float(root_x[end] - root_x[index])
            result["post_landing_moving_fraction_0p5"] = float(move[index + 1:end + 1].mean())
        stop = first_stop_after_landing(move, landing_time, hz)
        if stop is not None:
            result["post_landing_stop_delay_s"] = stop - landing_time

    if len(c) >= hz // 2:
        tail = slice(-(hz // 2), None)
        for axis, values in zip(("roll", "pitch", "yaw"), omega[tail].T):
            result[f"tail_gyro_{axis}_rms"] = float(np.sqrt(np.mean(values**2)))
        result["tail_moving_fraction"] = float(move[tail].mean())
        result["tail_gyro_norm_peak"] = float(np.linalg.norm(omega[tail], axis=1).max())
        result["tail_pelvis_pitch_rms_deg"] = float(np.rad2deg(m[tail, names.index("pelvis_pitch")].std()))
        result["tail_torso_pitch_rms_deg"] = float(np.rad2deg(m[tail, names.index("torso_pitch")].std()))
        result["tail_double_support_fraction"] = float((c[tail, 4:6] > 0.5).all(axis=1).mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    groups, manifests = defaultdict(list), {}
    for trace in sorted(args.root.glob("*/traces.npz")):
        directory = trace.parent
        manifest = json.loads((directory / "manifest.json").read_text())
        key = (manifest["route"], manifest["label"], manifest["evaluation_seed"])
        if key in manifests:
            raise ValueError(f"Duplicate cohort {key}")
        manifests[key] = manifest
        with np.load(trace) as data:
            physics, control, motion = data["physics"], data["control"], data["motion"]
        names = manifest["motion_columns"]
        if motion.shape[:2] != control.shape[:2] or motion.shape[2] != len(names):
            raise ValueError(f"Trace column mismatch in {directory}")
        episodes = [json.loads(s) for s in (directory / "episodes.jsonl").read_text().splitlines()]
        if sorted(e["slot"] for e in episodes) != list(range(manifest["evaluation_episodes"])):
            raise ValueError(f"Missing or duplicate slots in {directory}")
        for e in episodes:
            slot = e["slot"]
            p, c, m = physics[:, slot], control[:, slot], motion[:, slot]
            row = episode_metrics(p, c, manifest)
            row.update(route_metrics(c, m, names, manifest["control_hz"], row.get("landing_time_s")))
            if e["steps"] < 20 * manifest["control_hz"]:
                row = {k: v for k, v in row.items() if not k.startswith("tail_")}
            row.update(
                run_id=directory.name, route=key[0], label=key[1], evaluation_seed=key[2],
                slot=slot, code_commit=manifest["code_commit"],
                config_sha256=manifest["config_sha256"], checkpoint_sha256=manifest["checkpoint_sha256"],
                **e["terminal_v1"],
            )
            groups[key].append(row)

    if not groups:
        raise ValueError("No complete cohorts")
    matching = ("initial_hashes", "initial_goals", "field_cache_identity", "protocol_sha256", "config_sha256", "code_commit")
    for (route, label, seed), manifest in manifests.items():
        other = manifests.get((route, "balance100", seed))
        if label == "balance50" and other and any(manifest[k] != other[k] for k in matching):
            raise ValueError(f"Models do not match initial conditions for {route}, {seed}")

    rows = [row for group in groups.values() for row in group]
    fixed = {"run_id", "route", "label", "evaluation_seed", "slot", "code_commit",
             "config_sha256", "checkpoint_sha256", "success", "strict_success", "failure_reason"}
    metrics = sorted(set().union(*(row.keys() for row in rows)) - fixed)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.root / "episode-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = []
    for key, group in sorted(groups.items()):
        report.append(dict(
            route=key[0], label=key[1], seed=key[2], episodes=len(group),
            success=sum(row["success"] for row in group),
            strict_success=sum(row["strict_success"] for row in group),
            metrics={k: stats([row[k] for row in group if row.get(k) is not None]) for k in metrics},
        ))
    (args.root / "route-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    comparisons = []
    for left in sorted(groups):
        route, label, seed = left
        candidates = []
        if label == "balance50":
            candidates.append((route, "balance100", seed))
        if route == "hurdle-near":
            candidates.append(("hurdle-far", label, seed))
            candidates.append(("empty-near", label, seed))
        if route == "empty-near":
            candidates.append(("empty-stand", label, seed))
        for right in candidates:
            if right not in groups:
                continue
            a, b = manifests[left], manifests[right]
            if left[0] == "hurdle-near" and right[0] == "hurdle-far":
                if a["initial_hashes"] != b["initial_hashes"]:
                    raise ValueError(f"Near/far initial conditions differ: {label}, {seed}")
                near, far = np.asarray(a["initial_goals"]), np.asarray(b["initial_goals"])
                if not np.array_equal(near[:, 1:], far[:, 1:]) or not (far[:, 0] > near[:, 0]).all():
                    raise ValueError(f"Near/far target assignment differs in lane: {label}, {seed}")
            left_rows = {r["slot"]: r for r in groups[left]}
            right_rows = {r["slot"]: r for r in groups[right]}
            if left_rows.keys() != right_rows.keys():
                raise ValueError(f"Unmatched slots: {left} vs {right}")
            differences = {
                metric: stats([
                    right_rows[slot][metric] - left_rows[slot][metric]
                    for slot in left_rows
                    if left_rows[slot].get(metric) is not None and right_rows[slot].get(metric) is not None
                ])
                for metric in metrics
            }
            comparisons.append(dict(
                baseline=list(left), candidate=list(right), pairs=len(left_rows),
                initial_hash_match=a["initial_hashes"] == b["initial_hashes"],
                delta_candidate_minus_baseline=differences,
            ))
    (args.root / "route-comparisons.json").write_text(
        json.dumps(comparisons, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
