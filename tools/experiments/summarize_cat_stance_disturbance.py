"""Summarize paired empty-ground standing disturbance interventions."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from summarize_cat_landing_stance import stats

ARMS = ("full", "no-push", "quiet", "no-push-no-observation-noise", "no-push-no-rfi")
ISOLATED_ARMS = {"no-push-no-observation-noise", "no-push-no-rfi"}
MATCHING = (
    "code_commit", "checkpoint_sha256", "config_sha256", "protocol_sha256",
    "field_cache_identity", "initial_hashes", "initial_goals", "gpu",
)
METRICS = ("gyro_peak", "gyro_rms", "knee_ankle_speed_rms", "knee_ankle_target_step_deg")


def summarize_episode(episode, control, motion, manifest):
    row = dict(success=episode["terminal_v1"]["success"],
               strict_success=episode["terminal_v1"]["strict_success"],
               reason=episode["terminal_v1"]["failure_reason"])
    hz = manifest["control_hz"]
    valid = control[:, 6].astype(bool)
    if int(valid.sum()) != episode["steps"]:
        raise ValueError("Trace mask does not agree with episode length")
    if episode["steps"] != 20 * hz:
        return row
    c, m = control[valid][-hz // 2:], motion[valid][-hz // 2:]
    names = manifest["motion_columns"]
    if m.shape[1] != len(names):
        raise ValueError("Motion schema mismatch")
    gyro = m[:, [names.index(f"gyro_{axis}") for axis in ("roll", "pitch", "yaw")]]
    phase = m[:, [names.index("phase_left"), names.index("phase_right")]]
    angle_speed = np.linalg.norm(gyro, axis=1)
    n = len(manifest["joint_names"])
    legs = [i for i, name in enumerate(manifest["joint_names"]) if "knee" in name or "ankle" in name]
    targets = c[:, 8 + 2 * n:8 + 3 * n][:, legs]
    velocities = c[:, 8 + n:8 + 2 * n][:, legs]
    row.update(
        gyro_peak=float(angle_speed.max()),
        gyro_rms=float(np.sqrt(np.mean(angle_speed**2))),
        angular_exceeded=bool((angle_speed >= 0.2).any()),
        static_command=bool((m[:, names.index("move")] < 0.5).all()),
        static_phase=bool((np.ptp(phase, axis=0) < 1e-6).all()),
        double_support=bool((c[:, 4:6] > 0.5).all()),
        knee_ankle_speed_rms=float(np.sqrt(np.mean(velocities**2))),
        knee_ankle_target_step_deg=float(np.rad2deg(np.sqrt(np.mean(np.diff(targets, axis=0)**2)))),
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifests, groups = {}, defaultdict(dict)
    for arm in ARMS:
        for directory in sorted((args.root / arm).glob("*/")):
            if not (directory / "evaluation.json").is_file():
                continue
            manifest = json.loads((directory / "manifest.json").read_text())
            key = (manifest["label"], manifest["evaluation_seed"], arm)
            if manifest["disturbance"] != arm or manifest["route"] != "empty-stand" or key in manifests:
                raise ValueError(f"Unexpected cohort {key}")
            manifests[key] = manifest
            with np.load(directory / "traces.npz") as arrays:
                control, motion = arrays["control"], arrays["motion"]
            episodes = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
            if sorted(e["slot"] for e in episodes) != list(range(manifest["evaluation_episodes"])):
                raise ValueError(f"Unexpected slots in {directory}")
            for e in episodes:
                slot = e["slot"]
                row = summarize_episode(e, control[:, slot], motion[:, slot], manifest)
                row.update(label=key[0], seed=key[1], arm=arm, slot=slot,
                           code_commit=manifest["code_commit"], checkpoint_sha256=manifest["checkpoint_sha256"])
                groups[key][slot] = row
    rows = [r for group in groups.values() for r in group.values()]
    if not rows:
        raise ValueError("No completed disturbance cohorts")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.root / "episode-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    effects, cohorts = [], []
    for (label, seed, arm), group in sorted(groups.items()):
        rows = list(group.values())
        cohorts.append(dict(
            model=label, seed=seed, arm=arm, episodes=len(rows),
            success=sum(r["success"] for r in rows),
            strict_success=sum(r["strict_success"] for r in rows),
            stationary_angular_exceeded=sum(all(r.get(k) for k in (
                "static_command", "static_phase", "double_support", "angular_exceeded")) for r in rows),
            metrics={m: stats([r[m] for r in rows if m in r]) for m in METRICS},
        ))
        baseline_arm = "no-push" if arm in ISOLATED_ARMS else "full"
        baseline = (label, seed, baseline_arm)
        candidate = (label, seed, arm)
        if arm == "full" or baseline not in groups:
            continue
        if any(manifests[baseline][k] != manifests[candidate][k] for k in MATCHING):
            raise ValueError(f"Unmatched disturbance comparison {candidate}")
        left, right = groups[baseline], group
        if left.keys() != right.keys():
            raise ValueError(f"Unmatched disturbance slots {candidate}")
        effects.append(dict(
            model=label, seed=seed, arm=arm, baseline_arm=baseline_arm, pairs=len(group),
            delta_success_count=sum(right[i]["success"] - left[i]["success"] for i in left),
            delta_candidate_minus_baseline={
                m: stats([right[i][m] - left[i][m] for i in left if m in right[i] and m in left[i]])
                for m in METRICS
            },
        ))
    (args.root / "comparison.json").write_text(
        json.dumps(dict(cohorts=cohorts, effects=effects), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
