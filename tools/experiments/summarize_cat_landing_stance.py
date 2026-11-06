"""Summarize matched fixed-hurdle traces by episode, model, and seed."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "landing_peak_bw", "landing_down_m_s", "landing_pitch_excursion_deg",
    "landing_knee_ankle_velocity_rms", "landing_knee_ankle_target_step_deg",
    "stance_joint_velocity_rms", "stance_target_step_rms_deg",
    "stance_foot_slip_m_s", "stance_torso_pitch_rms_deg",
)


def episode_metrics(physics, control, manifest):
    valid = control[:, 6].astype(bool)
    c = control[valid]
    n = len(manifest["joint_names"])
    legs = [
        i for i, name in enumerate(manifest["joint_names"])
        if "knee" in name or "ankle" in name
    ]
    if not len(c) or not legs:
        raise ValueError("Expected control observations and named knee/ankle channels")
    hz = manifest["physics_hz"]
    bw = manifest["robot_mass_kg"] * 9.81
    barriers = [
        b for b in manifest["boxes"]
        if b["center"][2] - b["half_size"][2] <= 1e-6
        and b["center"][2] + b["half_size"][2] < 0.5
    ]
    if len(barriers) != 1:
        raise ValueError("Probe requires exactly one ground hurdle")
    bar = barriers[0]
    front = bar["center"][0] - bar["half_size"][0]
    back = bar["center"][0] + bar["half_size"][0]
    lateral = bar["center"][1]
    half_width = bar["half_size"][1]
    p = physics[physics[:, 10].astype(bool)]
    if len(p) != len(c) * hz // manifest["control_hz"]:
        raise ValueError(f"Misaligned telemetry: {len(p)} physics vs {len(c)} control")
    row = dict.fromkeys(METRICS)
    foot = p[:, 2:8].reshape(-1, 2, 3)
    fz = np.abs(p[:, :2]) / bw
    touchdown = []
    for side in range(2):
        on = fz[:, side] > 0.05
        candidates = np.flatnonzero(on[1:] & ~on[:-1]) + 1
        for i in candidates:
            pre, end = i - round(0.02 * hz), i + round(0.05 * hz)
            if pre < 0 or end > len(on) or on[pre:i].any():
                continue
            if on[i:i + round(0.01 * hz)].sum() < round(0.006 * hz):
                continue
            earlier_contact = np.flatnonzero(on[:pre])
            if not len(earlier_contact):
                continue
            swing = foot[earlier_contact[-1]:i + 1, side]
            above_hurdle = (
                (swing[:, 0] >= front) & (swing[:, 0] <= back)
                & (np.abs(swing[:, 1] - lateral) <= half_width)
            )
            if not (swing[0, 0] < front and swing[-1, 0] > back and above_hurdle.any()):
                continue
            touchdown.append((i, side, end))
            break
    if touchdown:
        i, side, end = min(touchdown)
        control_index = i * manifest["control_hz"] // hz
        before = max(0, control_index - 1)
        after = min(len(c), control_index + round(0.25 * manifest["control_hz"]))
        side_joints = [
            j for j in legs if manifest["joint_names"][j].startswith(("left_", "right_")[side])
        ]
        target_change = c[control_index, 8 + 2 * n:8 + 3 * n] - c[before, 8 + 2 * n:8 + 3 * n]
        measured_velocity = c[before, 8 + n:8 + 2 * n]
        row.update(
            landing_knee_ankle_velocity_rms=float(np.sqrt(np.mean(measured_velocity[side_joints]**2))),
            landing_knee_ankle_target_step_deg=float(np.rad2deg(np.sqrt(np.mean(target_change[side_joints]**2)))),
            landing_time_s=(i + 1) / hz,
            landing_peak_bw=float(fz[i:end, side].max()),
            landing_down_m_s=float(-p[i - round(0.006 * hz):i, 8 + side].mean()),
            landing_pitch_excursion_deg=float(np.rad2deg(np.abs(c[control_index:after, 3] - c[before, 3]).max())),
        )
    # Late stance requires one continuous second inside the goal radius,
    # with both feet supported. Select the latest qualifying window.
    stance = (c[:, 0] <= 0.1) & c[:, 4].astype(bool) & c[:, 5].astype(bool)
    window = manifest["control_hz"]
    indices = np.flatnonzero(
        np.convolve(stance.astype(int), np.ones(window, dtype=int), mode="valid") == window
    ) if len(c) >= window else []
    if len(indices):
        start = int(indices[-1])
        data = c[start:start + window]
        targets = data[:, 8 + 2 * n:8 + 3 * n][:, legs]
        velocity = data[:, 8 + n:8 + 2 * n][:, legs]
        row.update(
            stance_window_end_s=(start + window) / window,
            stance_joint_velocity_rms=float(np.sqrt(np.mean(velocity**2))),
            stance_target_step_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.diff(targets, axis=0)**2)))),
            stance_foot_slip_m_s=float(data[:, -2:].mean()),
            stance_torso_pitch_rms_deg=float(np.rad2deg(data[:, 3].std())),
        )
    return row


def stats(values):
    if not values:
        return {"n": 0, "median": None, "p95": None}
    return dict(n=len(values), median=float(np.median(values)), p95=float(np.quantile(values, 0.95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    by_model = {}
    by_seed = {}
    for path in sorted(args.root.glob("*/traces.npz")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        with np.load(path) as traces:
            physics, control = traces["physics"], traces["control"]
        episodes = [json.loads(line) for line in (path.parent / "episodes.jsonl").read_text().splitlines()]
        label, seed = manifest["label"], manifest["evaluation_seed"]
        pair_key = (
            manifest["initial_hashes"], manifest["protocol_sha256"],
            manifest["config_sha256"], manifest["environment_config_sha256"], manifest["code_commit"],
        )
        if by_seed.setdefault(seed, pair_key) != pair_key:
            raise ValueError(f"Seed {seed} has mismatched initial conditions or code")
        for episode in episodes:
            slot = episode["slot"]
            row = episode_metrics(physics[:, slot], control[:, slot], manifest)
            row.update(
                label=label, evaluation_seed=seed, slot=slot,
                code_commit=manifest["code_commit"], config_sha256=manifest["config_sha256"],
                checkpoint_sha256=manifest["checkpoint_sha256"],
                **episode["terminal_v1"],
            )
            by_model.setdefault(label, {})[(seed, slot)] = row
    if set(by_model) != {"balance50", "balance100"}:
        raise ValueError("Expected results from both policies")
    baseline, candidate = by_model["balance50"], by_model["balance100"]
    if baseline.keys() != candidate.keys():
        raise ValueError("The two runs do not cover the same seeds and slots")
    rows = [row for model in by_model.values() for row in model.values()]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.root / "episode-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {}
    for seed in sorted(by_seed) + ["pooled"]:
        pairs = [(baseline[k], candidate[k]) for k in baseline if seed == "pooled" or k[0] == seed]
        report[str(seed)] = {
            "pairs": len(pairs),
            "balance50": {
                "success": sum(a["success"] for a, _ in pairs),
                "strict_success": sum(a["strict_success"] for a, _ in pairs),
            },
            "balance100": {
                "success": sum(b["success"] for _, b in pairs),
                "strict_success": sum(b["strict_success"] for _, b in pairs),
            },
            "paired_metrics": {},
        }
        for metric in METRICS:
            eligible = [(a[metric], b[metric]) for a, b in pairs if a[metric] is not None and b[metric] is not None]
            report[str(seed)]["paired_metrics"][metric] = dict(
                balance50=stats([a for a, _ in eligible]),
                balance100=stats([b for _, b in eligible]),
                delta_100_minus_50=stats([b - a for a, b in eligible]),
                available_50=sum(a[metric] is not None for a, _ in pairs),
                available_100=sum(b[metric] is not None for _, b in pairs),
            )
    (args.root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
