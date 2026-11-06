"""Match fixed-hurdle touchdowns and terminal transitions to empty-ground traces."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from summarize_cat_landing_stance import episode_metrics, stats

SPEED_BAND = 0.15
MATCHING = ("initial_hashes", "initial_goals", "code_commit", "checkpoint_sha256", "protocol_sha256", "gpu")


def closest_speed_match(contacts, speed):
    best = min(contacts, key=lambda x: abs(x["speed"] - speed), default=None)
    if best is None or abs(best["speed"] - speed) > SPEED_BAND:
        return None
    return best


def speed_before(control, frame, hz):
    width = round(0.1 * hz)
    if frame < width:
        return None
    return float(control[frame - width:frame, 1].mean())


def empty_touchdowns(physics, control, side, manifest):
    hz = manifest["physics_hz"]
    control_hz = manifest["control_hz"]
    bw = manifest["robot_mass_kg"] * 9.81
    fz = np.abs(physics[:, side]) / bw
    on = fz > 0.05
    for i in np.flatnonzero(on[1:] & ~on[:-1]) + 1:
        frame = i * control_hz // hz
        start, end = i - round(0.02 * hz), i + round(0.05 * hz)
        if i < round(0.3 * hz) or end > len(on) or frame >= len(control):
            continue
        if on[start:i].any() or on[i:i + round(0.01 * hz)].sum() < round(0.006 * hz):
            continue
        if control[frame, 0] <= 0.2 or not (control[frame - round(0.1 * control_hz):frame, 7] > 0.5).all():
            continue
        yield dict(
            time_s=(i + 1) / hz, speed=speed_before(control, frame, control_hz),
            peak_bw=float(fz[i:end].max()),
            down_m_s=float(-physics[i - round(0.006 * hz):i, 8 + side].mean()),
        )


def stop_window(control, hz):
    moving = control[:, 7] > 0.5
    pre, post = round(0.1 * hz), round(0.5 * hz)
    for i in np.flatnonzero(moving[:-1] & ~moving[1:]) + 1:
        if i < pre or i + post > len(control) or control[i, 0] > 0.2:
            continue
        if not moving[i - pre:i].all() or moving[i:i + 3].any():
            continue
        return dict(
            time_s=(i + 1) / hz, speed=speed_before(control, i, hz),
            gyro_peak=float(control[i:i + post, 2].max()),
        )
    return None


def load(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "traces.npz") as trace:
        return manifest, trace["physics"], trace["control"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route_output", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rows = []
    for obstacle in sorted(args.route_output.glob("hurdle-near-*/manifest.json")):
        a, ap, ac = load(obstacle.parent)
        b, bp, bc = load(args.route_output / obstacle.parent.name.replace("hurdle-near-", "empty-near-"))
        if any(a[k] != b[k] for k in MATCHING):
            raise ValueError(f"Initial conditions differ: {obstacle.parent.name}")
        for slot in range(a["evaluation_episodes"]):
            row = dict(model=a["label"], seed=a["evaluation_seed"], slot=slot,
                       landing_match=False, stop_match=False)
            hurdle_control = ac[ac[:, slot, 6].astype(bool), slot]
            empty_control = bc[bc[:, slot, 6].astype(bool), slot]
            empty_physics = bp[bp[:, slot, 10].astype(bool), slot]
            landing = episode_metrics(ap[:, slot], ac[:, slot], a)
            hz = a["control_hz"]
            if landing.get("landing_time_s") is not None:
                frame = (round(landing["landing_time_s"] * a["physics_hz"]) - 1) * hz // a["physics_hz"]
                speed = speed_before(hurdle_control, frame, hz)
                if speed is not None:
                    row.update(
                        hurdle_landing_speed=speed,
                        hurdle_landing_peak_bw=landing["landing_peak_bw"],
                        hurdle_landing_down_m_s=landing["landing_down_m_s"],
                    )
                    reference = closest_speed_match(
                        empty_touchdowns(empty_physics, empty_control, landing["landing_foot"], b), speed,
                    )
                    if reference is not None:
                        row.update(
                            landing_match=True, empty_landing_speed=reference["speed"],
                            empty_landing_peak_bw=reference["peak_bw"],
                            landing_peak_delta_bw=landing["landing_peak_bw"] - reference["peak_bw"],
                            landing_down_delta_m_s=landing["landing_down_m_s"] - reference["down_m_s"],
                        )
            stop_a, stop_b = stop_window(hurdle_control, hz), stop_window(empty_control, hz)
            if stop_a and stop_b:
                row.update(hurdle_pre_stop_speed=stop_a["speed"], empty_pre_stop_speed=stop_b["speed"])
                if abs(stop_a["speed"] - stop_b["speed"]) <= SPEED_BAND:
                    row.update(stop_match=True, stop_gyro_peak_delta=stop_a["gyro_peak"] - stop_b["gyro_peak"])
            rows.append(row)

    if not rows:
        raise ValueError("No matching route cohorts")
    args.destination.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.destination / "pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for model in sorted({row["model"] for row in rows}):
        summary[model] = {}
        for seed in sorted({row["seed"] for row in rows}):
            group = [row for row in rows if row["model"] == model and row["seed"] == seed]
            summary[model][str(seed)] = dict(
                slots=len(group), landing_matches=sum(r["landing_match"] for r in group),
                stop_matches=sum(r["stop_match"] for r in group),
                effects={k: stats([r[k] for r in group if k in r]) for k in (
                    "landing_peak_delta_bw", "landing_down_delta_m_s", "stop_gyro_peak_delta",
                )},
            )
    matched_models = {}
    for seed in sorted({row["seed"] for row in rows}):
        left, right = (
            {row["slot"]: row for row in rows if row["seed"] == seed and row["model"] == model}
            for model in ("balance50", "balance100")
        )
        eligible = [
            (left[i], right[i]) for i in sorted(left.keys() & right.keys())
            if "hurdle_landing_speed" in left[i] and "hurdle_landing_speed" in right[i]
            and abs(right[i]["hurdle_landing_speed"] - left[i]["hurdle_landing_speed"]) <= SPEED_BAND
        ]
        matched_models[str(seed)] = dict(
            pairs=len(eligible),
            max_abs_speed_delta_m_s=max(
                (abs(b["hurdle_landing_speed"] - a["hurdle_landing_speed"]) for a, b in eligible),
                default=None,
            ),
            delta_100_minus_50={
                name: stats([b[name] - a[name] for a, b in eligible])
                for name in ("hurdle_landing_peak_bw", "hurdle_landing_down_m_s")
            },
            harder_landing_pairs=sum(b["hurdle_landing_peak_bw"] > a["hurdle_landing_peak_bw"] for a, b in eligible),
        )
    (args.destination / "comparison.json").write_text(json.dumps(
        dict(speed_band_m_s=SPEED_BAND, by_model=summary, speed_matched_models=matched_models),
        indent=2, allow_nan=False,
    ) + "\n")


if __name__ == "__main__":
    main()
