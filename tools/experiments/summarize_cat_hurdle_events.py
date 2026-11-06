"""Compare event-timed hurdle impulses to a matched no-impulse rollout."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from summarize_cat_landing_stance import episode_metrics, stats

STAGES = ("swing", "touchdown", "after")
MATCHING = (
    "code_commit", "checkpoint_sha256", "config_sha256", "protocol_sha256",
    "field_cache_identity", "initial_hashes", "initial_goals", "gpu",
)


def recovery_seconds(pushed, reference, hz):
    """Time to a completed, 300 ms walking window within reference tolerance."""
    width = round(0.3 * hz)
    limit = min(len(pushed), len(reference), 2 * hz)
    for end in range(width, limit + 1):
        x, y = pushed[end - width:end], reference[end - width:end]
        if not (x[:, 2].all() and y[:, 2].all() and x[:, 0].mean() > 0):
            continue
        if abs(x[:, 0].mean() - y[:, 0].mean()) > 0.2:
            continue
        if np.sqrt(np.mean(x[:, 1] ** 2)) <= np.sqrt(np.mean(y[:, 1] ** 2)) + 0.2:
            return end / hz
    return None


def slot_signal(control, motion, slot, manifest):
    mask = control[:, slot, 6].astype(bool)
    c, m = control[mask, slot], motion[mask, slot]
    names = manifest["motion_columns"]
    forward = m[:, names.index("root_x")]
    speed = np.diff(forward, prepend=forward[0]) * manifest["control_hz"]
    return np.stack([speed, c[:, 2], m[:, names.index("move")] > 0.5], axis=-1)


def load_cohort(path):
    m = json.loads((path / "manifest.json").read_text())
    events = json.loads((path / "event-impulses.json").read_text())
    rows = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    if [e["slot"] for e in events] != list(range(m["evaluation_episodes"])):
        raise ValueError(f"Invalid event slots: {path}")
    if sorted(r["slot"] for r in rows) != list(range(m["evaluation_episodes"])):
        raise ValueError(f"Invalid episode slots: {path}")
    with np.load(path / "traces.npz") as traces:
        physics, control, motion = (traces[k] for k in ("physics", "control", "motion"))
    return m, events, {r["slot"]: r for r in rows}, physics, control, motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cohorts = {}
    for path in sorted(args.root.glob("*/*/evaluation.json")):
        value = load_cohort(path.parent)
        m = value[0]
        key = (m["label"], m["evaluation_seed"], m["event_push"])
        if key in cohorts:
            raise ValueError(f"Duplicate cohort {key}")
        cohorts[key] = value
    if not cohorts:
        raise ValueError("No event cohorts available")

    groups, rows = [], []
    for key, (manifest, events, episodes, physics, control, motion) in sorted(cohorts.items()):
        label, seed, stage = key
        baseline = cohorts.get((label, seed, "baseline"))
        hz = manifest["control_hz"]
        samples_per_tick = manifest["physics_hz"] // hz
        if baseline is not None and any(manifest[k] != baseline[0][k] for k in MATCHING):
            raise ValueError(f"Baseline initial conditions or provenance differ: {key}")
        group = []
        for event in events:
            slot, impulse = event["slot"], event["impulse_substep"]
            detected = event["events_substeps"]
            if stage == "baseline" and impulse >= 0:
                raise ValueError("Baseline must have no impulses")
            if impulse >= 0 and (
                detected[stage] < 0 or not 0 <= impulse - detected[stage] < samples_per_tick
            ):
                raise ValueError(f"Impulse was not applied at the next control boundary: {key}, {slot}")
            landed = episode_metrics(physics[:, slot], control[:, slot], manifest)
            row = dict(
                model=label, seed=seed, stage=stage, slot=slot,
                impulse_substep=impulse, swing_substep=detected["swing"],
                touchdown_substep=detected["touchdown"], after_substep=detected["after"],
                landing_peak_bw=landed.get("landing_peak_bw"),
                landing_down_m_s=landed.get("landing_down_m_s"),
                recovery_s=None, followup_s=None,
                failure=episodes[slot]["terminal_v1"]["failure_reason"],
            )
            if impulse >= 0 and baseline is not None:
                m, base_events, _, bp, bc, bm = baseline
                reference_tick = base_events[slot]["events_substeps"][stage]
                if reference_tick >= 0:
                    start = impulse // samples_per_tick
                    base_start = (reference_tick + samples_per_tick - 1) // samples_per_tick
                    x = slot_signal(control, motion, slot, manifest)[start:start + 2 * hz]
                    y = slot_signal(bc, bm, slot, m)[base_start:base_start + 2 * hz]
                    row["followup_s"] = min(len(x), len(y)) / hz
                    row["recovery_s"] = recovery_seconds(x, y, hz)
                    if min(len(x), len(y)) >= hz // 2:
                        row["gyro_peak_delta_0p5_rad_s"] = float(
                            x[:hz // 2, 1].max() - y[:hz // 2, 1].max()
                        )
                    b = episode_metrics(bp[:, slot], bc[:, slot], m)
                    if row["landing_peak_bw"] is not None and b.get("landing_peak_bw") is not None:
                        row["landing_peak_delta_bw"] = row["landing_peak_bw"] - b["landing_peak_bw"]
            group.append(row)
        rows.extend(group)
        groups.append(dict(
            model=label, seed=seed, stage=stage, episodes=len(group),
            impulses_applied=sum(r["impulse_substep"] >= 0 for r in group),
            matched_followup=sum(r["followup_s"] is not None for r in group),
            recovered=sum(r["recovery_s"] is not None for r in group),
            metrics={k: stats([r[k] for r in group if r.get(k) is not None]) for k in (
                "recovery_s", "followup_s", "gyro_peak_delta_0p5_rad_s",
                "landing_peak_bw", "landing_down_m_s", "landing_peak_delta_bw",
            )},
        ))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.root / "episode-events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "event-comparison.json").write_text(
        json.dumps(groups, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
