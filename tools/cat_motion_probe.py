"""Evaluation-only snapshots before automatic reset; does not change inputs."""

import json
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def record_motion(env, output):
    state = env.cat_state
    original = state.refresh
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    first = {
        k: torch.full((env.num_envs,), -1.0, device=env.device)
        for k in (
            "physical_contact",
            "point_inside",
            "root_past_bar",
            "both_feet_past_bar",
            "goal",
        )
    }
    max_height = torch.full((env.num_envs, 2), -torch.inf, device=env.device)
    min_head = torch.full((env.num_envs,), torch.inf, device=env.device)
    max_lag = torch.full((env.num_envs,), -torch.inf, device=env.device)
    trace = []
    scene = next(s for s in state.manifest.scenes if s.id == "hurdle_crouch")
    bars = [
        b
        for b in scene.boxes
        if b.center[2] - b.half_size[2] <= 1e-6
        and b.center[2] + b.half_size[2] < 0.5
    ]
    bar = min(bars, key=lambda b: b.center[0]) if bars else None
    front = bar.center[0] - bar.half_size[0] if bar else None
    back = bar.center[0] + bar.half_size[0] if bar else None

    def refresh(payload, controller, lengths, common):
        fresh = state._last_common != common
        selected = active & ~state.initial.clone()
        original(payload, controller, lengths, common)
        if not fresh or not bool(selected.any()):
            return
        pos, root = payload["body_positions"], payload["root_position"]
        feet = pos[:, 3:5]
        near = (
            (feet[..., 0] >= front - 0.3) & (feet[..., 0] <= back + 0.1)
            if bar
            else torch.zeros_like(feet[..., 0], dtype=torch.bool)
        )
        max_height.copy_(
            torch.maximum(
                max_height,
                torch.where(selected[:, None] & near, feet[..., 2], -torch.inf),
            )
        )
        min_head.copy_(
            torch.minimum(min_head, torch.where(selected, pos[:, 0, 2], torch.inf))
        )
        lag = root[:, 0] - feet[..., 0].amax(-1)
        max_lag.copy_(torch.maximum(max_lag, torch.where(selected, lag, -torch.inf)))
        events = dict(
            physical_contact=payload["physical_collision"],
            point_inside=(payload["reward_inputs"].sdf < 0).any(-1),
            root_past_bar=root[:, 0] > back if bar else torch.zeros_like(selected),
            both_feet_past_bar=(feet[..., 0] > back).all(-1)
            if bar
            else torch.zeros_like(selected),
            goal=state.saved_outcomes["goal"],
        )
        for key, event in events.items():
            hit = selected & event & (first[key] < 0)
            first[key][hit] = lengths[hit].float() * 0.02
        count = min(8, env.num_envs)
        trace.append(
            dict(
                step=int(common),
                active=selected[:count].cpu().numpy(),
                root=root[:count].detach().cpu().numpy(),
                body=pos[:count].detach().cpu().numpy(),
                root_velocity=payload["reward_inputs"]
                .root_vel[:count]
                .detach()
                .cpu()
                .numpy(),
                command=state.gait.last_command[:count].detach().cpu().numpy(),
                actor_command=state._obs["actor"][:count, 77:80]
                .detach()
                .cpu()
                .numpy(),
                feet_contact=payload["reward_inputs"]
                .feet_contact[:count]
                .detach()
                .cpu()
                .numpy(),
                foot_gf=payload["reward_inputs"]
                .gf[:count, 3:5]
                .detach()
                .cpu()
                .numpy(),
                foot_sdf=payload["reward_inputs"]
                .sdf[:count, 3:5]
                .detach()
                .cpu()
                .numpy(),
                stop_counter=state.gait.stop_timestep[:count].cpu().numpy(),
            )
        )
        flags = state.saved_outcomes
        active[
            selected
            & (
                flags["fall"]
                | flags["collision"]
                | flags["invalid"]
                | flags["timeout"]
            )
        ] = False

    state.refresh = refresh
    try:
        yield
    finally:
        state.refresh = original
        if trace:
            np.savez_compressed(
                output / "motion-trace.npz",
                **{k: np.stack([row[k] for row in trace]) for k in trace[0]},
            )

        def finite(x):
            return float(x) if np.isfinite(float(x)) else None

        per_episode = [
            dict(
                slot=i,
                first_seconds={
                    k: (float(v[i]) if v[i] >= 0 else None) for k, v in first.items()
                },
                max_foot_site_z_near_bar=[finite(x) for x in max_height[i]],
                min_head_z=finite(min_head[i]),
                max_root_ahead_of_both_foot_sites=finite(max_lag[i]),
            )
            for i in range(env.num_envs)
        ]
        (output / "motion-summary.json").write_text(
            json.dumps(
                dict(
                    bar_front_m=front,
                    bar_back_m=back,
                    bar_top_m=bar.center[2] + bar.half_size[2] if bar else None,
                    event_counts={k: int((v >= 0).sum()) for k, v in first.items()},
                    episodes=per_episode,
                    limits=[
                        "First episode per slot only; snapshots are before reset.",
                        "Passing foot sites is a progress metric, not whole-foot "
                        "clearance or proof of stepping over rather than detouring.",
                    ],
                ),
                indent=2,
                allow_nan=False,
            )
        )
