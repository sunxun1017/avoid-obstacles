"""Evaluate centered static traversal and the subsequent walking handoff."""

from collections import Counter

import numpy as np


EXPERT_PROFILE = "cat_obstacle_expert_v1"
RESULTS = (
    "success", "collision", "fall", "invalid", "outside_corridor",
    "no_crossing", "exit_not_recovered",
)


class ExpertProgress:
    def __init__(self, count, *, front_x, exit_x, dt=0.02):
        self.front_x = front_x
        self.exit_x = exit_x
        self.dt = dt
        self.window = round(2.5 / dt)
        self.hold = round(0.2 / dt)
        self.result = np.full(count, "", dtype="<U20")
        self.seen_approach = np.zeros(count, dtype=bool)
        self.entered = np.zeros(count, dtype=bool)
        self.crossing = np.zeros(count, dtype=int)
        self.stable = np.zeros(count, dtype=int)
        self.end = np.zeros(count, dtype=int)
        self.max_x = np.full(count, -np.inf)
        self.pushes = np.zeros(count, dtype=int)

    def observe(self, step, x, y, vx, tilt, angular_speed, contact, fall, invalid, done):
        active = self.result == ""
        measurable = active & ~done & np.isfinite(x)
        self.max_x[measurable] = np.maximum(self.max_x[measurable], x[measurable])
        # Evaluate safety first and never score a pose from an auto-reset slot.
        invalid = invalid | ~np.isfinite(x + y + vx + tilt + angular_speed)
        self.result[active & contact] = "collision"
        self.result[active & fall] = "fall"
        self.result[active & invalid] = "invalid"
        eligible = (self.result == "") & ~done
        approaching = x < self.front_x
        self.seen_approach |= eligible & approaching
        in_corridor = eligible & self.seen_approach & ~approaching & (x < self.exit_x)
        self.result[in_corridor & (np.abs(y) > 0.32)] = "outside_corridor"
        self.entered |= in_corridor & (np.abs(y) <= 0.32)
        crossing = (
            eligible & self.entered & (x >= self.exit_x) & (self.crossing == 0)
        )
        self.result[crossing & (np.abs(y) > 0.32)] = "outside_corridor"
        self.crossing[crossing & (self.result == "")] = step
        recovering = eligible & (self.result == "") & (self.crossing > 0)
        expired = recovering & (step - self.crossing > self.window)
        stable = (
            recovering & (step > self.crossing) & (x >= self.exit_x)
            & (vx > 0.08) & (tilt < np.pi / 6) & (angular_speed < 1.5)
        )
        self.stable = np.where(stable, self.stable + 1, 0)
        self.result[expired] = "exit_not_recovered"
        success = recovering & ~expired & (self.stable >= self.hold) & (x >= self.exit_x + 0.35)
        self.result[success] = "success"
        self.finish(done, step)
        self.end[active & (self.result != "")] = step

    def finish(self, slots, step):
        pending = slots & (self.result == "")
        self.result[pending & (self.crossing == 0)] = "no_crossing"
        self.result[pending & (self.crossing > 0)] = "exit_not_recovered"
        self.end[pending] = step

    def records(self):
        return [
            {
                "episode": i, "result": reason,
                "crossing_s": int(self.crossing[i]) * self.dt if self.crossing[i] else None,
                "end_s": int(self.end[i]) * self.dt,
                "recovery_s": (
                    int(self.end[i] - self.crossing[i]) * self.dt if reason == "success" else None
                ),
                "pushes_during_skill": int(self.pushes[i]),
                "max_x_m": float(self.max_x[i]) if np.isfinite(self.max_x[i]) else None,
            }
            for i, reason in enumerate(self.result.tolist())
        ]


def evaluate_expert(env, policy, scene, *, episode_path):
    import json

    import torch

    from embodied_learning.evaluation.cat_substep_contacts import capture_prohibited_contacts

    front = min(box.center[0] - box.half_size[0] for box in scene.boxes)
    back = max(box.center[0] + box.half_size[0] for box in scene.boxes)
    native = env.unwrapped
    progress = ExpertProgress(env.num_envs, front_x=front, exit_x=back + 0.30, dt=native.step_dt)
    robot = native.scene["robot"].data
    origins = native.scene.env_origins
    obs = env.get_observations()
    with torch.inference_mode(), capture_prohibited_contacts(env):
        for step in range(1, native.max_episode_length + 1):
            push_due = native.episode_length_buf + 1 == env.cat_state.push_interval
            obs, _, done, extras = env.step(policy.predict(obs))
            outcomes = extras["field_outcomes"]
            position = robot.root_link_pos_w - origins
            vx = robot.root_link_lin_vel_w[:, 0]
            tilt = (-robot.projected_gravity_b[:, 2]).clamp(-1, 1).acos()
            angular_speed = robot.root_link_ang_vel_w.norm(dim=-1)
            values = torch.stack((
                position[:, 0], position[:, 1], vx, tilt, angular_speed,
                outcomes["prohibited_contact"], outcomes["fall"], outcomes["invalid"], done, push_due,
            )).cpu().numpy()
            progress.pushes += (progress.result == "") & values[9].astype(bool)
            progress.observe(step, *values[:5], *values[5:9].astype(bool))
            if (progress.result != "").all():
                break
    progress.finish(np.ones(env.num_envs, dtype=bool), step)
    records = progress.records()
    episode_path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
    results = Counter(r["result"] for r in records)
    return {
        "expert_version": EXPERT_PROFILE,
        "contact_sampling": "physics_substep",
        "push_exposed_episodes": int(np.count_nonzero(progress.pushes)),
        "results": {key: results[key] for key in RESULTS},
        "rates": {key: results[key] / len(records) for key in RESULTS},
        "geometry_gate": {"front_x_m": front, "exit_x_m": progress.exit_x, "half_width_m": 0.32},
    }


def gate_from_expert_evaluations(reports):
    if len(reports) != 2 or len({report["seed"] for report in reports}) != 2:
        raise ValueError("Expert gate requires two distinct evaluation cohorts")
    first = reports[0]
    keys = (
        "checkpoint_sha256", "evaluation_code_commit", "protocol_sha256",
        "training_seed", "training_stage", "disturbance_profile",
    )
    for report in reports:
        counts = report["results"]
        if (
            report["completed_updates"] != 600
            or report["expert_version"] != EXPERT_PROFILE
            or report["training_stage"] != report["evaluation_stage"]
            or report["seed"] == report["training_seed"]
            or any(report[key] != first[key] for key in keys)
            or set(counts) - set(RESULTS)
            or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) != 128
        ):
            raise ValueError("Invalid expert cohort size or provenance")
    totals = {key: sum(report["results"].get(key, 0) for report in reports) for key in RESULTS}
    return {
        **{key: first[key] for key in keys},
        "expert_version": EXPERT_PROFILE,
        "completed_updates": 600,
        "evaluation_seeds": [report["seed"] for report in reports],
        "successes": totals["success"], "episodes": 256,
        "passed": totals["success"] >= 226, "results": totals,
    }
