import numpy as np
import pytest

from embodied_learning.evaluation.cat_obstacle_expert import (
    ExpertProgress,
    gate_from_expert_evaluations,
)


def tick(progress, step, x, *, y=0.0, vx=0.3, collision=False, fall=False, done=False):
    progress.observe(
        step, np.array([x]), np.array([y]), np.array([vx]),
        np.array([0.0]), np.array([0.0]),
        np.array([collision]), np.array([fall]), np.array([False]), np.array([done]),
    )


@pytest.mark.parametrize(
    ("y", "vx", "delay", "collision", "fall", "done", "reason"),
    [
        (0, 0.3, 0, False, False, False, "success"),
        (0.6, 0.3, 0, False, False, False, "outside_corridor"),
        (0, -0.3, 0, False, False, False, "exit_not_recovered"),
        (0, 0.3, 130, False, False, False, "exit_not_recovered"),
        (0, 0.3, 0, True, False, False, "collision"),
        (0, 0.3, 0, True, True, False, "fall"),
        (0, 0.3, 0, False, False, True, "exit_not_recovered"),
    ],
)
def test_expert_gate_requires_traversal_then_safe_locomotion(y, vx, delay, collision, fall, done, reason):
    progress = ExpertProgress(1, front_x=-0.1, exit_x=0.35)
    tick(progress, 1, -0.5)
    tick(progress, 10, 0.0, y=y)
    tick(progress, 40, 0.36, vx=vx)
    for step in range(41, 50):
        tick(progress, step + delay, 0.45, vx=vx)
    tick(progress, 50 + delay, 0.71, vx=vx, collision=collision, fall=fall, done=done)
    progress.finish(np.array([True]), 1000)
    assert progress.records()[0]["result"] == reason


def test_success_finishes_skill_before_navigation_stop():
    progress = ExpertProgress(1, front_x=-0.1, exit_x=0.35)
    tick(progress, 1, -0.5)
    tick(progress, 10, 0.0)
    tick(progress, 40, 0.36)
    for step in range(41, 51):
        tick(progress, step, 0.75)
    tick(progress, 51, 0.76, fall=True)
    assert progress.records()[0]["result"] == "success"


@pytest.mark.parametrize(("successes", "passed"), [(225, False), (226, True)])
def test_promotion_boundary(successes, passed):
    cohorts = []
    for seed, count in [(10, 128), (11, successes - 128)]:
        cohorts.append(dict(
            seed=seed, completed_updates=600, checkpoint_sha256="ckpt",
            evaluation_code_commit="git", training_seed=7, protocol_sha256="protocol",
            training_stage=0, evaluation_stage=0, expert_version="cat_obstacle_expert_v1",
            disturbance_profile="unitree", results={"success": count, "no_crossing": 128-count},
        ))
    assert gate_from_expert_evaluations(cohorts)["passed"] is passed
    with pytest.raises(ValueError):
        gate_from_expert_evaluations([cohorts[0], cohorts[0]])
