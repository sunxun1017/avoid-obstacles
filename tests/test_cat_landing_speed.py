import copy
from types import SimpleNamespace

import mujoco
import pytest
import torch
from test_cat_field_weights import load, trial
from test_cat_joint_velocity import inputs

from embodied_learning.tasks.cat_foot_guidance import sha256
from embodied_learning.tasks.cat_landing_speed import (
    FootLandingState,
    SoleGeometry,
    landing_speed_cost,
    validate_schema,
)
from embodied_learning.tasks.cat_reward import (
    aggregate_cat_reward,
    cat_reward_terms,
)

SCHEMA = dict(
    version="sole_descent_substep_v1", height_m=0.05, speed_m_s=0.5, weight=0.1
)


def test_landing_cost_covers_neutral_phase_without_penalizing_lift_or_high_swing():
    h = torch.tensor([[0.0, 0.025], [0.05, 0.2], [0.0, 0.0]])
    v = torch.tensor([[-2.5, -1.5], [-4.0, -4.0], [2.0, -0.5]])
    # No gait argument: the actual near-ground motion is the gate.
    torch.testing.assert_close(
        landing_speed_cost(h, v, height_m=0.05, speed_m_s=0.5),
        torch.tensor([4.5, 0.0, 0.0]),
    )


def test_substep_landing_between_control_observations_is_integrated_once():
    state = FootLandingState(1, "cpu", height_m=0.05, speed_m_s=0.5, control_dt=0.02)
    for step in range(10):
        state.accumulate(
            torch.zeros(1, 2),
            torch.tensor([[-2.5, 0.0]]) if step in (3, 4) else torch.zeros(1, 2),
            dt=0.002,
        )
    # A 4ms landing contributes0.016 integral/0.8 rate even with quiet endpoints.
    torch.testing.assert_close(state.cost_rate, torch.tensor([0.8]))
    torch.testing.assert_close(state.cost_rate, torch.tensor([0.8]))
    state.begin_control()
    torch.testing.assert_close(state.cost_rate, torch.zeros(1))


def test_reset_only_clears_selected_environments():
    state = FootLandingState(2, "cpu", height_m=0.05, speed_m_s=0.5, control_dt=0.02)
    state.accumulate(torch.zeros(2, 2), torch.full((2, 2), -1.5), dt=0.002)
    state.reset(torch.tensor([0]))
    torch.testing.assert_close(state.cost_rate, torch.tensor([0.0, 0.2]))


def test_native_sampling_counts_initial_and_final_physics_substeps_once():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".002" gravity="0 0 0"/>
      <worldbody><body pos="0 0 1"><freejoint/><geom size=".01"/></body></worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[2] = -1.0
    mujoco.mj_forward(model, data)
    state = FootLandingState(1, "cpu", height_m=0.05, speed_m_s=0.5, control_dt=0.02)
    heights = []

    def sample(native):
        z = float(data.geom_xpos[0, 2])
        heights.append(z)
        # Only the final substep is a near-ground descent event.
        h = 0.0 if 0.981 < z < 0.983 else 0.1
        state.accumulate(
            torch.tensor([[h, 1.0]]), torch.tensor([[-2.5, 0.0]]), dt=0.002
        )

    state.sample = sample
    native = SimpleNamespace(
        sim=SimpleNamespace(step=lambda: mujoco.mj_step(model, data))
    )
    state.install(native)
    for _ in range(10):
        native.sim.step()
    assert heights == pytest.approx(
        [1.0, 0.998, 0.996, 0.994, 0.992, 0.990, 0.988, 0.986, 0.984, 0.982]
    )
    torch.testing.assert_close(state.cost_rate, torch.tensor([0.4]))
    # Native control-boundary forward must not charge the old first state again.
    mujoco.mj_forward(model, data)
    state.begin_control()
    native.sim.step()
    assert heights[-1] == pytest.approx(0.980)
    torch.testing.assert_close(state.cost_rate, torch.tensor([0.0]))


def test_rotated_capsule_and_floor_offset_use_lowest_collision_geometry():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="robot/left_foot0_collision" type="capsule" size=".01 .1"/>
      <geom name="robot/left_foot1_collision" type="sphere" size=".02"/>
      <geom name="robot/right_foot0_collision" type="sphere" size=".02"/>
    </worldbody></mujoco>""")
    geometry = SoleGeometry.from_model(model, "cpu")
    pos = torch.tensor([[[0.0, 0.0, 0.2], [0.0, 0.0, 0.25], [0.0, 0.0, 0.1]]])
    rot = torch.eye(3).repeat(1, 3, 1, 1)
    origins = torch.tensor([[0.0, 0.0, 0.05]])
    torch.testing.assert_close(
        geometry.heights(pos, rot, origins), torch.tensor([[0.04, 0.03]])
    )
    rot[0, 0] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    torch.testing.assert_close(
        geometry.heights(pos, rot, origins), torch.tensor([[0.14, 0.03]])
    )


def test_missing_foot_geometry_is_rejected():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="robot/left_foot0_collision" '
        'size=".02"/></worldbody></mujoco>'
    )
    with pytest.raises(ValueError):
        SoleGeometry.from_model(model, "cpu")


@pytest.mark.parametrize(
    "key,value",
    [
        ("weight", 0),
        ("height_m", 0),
        ("speed_m_s", -1),
        ("weight", True),
        ("weight", float("nan")),
        ("height_m", float("inf")),
        ("version", "unknown"),
        ("extra", 1),
    ],
)
def test_invalid_landing_schema_is_rejected(key, value):
    with pytest.raises(ValueError):
        validate_schema({**SCHEMA, key: value})


def test_historical_reward_is_unchanged_and_new_term_has_exact_negative_sign():
    x, parameters = inputs()
    old = cat_reward_terms(x, parameters=parameters)
    new = cat_reward_terms(
        x,
        parameters=parameters,
        landing_speed_rate=torch.full_like(x.move_flag, 0.8),
    )
    assert set(new) == set(old) | {"foot_landing_speed"}
    for k in old:
        torch.testing.assert_close(new[k], old[k], rtol=0, atol=0)
    # Use unclipped positive reward to expose a wrong weight/sign/dt.
    rate = torch.tensor([2.0])
    actual = aggregate_cat_reward(
        {"positive": torch.tensor([10.0]), "foot_landing_speed": rate},
        {"positive": 1.0, "foot_landing_speed": -0.1},
        dt=0.02,
    )
    torch.testing.assert_close(actual, torch.tensor([0.196]))


@pytest.mark.parametrize(
    "change",
    [
        None,
        "sha",
        "stage",
        "pd",
        "physics",
        "observation",
        "missing",
        "extra",
        "same_run",
    ],
)
def test_actor_transfer_changes_only_declared_landing_reward(tmp_path, change):
    p, saved, target, meta = trial(tmp_path, {})
    meta.pop("field_reward_refinement")
    target["schema"]["foot_landing_speed"] = copy.deepcopy(SCHEMA)
    meta["landing_speed_refinement"] = dict(
        version="cat_landing_speed_actor_transfer_v1",
        parent_sha256=sha256(p),
        parent_experiment="test",
        parent_protocol_sha256="protocol",
        before=None,
        after=copy.deepcopy(SCHEMA),
    )
    if change == "sha":
        meta["landing_speed_refinement"]["parent_sha256"] = "bad"
    if change == "stage":
        meta["stage"] = 3
    if change == "pd":
        target["schema"]["controller_pd"] = {"kp": [140]}
    if change == "physics":
        target["schema"]["physics"] = {"timestep": 0.001}
    if change == "observation":
        target["actor_dim"] += 1
    if change == "missing":
        meta.pop("landing_speed_refinement")
    if change == "extra":
        meta["landing_speed_refinement"]["extra"] = 1
    if change == "same_run":
        meta.update(experiment="test", protocol_sha256="protocol")
    if change:
        with pytest.raises(ValueError):
            load(p, saved, target, meta)
    else:
        actual, info = load(p, saved, target, meta)
        assert actual["field_contract"] == saved["field_contract"]
        assert info["contract_changes"]["foot_landing_speed"] == dict(
            before=None, after=SCHEMA
        )
        assert not info["critic_restored"] and not info["optimizer_restored"]


def test_neutral_phase_task_reward_receives_substep_integral_exactly_once():
    from dataclasses import fields

    from test_cat_review_regressions import task_case

    baseline, controller, payload, _ = task_case()
    x = payload["reward_inputs"]
    for item in fields(x):
        getattr(x, item.name).zero_()
    x.joint_pos.copy_(controller.default.expand_as(x.joint_pos))
    x.foot_pos[:, 0, 1] = 0.1
    x.foot_pos[:, 1, 1] = -0.1
    x.head_pos[:, 2] = 1.0
    x.navigation_rotation.copy_(torch.eye(3).expand_as(x.navigation_rotation))
    x.leg_rotation.copy_(torch.eye(3).expand_as(x.leg_rotation))
    baseline.bank.sample_source = lambda points, *args: SimpleNamespace(
        guidance=torch.zeros_like(points),
        sdf_gradient=torch.zeros_like(points),
        sdf=torch.full(points.shape[:2], 0.6),
    )
    candidate = copy.deepcopy(baseline)
    cc = copy.deepcopy(controller)
    cp = copy.deepcopy(payload)
    cc.landing = FootLandingState(
        16, "cpu", height_m=0.05, speed_m_s=0.5, control_dt=0.02
    )
    cc.landing.accumulate(torch.zeros(16, 2), torch.full((16, 2), -1.5), dt=0.002)
    candidate.scales["foot_landing_speed"] = -0.1
    rng = torch.get_rng_state()
    baseline.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    torch.set_rng_state(rng)
    candidate.refresh(cp, cc, torch.zeros(16, dtype=torch.long), 0)
    for key in baseline.terms:
        torch.testing.assert_close(candidate.terms[key], baseline.terms[key])
    torch.testing.assert_close(
        candidate.terms["foot_landing_speed"], torch.full((16,), 0.2)
    )
    assert bool((baseline.reward() > 0.01).all())
    torch.testing.assert_close(
        candidate.reward() - baseline.reward(), torch.full((16,), -0.0004)
    )
    # Re-reading the same step does not add another substep charge.
    before = candidate.reward().clone()
    candidate.refresh(cp, cc, torch.zeros(16, dtype=torch.long), 0)
    torch.testing.assert_close(candidate.reward(), before)


def test_full_state_resume_rejects_new_landing_reward(tmp_path):
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    env = TinyEnv()
    cfg = PpoConfig(
        iterations=1, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    parent = train_specialist(env, cfg, tmp_path / "parent")
    env.schema = {**env.schema, "foot_landing_speed": SCHEMA}
    with pytest.raises(ValueError):
        train_specialist(env, cfg, tmp_path / "bad-resume", resume=parent)


def test_baseline_transfer_requires_explicit_unchanged_reward_parent(tmp_path):
    p, saved, target, meta = trial(tmp_path, {})
    meta.pop("field_reward_refinement")
    with pytest.raises(ValueError):
        load(p, saved, target, meta)
    meta["landing_speed_refinement"] = dict(
        version="cat_landing_speed_actor_transfer_v1",
        parent_sha256=sha256(p),
        parent_experiment="test",
        parent_protocol_sha256="protocol",
        before=None,
        after=None,
    )
    actual, info = load(p, saved, target, meta)
    assert actual["field_contract"] == target
    assert info["contract_changes"] == {}
    assert not info["critic_restored"] and not info["optimizer_restored"]
