import copy
from types import SimpleNamespace

import mujoco
import pytest
import torch
from test_cat_field_weights import load, trial
from test_cat_joint_velocity import inputs

from embodied_learning.tasks.cat_contact_force import (
    FootContactForceState,
    contact_force_cost,
    validate_schema,
    verify_measurement_model,
)
from embodied_learning.tasks.cat_foot_guidance import sha256
from embodied_learning.tasks.cat_reward import cat_reward_terms

SCHEMA = dict(version="ground_force_peak_bw_v1", threshold_bw=3.0, weight=0.2)


def test_support_below_threshold_is_free_and_excess_is_body_weight_normalized():
    force = torch.tensor(
        [
            [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
            [[0.0, 0.0, 1200.0], [0.0, 0.0, 400.0]],
        ]
    )
    torch.testing.assert_close(
        contact_force_cost(force.norm(dim=-1), weight_n=400.0, threshold_bw=3.0),
        torch.tensor([0.0, 0.0]),
    )
    force[1, 0, 2] = 1600.0
    torch.testing.assert_close(
        contact_force_cost(force.norm(dim=-1), weight_n=400.0, threshold_bw=3.0),
        torch.tensor([0.0, 1.0]),
    )


def test_force_peak_catches_short_final_substep_without_double_charge():
    s = FootContactForceState(1, "cpu", weight_n=100.0, threshold_bw=3.0)
    for i in range(10):
        s.accumulate(
            torch.tensor([[[0.0, 0.0, 500.0 if i == 9 else 50.0], [0.0, 0.0, 50.0]]])
        )
    torch.testing.assert_close(s.cost_rate, torch.tensor([2.0]))
    torch.testing.assert_close(s.cost_rate, torch.tensor([2.0]))
    s.begin_control()
    torch.testing.assert_close(s.cost_rate, torch.tensor([0.0]))


def test_contact_force_reset_preserves_other_environments():
    s = FootContactForceState(2, "cpu", weight_n=100.0, threshold_bw=3.0)
    s.accumulate(torch.tensor([[[0.0, 0.0, 500.0], [0.0, 0.0, 50.0]]] * 2))
    s.reset(torch.tensor([0]))
    torch.testing.assert_close(s.cost_rate, torch.tensor([0.0, 2.0]))


def test_sensor_sampling_occurs_after_step_and_refreshes_force_cache():
    events = []

    def sensor(value):
        return SimpleNamespace(
            update=lambda dt: events.append(("update", dt)),
            data=SimpleNamespace(force=torch.tensor([[[0.0, 0.0, value]]])),
        )

    n = SimpleNamespace(
        sim=SimpleNamespace(
            step=lambda: events.append("step"), sense=lambda: events.append("sense")
        ),
        scene={"cat_left_ground": sensor(500.0), "cat_right_ground": sensor(50.0)},
    )
    s = FootContactForceState(1, "cpu", weight_n=100.0, threshold_bw=3.0)
    s.install(n)
    n.sim.step()
    assert events == ["step", "sense", ("update", 0.0), ("update", 0.0)]
    torch.testing.assert_close(s.cost_rate, torch.tensor([2.0]))


def test_sensor_extension_cannot_hide_changed_physics():
    xml = (
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/>'
        '<site name="imu"/></body></worldbody>{sensor}</mujoco>'
    )
    base = mujoco.MjModel.from_xml_string(xml.format(sensor=""))
    actual = mujoco.MjModel.from_xml_string(
        xml.format(sensor='<sensor><accelerometer site="imu"/></sensor>')
    )
    verify_measurement_model(actual, base)
    actual.body_mass[1] += 1.0
    with pytest.raises(ValueError):
        verify_measurement_model(actual, base)


def test_sensor_verification_rejects_changed_integration_timestep():
    xml = '<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>'
    reference = mujoco.MjModel.from_xml_string(xml)
    actual = mujoco.MjModel.from_xml_string(xml)
    actual.opt.timestep *= 0.5
    with pytest.raises(ValueError):
        verify_measurement_model(actual, reference)


def test_force_norm_includes_horizontal_ground_reaction():
    state = FootContactForceState(1, "cpu", weight_n=100.0, threshold_bw=3.0)
    state.accumulate(torch.tensor([[[400.0, 0.0, 0.0], [0.0, 0.0, 50.0]]]))
    torch.testing.assert_close(state.cost_rate, torch.tensor([1.0]))


@pytest.mark.parametrize(
    "key,value",
    [
        ("threshold_bw", 0),
        ("threshold_bw", float("nan")),
        ("weight", -1),
        ("weight", True),
        ("extra", 1),
        ("version", "bad"),
    ],
)
def test_bad_force_schema_rejected(key, value):
    with pytest.raises(ValueError):
        validate_schema({**SCHEMA, key: value})


def test_new_term_does_not_mutate_old_rewards_or_require_a_gait_phase():
    x, p = inputs()
    x.gait_mask.zero_()
    old = cat_reward_terms(x, parameters=p)
    new = cat_reward_terms(
        x, parameters=p, contact_force_rate=torch.ones_like(x.move_flag)
    )
    assert set(new) == set(old) | {"foot_contact_force"}
    for k in old:
        torch.testing.assert_close(new[k], old[k], rtol=0, atol=0)
    torch.testing.assert_close(
        new["foot_contact_force"], torch.ones_like(x.move_flag)
    )


@pytest.mark.parametrize(
    "change",
    [None, "hash", "pd", "physics", "stage", "missing", "observation", "extra"],
)
def test_force_transfer_only_permits_declared_reward(tmp_path, change):
    p, saved, target, meta = trial(tmp_path, {})
    meta.pop("field_reward_refinement")
    target["schema"]["foot_contact_force"] = copy.deepcopy(SCHEMA)
    meta["contact_force_refinement"] = dict(
        version="cat_contact_force_actor_transfer_v1",
        parent_sha256=sha256(p),
        parent_experiment="test",
        parent_protocol_sha256="protocol",
        before=None,
        after=copy.deepcopy(SCHEMA),
    )
    if change == "hash":
        meta["contact_force_refinement"]["parent_sha256"] = "bad"
    if change == "pd":
        target["schema"]["controller_pd"] = {"kp": [140.0]}
    if change == "physics":
        target["schema"]["physics"] = {"timestep": 0.001}
    if change == "stage":
        meta["stage"] = 3
    if change == "missing":
        meta.pop("contact_force_refinement")
    if change == "observation":
        target["actor_dim"] += 1
    if change == "extra":
        meta["contact_force_refinement"]["extra"] = 1
    if change:
        with pytest.raises(ValueError):
            load(p, saved, target, meta)
    else:
        actual, info = load(p, saved, target, meta)
        assert actual["field_contract"] == saved["field_contract"]
        assert info["contract_changes"]["foot_contact_force"] == dict(
            before=None, after=SCHEMA
        )
        assert not info["critic_restored"] and not info["optimizer_restored"]


def test_unchanged_baseline_has_explicit_actor_transfer_but_no_force_reward(
    tmp_path,
):
    p, saved, target, meta = trial(tmp_path, {})
    meta.pop("field_reward_refinement")
    meta["contact_force_refinement"] = dict(
        version="cat_contact_force_actor_transfer_v1",
        parent_sha256=sha256(p),
        parent_experiment="test",
        parent_protocol_sha256="protocol",
        before=None,
        after=None,
    )
    actual, info = load(p, saved, target, meta)
    assert actual["field_contract"] == target
    assert info["contract_changes"] == {}


def test_full_resume_cannot_add_force_penalty(tmp_path):
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    env = TinyEnv()
    cfg = PpoConfig(
        iterations=1, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    parent = train_specialist(env, cfg, tmp_path / "parent")
    env.schema = {**env.schema, "foot_contact_force": SCHEMA}
    with pytest.raises(ValueError):
        train_specialist(env, cfg, tmp_path / "bad-resume", resume=parent)


def test_existing_landing_experiment_cannot_silently_stack_with_force_cost():
    from embodied_learning.tasks.cat_contact_force import apply_contact_force
    from embodied_learning.tasks.cat_hurdle import apply_runtime_settings

    empty = SimpleNamespace(
        unwrapped=SimpleNamespace(cat_controller=object()),
        cat_state=SimpleNamespace(scales={}),
        schema={},
    )
    apply_runtime_settings(empty, {})
    assert empty.schema == {} and empty.cat_state.scales == {}
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cat_controller=SimpleNamespace(landing=object())),
        cat_state=SimpleNamespace(recovery=None),
    )
    with pytest.raises(ValueError):
        apply_contact_force(env, {"foot_contact_force": SCHEMA})
