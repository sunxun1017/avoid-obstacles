import copy
from types import SimpleNamespace

import pytest
import torch
from test_cat_curriculum import fixture_contract, fixture_metadata

from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart
from embodied_learning.robots.g1 import CAT_JOINT_NAMES, CAT_KD, CAT_KP
from embodied_learning.tasks.cat_foot_guidance import sha256
from embodied_learning.tasks.cat_pd_transfer import validate_pd


def trial(tmp_path):
    old = fixture_contract(3)
    pd = dict(
        version="cat_ankle_pd30_2_v1",
        joint_names=list(CAT_JOINT_NAMES),
        kp=list(CAT_KP),
        kd=list(CAT_KD),
    )
    old["schema"]["controller_pd"] = pd
    saved = dict(
        field_contract=old,
        curriculum=fixture_metadata(3),
        training_scenes=[2],
        actor_state_dict={"w": torch.ones(3)},
        completed_iterations=416,
    )
    p = tmp_path / "parent.pt"
    torch.save(saved, p)
    target = fixture_contract(4)
    after = copy.deepcopy(pd)
    after["version"] = "cat_hip140_knee180_v1"
    for i in [0, 6]:
        after["kp"][i] = 140.0
        after["kd"][i] = 3.0
    for i in [3, 9]:
        after["kp"][i] = 180.0
        after["kd"][i] = 4.0
    target["schema"]["controller_pd"] = after
    meta = {
        **fixture_metadata(4),
        "experiment": "pd-restart",
        "protocol_sha256": "new-protocol",
        "controller_pd_transfer": dict(
            version="cat_controller_pd_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment="test",
            parent_protocol_sha256="protocol",
            source_stage=3,
            target_stage=4,
            before=pd,
            after=after,
        ),
    }
    return p, saved, target, meta


def load(p, saved, target, meta):
    return load_actor_warmstart(
        p,
        contract=SimpleNamespace(to_dict=lambda: target),
        curriculum=meta,
        training_scenes=saved["training_scenes"],
        device="cpu",
    )


def test_declared_next_stage_pd_change_loads_original_actor_only(tmp_path):
    p, saved, target, meta = trial(tmp_path)
    actual, info = load(p, saved, target, meta)
    assert actual["field_contract"] == saved["field_contract"]
    assert not info["critic_restored"] and not info["optimizer_restored"]
    assert info["parent_completed_iterations"] == 416
    assert info["contract_changes"]["controller_pd"] == dict(
        before=saved["field_contract"]["schema"]["controller_pd"],
        after=target["schema"]["controller_pd"],
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "hash",
        "parent_stage",
        "target_stage",
        "skip_stage",
        "robot",
        "physics",
        "reward",
        "observation",
        "declaration_extra",
        "pd_mismatch",
    ],
)
def test_pd_transfer_does_not_bypass_other_contract_checks(tmp_path, change):
    p, saved, target, meta = trial(tmp_path)
    if change == "missing":
        meta.pop("controller_pd_transfer")
    if change == "hash":
        meta["controller_pd_transfer"]["parent_sha256"] = "bad"
    if change == "parent_stage":
        meta["controller_pd_transfer"]["source_stage"] = 2
    if change == "target_stage":
        meta["controller_pd_transfer"]["target_stage"] = 5
    if change == "skip_stage":
        meta["stage"] = 5
        meta["controller_pd_transfer"]["target_stage"] = 5
    if change == "robot":
        meta["robot_fingerprint"] = "changed"
    if change == "physics":
        target["schema"]["physics"] = {"dt": 0.01}
    if change == "reward":
        target["schema"]["task"]["root_tracking_weight"] = 2.0
    if change == "observation":
        target["actor_dim"] = 171
    if change == "declaration_extra":
        meta["controller_pd_transfer"]["extra"] = 1
    if change == "pd_mismatch":
        target = copy.deepcopy(target)
        target["schema"]["controller_pd"]["kp"][0] = 150.0
    with pytest.raises(ValueError):
        load(p, saved, target, meta)


@pytest.mark.parametrize(
    "change", ["bool", "zero", "nan", "length", "names", "extra"]
)
def test_invalid_pd_schema_rejected(change):
    pd = dict(
        version="test",
        joint_names=list(CAT_JOINT_NAMES),
        kp=list(CAT_KP),
        kd=list(CAT_KD),
    )
    if change == "bool":
        pd["kp"][0] = True
    if change == "zero":
        pd["kd"][0] = 0
    if change == "nan":
        pd["kp"][0] = float("nan")
    if change == "length":
        pd["kp"].pop()
    if change == "names":
        pd["joint_names"].reverse()
    if change == "extra":
        pd["extra"] = 1
    with pytest.raises(ValueError):
        validate_pd(pd)


def test_same_experiment_reuses_strict_contract_instead_of_reapplying_old_transfer(
    tmp_path,
):
    p, saved, target, meta = trial(tmp_path)
    saved["field_contract"] = copy.deepcopy(target)
    saved["curriculum"] = copy.deepcopy(meta)
    torch.save(saved, p)
    actual, info = load(p, saved, target, meta)
    assert "controller_pd" not in info["contract_changes"]


def test_same_stage_pd_transfer_still_requires_unchanged_geometry(tmp_path):
    p, saved, target, meta = trial(tmp_path)
    after = target["schema"]["controller_pd"]
    target = copy.deepcopy(saved["field_contract"])
    target["schema"]["controller_pd"] = after
    meta["stage"] = 3
    meta["controller_pd_transfer"]["target_stage"] = 3
    _, info = load(p, saved, target, meta)
    assert "controller_pd" in info["contract_changes"]
    target["schema"]["scene_hash"] = "unrelated-change"
    with pytest.raises(ValueError):
        load(p, saved, target, meta)


def test_same_experiment_does_not_hide_new_pd_mismatch(tmp_path):
    p, saved, target, meta = trial(tmp_path)
    saved["field_contract"] = copy.deepcopy(target)
    saved["curriculum"] = copy.deepcopy(meta)
    torch.save(saved, p)
    target = copy.deepcopy(target)
    target["schema"]["controller_pd"]["kp"][0] = 150.0
    with pytest.raises(ValueError):
        load(p, saved, target, meta)


def test_full_resume_cannot_change_pd(tmp_path):
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    env = TinyEnv()
    env.schema = {**env.schema, "controller_pd": {"hip_kp": 100}}
    cfg = PpoConfig(
        iterations=1, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    parent = train_specialist(env, cfg, tmp_path / "parent")
    env.schema = {**env.schema, "controller_pd": {"hip_kp": 140}}
    with pytest.raises(ValueError):
        train_specialist(env, cfg, tmp_path / "bad-resume", resume=parent)
