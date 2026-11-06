"""Standing recovery should not mistake a swinging foot for ground slip."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_cat_curriculum import fixture_contract, fixture_metadata

from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart
from embodied_learning.tasks.cat_foot_guidance import sha256
from embodied_learning.tasks.cat_reward import (
    CatRewardInputs,
    CatRewardParameters,
    cat_reward_terms,
)


def reward_fixture():
    f = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    x = CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(f[k]).clone()
            for k in f.files
            if k.startswith("input_")
        }
    )
    p = CatRewardParameters(
        torch.from_numpy(f["soft_lower"]), torch.from_numpy(f["soft_upper"])
    )
    return x, p


def test_contact_mask_only_applies_in_fully_standing_slots():
    x, p = reward_fixture()
    x.move_flag[:3] = torch.tensor([0.0, 0.1, 1.0])
    x.gait_mask[:3] = 1.0
    x.feet_contact[:3] = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    x.foot_vel[:3] = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]).repeat(
        3, 1, 1
    )
    orig = cat_reward_terms(x, parameters=p)
    candidate = cat_reward_terms(x, parameters=p, standing_contact_slip=True)
    torch.testing.assert_close(orig["foot_slip"][:3], torch.tensor([5.0, 5.0, 5.0]))
    torch.testing.assert_close(
        candidate["foot_slip"][:3], torch.tensor([1.0, 5.0, 5.0])
    )
    for name in orig:
        if name != "foot_slip":
            torch.testing.assert_close(orig[name], candidate[name])


def test_warmstart_allows_only_explicit_hash_bound_slip_change(tmp_path):
    old = fixture_contract(4)
    meta = fixture_metadata(4)
    p = tmp_path / "parent.pt"
    saved = dict(
        field_contract=old,
        curriculum=meta,
        training_scenes=[2],
        actor_state_dict={"w": torch.ones(3)},
        completed_iterations=800,
    )
    torch.save(saved, p)
    target = copy.deepcopy(old)
    target["schema"]["standing_slip_mode"] = "ground_contact_v1"
    dst = {
        **meta,
        "experiment": "trial",
        "protocol_sha256": "new",
        "standing_slip_refinement": dict(
            version="cat_standing_slip_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment=meta["experiment"],
            parent_protocol_sha256=meta["protocol_sha256"],
            before="phase_mask_v1",
            after="ground_contact_v1",
        ),
    }
    _, info = load_actor_warmstart(
        p,
        contract=SimpleNamespace(to_dict=lambda: target),
        curriculum=dst,
        training_scenes=[2],
        device="cpu",
    )
    assert info["contract_changes"]["standing_slip_mode"] == dict(
        before="phase_mask_v1", after="ground_contact_v1"
    )
    assert not info["optimizer_restored"] and not info["critic_restored"]
    bad = copy.deepcopy(dst)
    bad["standing_slip_refinement"]["parent_sha256"] = "wrong"
    with pytest.raises(ValueError):
        load_actor_warmstart(
            p,
            contract=SimpleNamespace(to_dict=lambda: target),
            curriculum=bad,
            training_scenes=[2],
            device="cpu",
        )
    changed = copy.deepcopy(target)
    changed["schema"]["task"]["push"] = False
    with pytest.raises(ValueError):
        load_actor_warmstart(
            p,
            contract=SimpleNamespace(to_dict=lambda: changed),
            curriculum=dst,
            training_scenes=[2],
            device="cpu",
        )


def test_full_resume_rejects_changed_reward_contract():
    from embodied_learning.tasks.cat_curriculum import validate_transfer

    old = fixture_contract(4)
    target = copy.deepcopy(old)
    target["schema"]["standing_slip_mode"] = "ground_contact_v1"
    with pytest.raises(ValueError):
        validate_transfer(old, target, fixture_metadata(4), fixture_metadata(4))


def test_standing_both_feet_contact_keeps_original_penalty():
    x, p = reward_fixture()
    x.move_flag[:] = 0
    x.gait_mask[:] = 1
    x.feet_contact[:] = 1
    orig = cat_reward_terms(x, parameters=p)
    candidate = cat_reward_terms(x, parameters=p, standing_contact_slip=True)
    torch.testing.assert_close(orig["foot_slip"], candidate["foot_slip"])


def test_legacy_control_transfer_records_no_reward_change(tmp_path):
    old = fixture_contract(4)
    meta = fixture_metadata(4)
    p = tmp_path / "parent.pt"
    saved = dict(
        field_contract=old,
        curriculum=meta,
        training_scenes=[2],
        actor_state_dict={"w": torch.ones(3)},
        completed_iterations=800,
    )
    torch.save(saved, p)
    dst = {
        **meta,
        "experiment": "control",
        "protocol_sha256": "fresh",
        "standing_slip_refinement": dict(
            version="cat_standing_slip_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment=meta["experiment"],
            parent_protocol_sha256=meta["protocol_sha256"],
            before="phase_mask_v1",
            after="phase_mask_v1",
        ),
    }
    _, info = load_actor_warmstart(
        p,
        contract=SimpleNamespace(to_dict=lambda: old),
        curriculum=dst,
        training_scenes=[2],
        device="cpu",
    )
    assert info["contract_changes"] == {}
