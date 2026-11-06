import torch

from test_cat_standing_slip import reward_fixture
from embodied_learning.tasks.cat_reward import cat_reward_terms


def test_standing_balance_only_is_released():
    x, p = reward_fixture()
    x.move_flag[:] = 1.0
    x.move_flag[:3] = torch.tensor([0.0, 0.1, 1.0])
    x.pelvis_subtree_com[:3, :2] += 0.4
    old = cat_reward_terms(x, parameters=p)
    new = cat_reward_terms(x, parameters=p, standing_balance_free=True)
    assert old['foot_balance'][0] > 0
    assert new['foot_balance'][0] == 0
    torch.testing.assert_close(old['foot_balance'][1:], new['foot_balance'][1:])
    for k in old:
        if k != 'foot_balance':
            torch.testing.assert_close(old[k], new[k])


import copy
from types import SimpleNamespace
import pytest
from test_cat_curriculum import fixture_contract, fixture_metadata
from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart
from embodied_learning.tasks.cat_foot_guidance import sha256

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
    target["schema"]["standing_balance_mode"] = "stand_free_v1"
    dst = {
        **meta,
        "experiment": "trial",
        "protocol_sha256": "new",
        "standing_balance_refinement": dict(
            version="cat_standing_balance_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment=meta["experiment"],
            parent_protocol_sha256=meta["protocol_sha256"],
            before="original_v1",
            after="stand_free_v1",
        ),
    }
    _, info = load_actor_warmstart(
        p,
        contract=SimpleNamespace(to_dict=lambda: target),
        curriculum=dst,
        training_scenes=[2],
        device="cpu",
    )
    assert info["contract_changes"]["standing_balance_mode"] == dict(
        before="original_v1", after="stand_free_v1"
    )
    assert not info["optimizer_restored"] and not info["critic_restored"]
    bad = copy.deepcopy(dst)
    bad["standing_balance_refinement"]["parent_sha256"] = "wrong"
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
    target["schema"]["standing_balance_mode"] = "stand_free_v1"
    with pytest.raises(ValueError):
        validate_transfer(old, target, fixture_metadata(4), fixture_metadata(4))


