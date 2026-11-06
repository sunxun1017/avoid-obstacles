"""Only the explicit online inference boundary may substitute field identity."""

from dataclasses import replace

import pytest
import torch

from embodied_learning.policies.cat_online import (
    load_online_policy,
    online_inference_contract,
)
from embodied_learning.policies.field_policy import FieldStudent
from embodied_learning.protocols.imitation import PolicyContract


def contracts():
    base = dict(
        task_profile="cat_source_v1",
        field_cache_identity="verified-cache",
        robot_model_identity="robot",
        scene_hash="scene",
        action_transform="latent_tanh_v1",
        normalization=False,
    )
    saved = PolicyContract(base, 162, 250, 12, (16,), normalization=False)
    schema = {
        **base,
        "field_cache_identity": None,
        "robot_model_identity": "with-camera",
        "online_perception": dict(
            version="cat_online_v1",
            source="depth",
            sensorless_model_identity="robot",
        ),
    }
    return saved, replace(saved, schema=schema)


def test_explicit_online_inference_allows_only_field_and_sensor_substitution(
    tmp_path,
):
    saved, online = contracts()
    assert online_inference_contract(online, saved) == saved
    with pytest.raises(ValueError, match="contract"):
        online.require(saved)
    for key, value in [
        ("scene_hash", "other"),
        ("action_transform", "clip"),
        ("robot_model_identity", "other"),
    ]:
        with pytest.raises(ValueError):
            online_inference_contract(
                online, replace(saved, schema={**saved.schema, key: value})
            )
    with pytest.raises(ValueError):
        online_inference_contract(online, replace(saved, actor_dim=173))
    with pytest.raises(ValueError):
        online_inference_contract(online, replace(saved, normalization=True))
    student = FieldStudent(saved, "cpu")
    path = tmp_path / "student.pt"
    torch.save(
        dict(field_contract=saved.to_dict(), student=student.state_dict()), path
    )
    restored = load_online_policy(path, online_contract=online, device="cpu")
    assert not restored.training
    for name, value in restored.state_dict().items():
        assert torch.equal(value, student.state_dict()[name])


def test_cli_requires_checkpoint_or_explicit_smoke():
    from embodied_learning.cli.cat_online import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args([])
    args = _parser().parse_args(["--sensor", "depth", "--smoke", "--steps", "5"])
    assert args.sensor == "depth" and args.smoke and args.steps == 5


def test_online_environment_cannot_enter_training(tmp_path):
    from types import SimpleNamespace

    from embodied_learning.algorithms.dagger import train_dagger
    from embodied_learning.algorithms.ppo import train_specialist

    env = SimpleNamespace(inference_only=True)
    with pytest.raises(ValueError, match="inference"):
        train_specialist(env, None, tmp_path)
    with pytest.raises(ValueError, match="inference"):
        train_dagger(env, None, {}, None, tmp_path)
