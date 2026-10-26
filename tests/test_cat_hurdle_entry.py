"""Project entry reproduces the declared single-reward hurdle experiment."""

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_cat_curriculum import fixture_contract, fixture_metadata

from embodied_learning.algorithms.ppo import PpoConfig
from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart
from embodied_learning.robots.g1 import CAT_JOINT_NAMES, CAT_KD, CAT_KP
from embodied_learning.simulators.mjlab.cat_control import (
    CatAction,
    CatActionCfg,
    CatJointController,
)
from embodied_learning.tasks.cat_foot_guidance import sha256

PROFILE = "root_displacement_0p5s_v1"


@pytest.fixture
def parent(tmp_path):
    contract = fixture_contract(1)
    contract.update(action_dim=12)
    contract["schema"].update(
        controller_pd=dict(
            version="cat_ankle_pd30_2_v1",
            joint_names=list(CAT_JOINT_NAMES),
            kp=list(CAT_KP),
            kd=list(CAT_KD),
        ),
        reward_smoothness_action_weight=-0.001,
        rfi_amplitude_multiplier=0.375,
        source_interpolation="xyz_weights_v1",
    )
    contract["schema"]["task"].update(
        noise_level=1.0,
        random_pd=True,
        random_rfi=True,
        push=True,
        command_deadband_m_s=0.01,
        arrival_control="root_linear_v1",
    )
    saved = dict(
        field_contract=contract,
        curriculum=fixture_metadata(1),
        training_scenes=["hurdle_crouch"],
        actor_state_dict={"weights": torch.ones(3)},
        completed_iterations=611,
        ppo_config=asdict(
            PpoConfig(
                rollout_steps=160,
                source_unroll_steps=20,
                optimization_profile="brax_0_12_3_v1",
            )
        ),
    )
    path = tmp_path / "parent.pt"
    torch.save(saved, path)
    return path, saved


def declaration(path, saved):
    return dict(
        version="cat_feetgf_progress_actor_transfer_v1",
        parent_sha256=sha256(path),
        parent_experiment=saved["curriculum"]["experiment"],
        parent_protocol_sha256=saved["curriculum"]["protocol_sha256"],
        before=None,
        after=PROFILE,
    )


def test_gate_transfer_works_without_any_historical_run_directory(parent):
    path, saved = parent
    target = copy.deepcopy(saved["field_contract"])
    target["schema"]["task"]["feetgf_progress_gate"] = PROFILE
    metadata = {
        **saved["curriculum"],
        "experiment": "gated",
        "protocol_sha256": "new",
        "feetgf_progress_refinement": declaration(path, saved),
    }
    actual, info = load_actor_warmstart(
        path,
        contract=SimpleNamespace(to_dict=lambda: target),
        curriculum=metadata,
        training_scenes=saved["training_scenes"],
        device="cpu",
    )
    assert info["contract_changes"] == {
        "task.feetgf_progress_gate": {"before": None, "after": PROFILE}
    }
    assert info["actor_restored"] and not info["critic_restored"]
    assert not info["optimizer_restored"]
    assert "feetgf_progress_gate" not in actual["field_contract"]["schema"]["task"]


@pytest.mark.parametrize(
    "change", ["sha", "pd", "rfi", "action", "robot", "stage", "missing"]
)
def test_gate_transfer_rejects_unrelated_changes(parent, change):
    path, saved = parent
    target = copy.deepcopy(saved["field_contract"])
    target["schema"]["task"]["feetgf_progress_gate"] = PROFILE
    meta = {
        **saved["curriculum"],
        "experiment": "gated",
        "protocol_sha256": "new",
        "feetgf_progress_refinement": declaration(path, saved),
    }
    if change == "sha":
        meta["feetgf_progress_refinement"]["parent_sha256"] = "0" * 64
    if change == "pd":
        target["schema"]["controller_pd"]["kp"][4] = 80.0
    if change == "rfi":
        target["schema"]["rfi_amplitude_multiplier"] = 1.0
    if change == "action":
        target["action_dim"] = 6
    if change == "robot":
        meta["robot_fingerprint"] = "changed"
    if change == "stage":
        meta["stage"] = 2
    if change == "missing":
        meta.pop("feetgf_progress_refinement")
    with pytest.raises(ValueError):
        load_actor_warmstart(
            path,
            contract=SimpleNamespace(to_dict=lambda: target),
            curriculum=meta,
            training_scenes=saved["training_scenes"],
            device="cpu",
        )


def training_module():
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools/experiments"))
    import train_cat_curriculum

    return train_cat_curriculum


def test_generic_init_builds_portable_gated_recipe_from_declared_parent(
    parent, tmp_path
):
    path, saved = parent
    training = training_module()
    root = tmp_path / "new-run"
    training.initialize(
        root, recipe="feetgf_progress_stage1_v1", parent_checkpoint=path
    )
    protocol = json.loads((root / "protocol.json").read_text())
    stages = training.stages_for(root, protocol)
    settings = stages[1]
    assert settings["task"]["feetgf_progress_gate"] == PROFILE
    assert "training_initialization" not in settings["task"]
    assert settings["task"]["push"] and settings["task"]["noise_level"] == 1.0
    bar = settings["scenes"][2]["boxes"][0]
    assert bar["half_size"] == [0.04, 0.5, 0.02]
    assert (
        protocol["controller_pd"]
        == saved["field_contract"]["schema"]["controller_pd"]
    )
    assert protocol["rfi_amplitude_multiplier"] == 0.375
    assert protocol["reward_smoothness_action_weight"] == -0.001
    assert protocol["evaluation_profile"] == "source_train"
    assert protocol["training_envs"] == 4096 and protocol["total_updates"] == 611
    assert protocol["parent_checkpoint"] == str(path.resolve())
    assert protocol["feetgf_progress_refinement"] == declaration(path, saved)


def test_invalid_recipe_parent_does_not_create_run(parent, tmp_path):
    path, saved = parent
    saved["curriculum"]["stage"] = 0
    torch.save(saved, path)
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="stage"):
        training_module().initialize(
            root, recipe="feetgf_progress_stage1_v1", parent_checkpoint=path
        )
    assert not root.exists()


def test_recipe_accepts_historical_numeric_scene_ids(parent, tmp_path):
    path, saved = parent
    saved["training_scenes"] = [2]
    saved["field_contract"]["schema"]["scene_names"] = [
        "wall_detour",
        "passage",
        "hurdle_crouch",
    ]
    torch.save(saved, path)
    root = tmp_path / "numeric-parent"
    training_module().initialize(
        root, recipe="feetgf_progress_stage1_v1", parent_checkpoint=path
    )
    assert json.loads((root / "protocol.json").read_text())[
        "parent_checkpoint"
    ] == str(path)


@pytest.mark.parametrize("ids", [[0], [7], [-1], [True]])
def test_recipe_rejects_other_or_invalid_scene_ids(parent, tmp_path, ids):
    path, saved = parent
    saved["training_scenes"] = ids
    saved["field_contract"]["schema"]["scene_names"] = [
        "wall_detour",
        "passage",
        "hurdle_crouch",
    ]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="scene"):
        training_module().initialize(
            tmp_path / "bad-scenes",
            recipe="feetgf_progress_stage1_v1",
            parent_checkpoint=path,
        )


def test_physical_gpu_index_is_resolved_before_setting_cuda_visibility(monkeypatch):
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools/experiments"))
    import run_cat_hurdle as driver

    def query(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "-i",
            "5",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ]
        return "GPU-physical-five\n"

    monkeypatch.setattr(driver.subprocess, "check_output", query)
    assert driver.resolve_gpu("5") == "GPU-physical-five"


def make_action(multiplier):
    action = CatAction.__new__(CatAction)
    action._env = SimpleNamespace(num_envs=3, device="cpu")
    action.cfg = CatActionCfg(
        entity_name="robot", rfi_amplitude_multiplier=multiplier
    )
    action.controller = CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), 3, "cpu"
    )
    return action


def test_rfi_attenuation_survives_repeated_and_partial_resets():
    original, attenuated = make_action(1.0), make_action(0.375)
    for ids in [None, torch.tensor([0, 2]), None]:
        torch.manual_seed(9)
        original.reset(ids)
        torch.manual_seed(9)
        attenuated.reset(ids)
        torch.testing.assert_close(
            attenuated.controller.rfi_scale, original.controller.rfi_scale * 0.375
        )
        torch.testing.assert_close(
            attenuated.controller.kp_scale, original.controller.kp_scale
        )
        torch.testing.assert_close(
            attenuated.controller.kd_scale, original.controller.kd_scale
        )


@pytest.mark.parametrize("value", [-1.0, 1.1, float("nan"), True])
def test_invalid_rfi_attenuation_is_rejected(value):
    with pytest.raises(ValueError, match="RFI"):
        CatActionCfg(entity_name="robot", rfi_amplitude_multiplier=value)


def test_no_push_evaluation_restores_options_and_consumes_original_draws():
    from embodied_learning.tasks.cat_hurdle import evaluation_push_mode

    state = SimpleNamespace(
        options={"push": True}, push_interval=torch.tensor([50, 100])
    )
    calls = []

    def original(actions):
        calls.append(actions)

    state.advance_before_physics = original
    env = SimpleNamespace(
        cat_state=state,
        device="cpu",
        unwrapped=SimpleNamespace(episode_length_buf=torch.tensor([49, 7])),
    )
    torch.manual_seed(8)
    torch.rand(1)
    torch.rand(1)
    expected = torch.rand(3)
    torch.manual_seed(8)
    with evaluation_push_mode(env, True):
        assert not state.options["push"]
        state.advance_before_physics("action")
        torch.testing.assert_close(torch.rand(3), expected)
    assert (
        state.options == {"push": True} and state.advance_before_physics is original
    )
    assert calls == ["action"]


def test_repository_driver_records_success_and_failure_without_run_code_copy(
    tmp_path,
    monkeypatch,
):
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools/experiments"))
    import run_cat_hurdle as driver

    root = tmp_path / "run"
    root.mkdir()
    (root / "protocol.json").write_text(
        json.dumps(
            {"recipe": "feetgf_progress_stage1_v1", "parent_checkpoint": "parent.pt"}
        )
    )

    def pipeline(root, level, output, **kw):
        output.mkdir()
        return {"checkpoint": str(output / "latest.pt"), "state": "complete"}

    def evaluate(root, level, checkpoint, output, seed, **kw):
        output.mkdir()
        (output / "evaluation.json").write_text(
            json.dumps({"rates": {"success": 0.75}})
        )

    monkeypatch.setattr(driver, "run_pipeline", pipeline)
    monkeypatch.setattr(driver, "evaluate", evaluate)
    driver.run_experiment(root, root / "session")
    status = json.loads((root / "driver-status.json").read_text())
    assert status["state"] == "complete"
    assert (
        json.loads(
            (root / "session/final-no-push-evaluation/evaluation.json").read_text()
        )["rates"]["success"]
        == 0.75
    )
    assert not (root / "code").exists()

    def failed(*args, **kw):
        raise RuntimeError("training failed")

    monkeypatch.setattr(driver, "run_pipeline", failed)
    with pytest.raises(RuntimeError, match="training failed"):
        driver.run_experiment(root, root / "failure")
    assert json.loads((root / "driver-status.json").read_text())["state"] == "failed"


def test_resume_rejects_changed_source_without_overwriting_provenance(
    tmp_path, monkeypatch
):
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools/experiments"))
    import run_cat_hurdle as driver

    root = tmp_path / "run"
    root.mkdir()
    (root / "protocol.json").write_text(
        json.dumps(
            {"recipe": "feetgf_progress_stage1_v1", "parent_checkpoint": "parent.pt"}
        )
    )
    old = {"src/module.py": "old"}
    (root / "project-code-manifest.json").write_text(json.dumps(old))
    monkeypatch.setattr(driver, "source_manifest", lambda: {"src/module.py": "new"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Changed source reached training.")

    monkeypatch.setattr(driver, "run_pipeline", forbidden)
    with pytest.raises(ValueError, match="source"):
        driver.run_experiment(root, root / "resume", resume=Path("checkpoint.pt"))
    assert json.loads((root / "project-code-manifest.json").read_text()) == old
    assert not (root / "resume").exists()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native factory needs CUDA."
)
def test_native_factory_contract_tracks_rfi_attenuation(tmp_path):
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    manifest = load_scene_manifest(Path("configs/experiments/g1_cat_source.yaml"))
    profile = {"version": "cat_source_v1"}
    prepare_fields(manifest, tmp_path, source_profile=profile)
    bank = load_field_cache(
        tmp_path / "manifest.json", manifest, device="cuda:0", source_profile=profile
    )
    for multiplier in [1.0, 0.375]:
        env = make_cat_env(
            manifest,
            bank,
            scene_ids=["hurdle_crouch"],
            num_envs=2,
            device="cuda:0",
            seed=42,
            options={},
            rfi_amplitude_multiplier=multiplier,
        )
        try:
            if multiplier == 1:
                assert "rfi_amplitude_multiplier" not in env.schema
            else:
                assert env.schema["rfi_amplitude_multiplier"] == 0.375
            env.reset()
            c = env.unwrapped.cat_controller
            ratio = c.rfi_scale / c.limits
            assert float(ratio.max()) <= 0.15 * multiplier + 1e-6
        finally:
            env.close()
