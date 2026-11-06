import importlib.util
import json
import sys
from pathlib import Path


def pipeline_module():
    folder = Path(__file__).parents[1] / "tools/experiments"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(
        "manual_cat_pipeline", folder / "run_cat_curriculum_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_even_100_percent_success_does_not_advance_and_chunks_resume(tmp_path):
    pipeline = pipeline_module()
    from train_cat_curriculum import initialize, write_json

    root = tmp_path / "experiment"
    initialize(root)
    protocol = json.loads((root / "protocol.json").read_text())
    assert "promotion_success_rate" not in protocol
    assert protocol["curriculum_profile"] == "foundation_v1"
    protocol.update(total_updates=2, chunk_updates=1)
    write_json(root / "protocol.json", protocol)
    calls = []

    def child(args, log):
        calls.append(args)
        output = Path(args[args.index("--output") + 1])
        assert args[args.index("--stage") + 1] == "1"
        if args[0] == "train":
            count = sum(a[0] == "train" for a in calls)
            write_json(
                output / "complete.json",
                dict(
                    checkpoint=str(output / "latest.pt"), completed_iterations=count
                ),
            )
        else:
            write_json(
                output / "evaluation.json",
                dict(rates=dict(success=1.0), paper_success_rate=1.0),
            )

    result = pipeline.run(
        root,
        1,
        tmp_path / "stage1",
        warmstart_actor=Path("stage0.pt"),
        run_child=child,
    )
    assert result["state"] == "complete" and result["training_last_stage"] == 1
    assert len(calls) == 4
    assert "--warmstart-actor" in calls[0] and "--resume" not in calls[0]
    assert "--resume" in calls[2] and "--warmstart-actor" not in calls[2]


def test_playback_contract_adaptation_rejects_unrelated_interface_changes():
    import copy

    import pytest
    from test_cat_curriculum import fixture_contract, fixture_metadata

    from embodied_learning.tasks.cat_curriculum import validate_transfer

    parent = fixture_contract(0)
    parent["schema"]["task"] = dict(
        noise_level=1.0,
        random_pd=True,
        random_rfi=True,
        push=True,
        ground=1,
        lateral=1,
        overhead=1,
    )
    actual = copy.deepcopy(parent)
    actual["schema"]["execution_profile"] = "author_playback"
    actual["schema"]["task"].update(
        noise_level=0.0, random_pd=False, random_rfi=False, push=False
    )
    changes = validate_transfer(
        parent, actual, fixture_metadata(0), fixture_metadata(0), evaluation=True
    )
    assert set(changes) == {"execution_profile"}
    actual["schema"]["body_sites"].reverse()
    with pytest.raises(ValueError, match="contract"):
        validate_transfer(
            parent, actual, fixture_metadata(0), fixture_metadata(0), evaluation=True
        )


def test_migration_resume_keeps_global_budget_instead_of_adding_a_new_budget(
    tmp_path,
):
    import torch

    pipeline = pipeline_module()
    from train_cat_curriculum import initialize, write_json

    root = tmp_path / "experiment"
    initialize(root)
    protocol = json.loads((root / "protocol.json").read_text())
    protocol.update(total_updates=5, chunk_updates=2)
    write_json(root / "protocol.json", protocol)
    parent = tmp_path / "parent.pt"
    torch.save(dict(completed_iterations=2, curriculum=dict(stage=0)), parent)
    updates = []

    def child(args, log):
        output = Path(args[args.index("--output") + 1])
        if args[0] == "train":
            count = int(args[args.index("--updates") + 1])
            updates.append(count)
            previous = torch.load(
                Path(args[args.index("--resume") + 1]), weights_only=False
            )
            checkpoint = output / "latest.pt"
            torch.save(
                dict(
                    completed_iterations=previous["completed_iterations"] + count,
                    curriculum=dict(stage=0),
                ),
                checkpoint,
            )
            write_json(
                output / "complete.json",
                dict(
                    checkpoint=str(checkpoint),
                    completed_iterations=previous["completed_iterations"] + count,
                ),
            )
        else:
            write_json(
                output / "evaluation.json",
                dict(rates=dict(success=0.0), paper_success_rate=0.0),
            )

    result = pipeline.run(
        root, 0, tmp_path / "resumed", resume=parent, run_child=child
    )
    assert updates == [2, 1]
    assert result["completed_updates"] == 5
    assert result["resumed_from_updates"] == 2


def test_gated_scheduler_advances_only_after_two_seed_update_600_reports(tmp_path, monkeypatch):
    import hashlib

    pipeline = pipeline_module()
    monkeypatch.setattr(pipeline, "stages_for", lambda *_: [None, None, None])
    root = tmp_path / "experiment"
    protocol = dict(
        total_updates=600, chunk_updates=350, terminal_gate_seeds=[10, 11],
        ppo={"seed": 42},
    )
    pipeline.write_json(root / "protocol.json", protocol)
    protocol_hash = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    calls = []

    def child(args, log):
        calls.append(args)
        read = lambda name: args[args.index(name) + 1]
        stage = int(read("--stage"))
        out = Path(read("--output"))
        if args[0] == "train":
            checkpoint = out / "latest.pt"
            checkpoint.write_bytes(f"stage={stage} chunk={out.name}".encode())
            pipeline.write_json(out / "complete.json", dict(
                checkpoint=str(checkpoint), completed_iterations=int(out.name.split("-")[1]),
            ))
        else:
            assert "--terminal-gate" in args
            assert read("--execution-profile") == "source_train"
            checkpoint = Path(read("--checkpoint"))
            assert checkpoint.parent.name == "update-0600"
            seed = int(read("--seed"))
            pipeline.write_json(out / "evaluation.json", dict(
                seed=seed, training_seed=42, training_stage=stage, evaluation_stage=stage,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                completed_updates=600, execution_profile="source_train",
                contact_sampling="physics_substep", evaluation_code_commit="a" * 40,
                protocol_sha256=protocol_hash,
                terminal_v1=dict(
                    protocol="cat_terminal_20s_v1", episodes=128,
                    success_count=113 if stage == 0 or seed == 10 else 112,
                ),
            ))

    output = tmp_path / "automatic"
    result = pipeline.run_gated(root, output, run_child=child)
    assert result["state"] == "stopped_at_gate"
    assert result["failed_stage"] == 1
    assert [x["success_count"] for x in result["gates"]] == [226, 225]
    assert [(a[0], a[a.index("--stage") + 1]) for a in calls] == [
        ("train", "0"), ("train", "0"), ("evaluate", "0"), ("evaluate", "0"),
        ("train", "1"), ("train", "1"), ("evaluate", "1"), ("evaluate", "1"),
    ]
    assert "--warmstart-actor" not in calls[0]
    assert calls[4][calls[4].index("--warmstart-actor") + 1].endswith(
        "stage-00/chunks/update-0600/latest.pt"
    )
    assert "--resume" in calls[1] and "--resume" in calls[5]
    assert (output / "stage-01/gate.json").is_file()
    assert not (output / "stage-02").exists()
