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
