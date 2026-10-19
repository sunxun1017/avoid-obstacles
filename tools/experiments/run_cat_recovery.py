"""Run the bounded CAT recovery experiment with immutable parents and stage logs.

Select a compatible GPU externally with CUDA_VISIBLE_DEVICES. This driver uses
the existing environment and cache; it neither installs packages nor stops jobs.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--wall-checkpoint", type=Path, required=True)
    parser.add_argument("--hurdle-checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config = repo / "configs/experiments/g1_cat_recovery.yaml"
    parents = dict(
        wall_detour=args.wall_checkpoint.resolve(),
        hurdle_crouch=args.hurdle_checkpoint.resolve(),
    )
    fields = args.fields.resolve()
    if min(args.iterations, args.num_envs, args.episodes) < 1:
        parser.error("Iterations, environments and episodes must be positive.")
    if not (fields / "manifest.json").is_file() or any(
        not path.is_file() for path in parents.values()
    ):
        parser.error("Existing parent checkpoints and field cache are required.")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        started_at=timestamp(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ),
        config=str(config),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        fields=str(fields),
        iterations_per_scene=args.iterations,
        training_envs=args.num_envs,
        evaluation_episodes_per_scene=args.episodes,
        training_seed=42,
        evaluation_seed=20261019,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        python=sys.executable,
        parents={
            scene: dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            for scene, path in parents.items()
        },
    )
    write_json(run / "metadata.json", metadata)
    status = dict(state="running", completed=[], started_at=metadata["started_at"])

    def execute(scene, stage, command):
        status.update(scene=scene, stage=stage, updated_at=timestamp())
        write_json(run / "status.json", status)
        print(json.dumps(status), flush=True)
        with (run / f"{scene}-{stage}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=repo, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            status.update(
                state="failed", returncode=result.returncode, updated_at=timestamp()
            )
            write_json(run / "status.json", status)
            raise SystemExit(result.returncode)
        status["completed"].append(f"{scene}/{stage}")

    for scene, parent in parents.items():
        output = run / scene
        output.mkdir()
        (output / "fields").symlink_to(fields, target_is_directory=True)
        execute(
            scene,
            "train",
            [
                sys.executable,
                "-u",
                "-m",
                "embodied_learning.cli.field_training",
                "ppo",
                "--config",
                str(config),
                "--output-dir",
                str(output),
                "--scene",
                scene,
                "--num-envs",
                str(args.num_envs),
                "--iterations",
                str(args.iterations),
                "--seed",
                "42",
                "--warmstart",
                str(parent),
            ],
        )
        checkpoint = output / "specialists" / scene / "latest.pt"
        execute(
            scene,
            "evaluate",
            [
                sys.executable,
                "-u",
                str(repo / "tools/evaluate_cat_checkpoint.py"),
                "--run-dir",
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(config),
                "--scene",
                scene,
                "--result-dir",
                str(output / "evaluation"),
                "--episodes",
                str(args.episodes),
                "--num-envs",
                str(args.episodes),
                "--seed",
                "20261019",
            ],
        )
    status.update(state="complete", updated_at=timestamp())
    write_json(run / "status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
