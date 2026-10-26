"""Run one manually selected CAT stage; evaluation never advances difficulty."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))
from train_cat_curriculum import stages_for, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def run(
    root,
    level,
    output,
    *,
    resume=None,
    warmstart_actor=None,
    smoke=False,
    run_child=None,
    source_manifest=None,
):
    protocol = json.loads((root / "protocol.json").read_text())
    stages = stages_for(root, protocol)
    if not 0 <= level < len(stages):
        raise ValueError("Invalid curriculum stage.")
    if resume is not None and warmstart_actor is not None:
        raise ValueError("Resume and Actor warmstart are mutually exclusive.")
    completed = 0
    if resume is not None:
        import torch

        restored = torch.load(resume, map_location="cpu", weights_only=False)
        if restored["curriculum"]["stage"] != level:
            raise ValueError("Resume must stay in the same stage.")
        completed = restored["completed_iterations"]
        if (
            type(completed) is not int
            or not 0 <= completed < protocol["total_updates"]
        ):
            raise ValueError(
                "Resume checkpoint is outside the remaining stage budget."
            )
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("This stage session requires an unused output directory.")
    status = dict(
        state="running",
        pid=os.getpid(),
        started_at=now(),
        completed_updates=completed,
        resumed_from_updates=completed,
        stage=level,
        progression="manual",
        initialization="resume"
        if resume
        else "actor_only"
        if warmstart_actor
        else "fresh",
    )
    script = Path(__file__).with_name("train_cat_curriculum.py")
    checkpoint, history = resume, []

    def child(arguments, log_path):
        if source_manifest is not None:
            for name, digest in source_manifest.items():
                if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(
                        f"Project source changed during training: {name}"
                    )
        status.update(phase_started_at=now())
        write_json(output / "status.json", status)
        if run_child is not None:
            run_child(arguments, log_path)
            return
        with log_path.open("x") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(script), *arguments, "--root", str(root)]
                + (["--smoke"] if smoke else []),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            status["child_pid"] = process.pid
            while process.poll() is None:
                status["heartbeat_at"] = now()
                write_json(output / "status.json", status)
                time.sleep(15)
            status.pop("child_pid", None)
            if process.returncode:
                raise RuntimeError(f"Child exited {process.returncode}: {log_path}")

    write_json(output / "status.json", status)
    try:
        while completed < protocol["total_updates"]:
            updates = min(
                protocol["chunk_updates"], protocol["total_updates"] - completed
            )
            dest = output / "chunks" / f"update-{completed + updates:04d}"
            dest.mkdir(parents=True, exist_ok=False)
            status.update(phase="train")
            args = [
                "train",
                "--stage",
                str(level),
                "--updates",
                str(updates),
                "--output",
                str(dest),
            ]
            if checkpoint is not None:
                args += ["--resume", str(checkpoint)]
            elif warmstart_actor is not None:
                args += ["--warmstart-actor", str(warmstart_actor)]
            child(args, dest / "train.log")
            done = json.loads((dest / "complete.json").read_text())
            completed += updates
            if done["completed_iterations"] != completed:
                raise ValueError(
                    "Checkpoint progress disagrees with the stage budget."
                )
            checkpoint = Path(done["checkpoint"])
            status.update(
                completed_updates=completed,
                checkpoint=str(checkpoint),
                phase="evaluation",
            )
            evaluation_dir = dest / "evaluation"
            child(
                [
                    "evaluate",
                    "--stage",
                    str(level),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(evaluation_dir),
                    "--seed",
                    str(protocol["gate_seed"]),
                    "--execution-profile",
                    protocol.get("evaluation_profile", "author_playback"),
                ],
                dest / "evaluate.log",
            )
            evaluation = json.loads((evaluation_dir / "evaluation.json").read_text())
            record = dict(
                completed_updates=completed,
                checkpoint_iterations=done["completed_iterations"],
                stage=level,
                rates=evaluation["rates"],
                paper_success_rate=evaluation.get("paper_success_rate"),
                checkpoint=str(checkpoint),
                at=now(),
            )
            history.append(record)
            write_json(output / "history.json", history)
            status["latest_evaluation"] = record
            write_json(output / "status.json", status)
        status.update(
            state="complete",
            completed_at=now(),
            training_last_stage=level,
            next_stage=(
                "Select explicitly and pass --warmstart-actor to a new stage session"
            ),
        )
        write_json(output / "status.json", status)
        return status
    except BaseException as error:
        status.update(state="failed", failed_at=now(), error=str(error))
        write_json(output / "status.json", status)
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--resume", type=Path)
    initial.add_argument("--warmstart-actor", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.stage,
        args.output.resolve(),
        resume=args.resume,
        warmstart_actor=args.warmstart_actor,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
