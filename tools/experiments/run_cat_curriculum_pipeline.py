"""Run one CAT stage manually or opt in to the update-600 curriculum gate."""

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
from embodied_learning.evaluation.cat_terminal_gate import gate_from_evaluations


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
    terminal_gate_seeds=None,
):
    protocol_path = root / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    gated = terminal_gate_seeds is not None
    if gated and (protocol["total_updates"] != 600 or smoke):
        raise ValueError("Formal gate requires 600 updates and full evaluation cohorts")
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
        progression="terminal_20s_v1" if gated else "manual",
        initialization="resume"
        if resume
        else "actor_only"
        if warmstart_actor
        else "fresh",
    )
    script = Path(__file__).with_name("train_cat_curriculum.py")
    checkpoint, history = resume, []

    def child(arguments, log_path):
        if gated and protocol_path.read_bytes() != protocol_bytes:
            raise ValueError("Protocol changed during gated training")
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
            if gated:
                write_json(output / "status.json", status)
                if completed < 600:
                    continue
                reports = []
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                for seed in terminal_gate_seeds:
                    evaluation_dir = dest / f"evaluation-seed-{seed}"
                    child(
                        [
                            "evaluate", "--stage", str(level), "--checkpoint", str(checkpoint),
                            "--output", str(evaluation_dir), "--seed", str(seed),
                            "--execution-profile", "source_train", "--terminal-gate",
                        ],
                        dest / f"evaluate-seed-{seed}.log",
                    )
                    report = json.loads((evaluation_dir / "evaluation.json").read_text())
                    if (
                        report["seed"] != seed or report["training_stage"] != level
                        or report["checkpoint_sha256"] != digest
                        or report["protocol_sha256"] != protocol_sha256
                        or report["training_seed"] != protocol["ppo"]["seed"]
                    ):
                        raise ValueError("Gate report does not match this run's checkpoint or provenance")
                    reports.append(report)
                gate = gate_from_evaluations(reports)
                write_json(output / "gate.json", gate)
                status["gate"] = gate
                break
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
                level + 1 if gated and status["gate"]["passed"] and level + 1 < len(stages)
                else None if gated
                else "Select explicitly and pass --warmstart-actor to a new stage session"
            ),
        )
        write_json(output / "status.json", status)
        return status
    except BaseException as error:
        status.update(state="failed", failed_at=now(), error=str(error))
        write_json(output / "status.json", status)
        traceback.print_exc()
        raise


def run_gated(root, output, *, run_child=None):
    """Start fresh at stage 0, promoting only after the held-out terminal gate."""
    protocol_path = root / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    seeds = protocol.get("terminal_gate_seeds", [])
    if (
        protocol.get("total_updates") != 600
        or type(protocol.get("chunk_updates")) is not int
        or protocol["chunk_updates"] < 1
        or not isinstance(seeds, list) or len(seeds) != 2
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len({protocol["ppo"]["seed"], *seeds}) != 3
    ):
        raise ValueError("Set 600 total updates, positive chunk size and gate seeds distinct from training")
    stages = stages_for(root, protocol)
    if not stages:
        raise ValueError("Gated curriculum requires at least one stage")
    if run_child is None:
        git = ["git", "-C", str(CODE)]
        if subprocess.check_output([*git, "status", "--porcelain"], text=True).strip():
            raise ValueError("Gated training requires a clean committed checkout")
        code_commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    else:
        code_commit = None
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Gated curriculum requires an unused output directory")
    status = dict(
        state="running", started_at=now(), training_seed=protocol["ppo"]["seed"],
        evaluation_seeds=seeds, protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        code_commit=code_commit, gates=[],
    )
    checkpoint = None
    try:
        for level in range(len(stages)):
            status["stage"] = level
            write_json(output / "status.json", status)
            if protocol_path.read_bytes() != protocol_bytes:
                raise ValueError("Protocol changed between gated stages")
            stage = run(
                root, level, output / f"stage-{level:02d}",
                warmstart_actor=checkpoint, run_child=run_child, terminal_gate_seeds=seeds,
            )
            checkpoint = Path(stage["checkpoint"])
            gate = stage["gate"]
            if code_commit is not None and gate["evaluation_code_commit"] != code_commit:
                raise ValueError("Checkout changed during gated training")
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != gate["checkpoint_sha256"]:
                raise ValueError("Gate checkpoint changed before promotion")
            status["gates"].append({**gate, "checkpoint": str(checkpoint)})
            if not gate["passed"]:
                status.update(state="stopped_at_gate", failed_stage=level)
                break
        else:
            status["state"] = "complete"
        status["completed_at"] = now()
        write_json(output / "status.json", status)
        return status
    except BaseException as error:
        status.update(state="failed", failed_at=now(), error=str(error))
        write_json(output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage", type=int)
    mode.add_argument("--auto-gate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--resume", type=Path)
    initial.add_argument("--warmstart-actor", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.auto_gate:
        if args.resume or args.warmstart_actor or args.smoke:
            parser.error("--auto-gate begins at stage 0 with full evaluations")
        result = run_gated(args.root.resolve(), args.output.resolve())
        return 0 if result["state"] == "complete" else 1
    run(
        args.root.resolve(),
        args.stage,
        args.output.resolve(),
        resume=args.resume,
        warmstart_actor=args.warmstart_actor,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    raise SystemExit(main())
