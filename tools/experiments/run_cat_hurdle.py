"""Launch the project hurdle pipeline without creating per-run code copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_cat_curriculum_pipeline import run as run_pipeline
from train_cat_curriculum import CODE, evaluate, write_json


def source_manifest() -> dict[str, str]:
    """Record the source used by each training/evaluation child process."""
    return {
        str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (CODE / "src", CODE / "tools")
        for p in sorted(folder.rglob("*.py"))
    }


def resolve_gpu(selector: str) -> str:
    """Resolve nvidia-smi physical indices to stable CUDA UUIDs."""
    result = subprocess.check_output(
        ["nvidia-smi", "-i", selector, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    )
    devices = [line.strip() for line in result.splitlines() if line.strip()]
    if len(devices) != 1 or not devices[0].startswith("GPU-"):
        raise ValueError("Select exactly one physical GPU index or UUID.")
    return devices[0]


def run_experiment(
    root: Path, output: Path, *, resume: Path | None = None, smoke: bool = False
) -> dict:
    """Run manual stage1 training and its final source-gait no-push evaluation."""
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol.get("recipe") != "feetgf_progress_stage1_v1":
        raise ValueError("Use init --recipe feetgf_progress_stage1_v1 first.")
    manifest = source_manifest()
    provenance = root / "project-code-manifest.json"
    if provenance.exists():
        if json.loads(provenance.read_text()) != manifest:
            raise ValueError("Project source differs from the original run.")
    elif resume is not None:
        raise ValueError("Resume requires the original project source manifest.")
    else:
        write_json(provenance, manifest)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training requires an unused session directory.")
    write_json(root / "source-manifests" / f"{output.name}.json", manifest)
    state = dict(
        state="running",
        pid=os.getpid(),
        phase="training",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        project_code=str(CODE),
        output=str(output),
    )
    write_json(root / "driver-status.json", state)
    try:
        result = run_pipeline(
            root,
            1,
            output,
            resume=resume,
            smoke=smoke,
            warmstart_actor=None if resume else Path(protocol["parent_checkpoint"]),
            source_manifest=manifest,
        )
        state.update(
            phase="final_no_push_evaluation", checkpoint=result["checkpoint"]
        )
        write_json(root / "driver-status.json", state)
        for name, digest in manifest.items():
            if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Project source changed during training: {name}")
        evaluate(
            root,
            1,
            Path(result["checkpoint"]),
            output / "final-no-push-evaluation",
            protocol.get("final_seed", 20261026),
            execution_profile="source_train",
            smoke=smoke,
            disable_push=True,
        )
        state.update(
            state="complete",
            phase="complete",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(root / "driver-status.json", state)
        return state
    except BaseException as error:
        state.update(
            state="failed",
            error=str(error),
            failed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(root / "driver-status.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--gpu", help="Physical GPU index or UUID;otherwise inherit CUDA visibility."
    )
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "stage1-session"
    gpu = resolve_gpu(args.gpu) if args.gpu is not None else None
    if not (root / "stages/1/assets.json").exists():
        parser.error("Run train_cat_curriculum.py prepare --stage 1 first.")
    if args.foreground:
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ.setdefault("MUJOCO_GL", "egl")
        run_experiment(root, output, resume=args.resume, smoke=args.smoke)
        return
    if output.exists() and any(output.iterdir()):
        parser.error(
            "Select an unused output directory;use --resume for the checkpoint."
        )
    runtime = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    if gpu is not None:
        runtime["CUDA_VISIBLE_DEVICES"] = gpu
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--output",
        str(output),
        "--foreground",
    ]
    if args.resume:
        command += ["--resume", str(args.resume.resolve())]
    if args.smoke:
        command += ["--smoke"]
    # Serialize launch reservations. The persisted child PID also protects the
    # interval before the driver writes its first status, and permits restart
    # into a new session after a previous process has exited.
    import fcntl

    with (root / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = root / "launch.json"
        if previous.exists():
            old = json.loads(previous.read_text())
            try:
                os.kill(old["pid"], 0)
            except ProcessLookupError:
                history = root / "launch-history"
                history.mkdir(exist_ok=True)
                write_json(history / f"{time.time_ns()}.json", old)
            else:
                parser.error(f"Training process {old['pid']} is still active.")
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"{output.name}.log").open("x") as log:
            child = subprocess.Popen(
                command,
                env=runtime,
                cwd=CODE,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record = dict(
            pid=child.pid,
            project_code=str(CODE),
            output=str(output),
            cuda_visible_devices=runtime.get("CUDA_VISIBLE_DEVICES"),
            started_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(previous, record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
