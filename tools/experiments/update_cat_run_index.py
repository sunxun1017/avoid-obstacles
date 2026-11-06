"""Add verified CAT replay cohorts to the workspace experiment index."""

import argparse
import json
from pathlib import Path

MANIFEST_KEYS = (
    "code_commit", "checkpoint_sha256", "evaluation_seed", "config_sha256",
    "protocol_sha256", "route",
)


def update(index, batch, change_id, training_seed, workspace):
    original = json.loads(index.read_text()) if index.exists() else []
    rows = [row for row in original if row["change_id"] != change_id]
    for path in sorted(batch.rglob("evaluation.json")):
        cohort = path.parent
        manifest_path = cohort / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        terminal = json.loads(path.read_text())["terminal_v1"]
        episodes = [
            json.loads(line)["terminal_v1"]
            for line in (cohort / "episodes.jsonl").read_text().splitlines()
        ]
        if (
            len(episodes) != manifest["evaluation_episodes"]
            or len(episodes) != terminal["episodes"]
            or sum(e["success"] for e in episodes) != terminal["success_count"]
            or sum(e["strict_success"] for e in episodes) != terminal["strict_success_count"]
        ):
            raise ValueError(f"Summary disagrees with episode records: {cohort}")
        relative = cohort.relative_to(batch)
        rows.append(dict(
            run_id=(Path(batch.name) / relative).as_posix(),
            change_id=change_id, training_seed=training_seed,
            **{key: manifest[key] for key in MANIFEST_KEYS},
            disturbance=manifest.get("disturbance"),
            **terminal, metrics_path=path.relative_to(workspace).as_posix(),
        ))
    if not any(row["change_id"] == change_id for row in rows):
        raise ValueError(f"No verified cohorts in {batch}")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate run ID")
    temp = index.with_suffix(".tmp")
    temp.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    temp.replace(index)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--change-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    rows = update(
        args.index.resolve(), args.batch.resolve(), args.change_id,
        args.training_seed, args.workspace.resolve(),
    )
    print(f"{len(rows)} indexed runs")


if __name__ == "__main__":
    main()
