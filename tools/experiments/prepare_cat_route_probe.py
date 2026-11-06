"""Build independent fixed and empty CAT fields for approach-distance probes."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.evaluation.cat_route_probe import route_goal_ids
from embodied_learning.tasks.cat_foot_guidance import prepare_foot_cache
from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_config", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    stage = args.output / "stages/4"
    stage.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.fixed_config.read_text())
    hurdle = next(scene for scene in source["scenes"] if scene["id"] == "hurdle_crouch")
    lanes = dict.fromkeys(tuple(goal[1:]) for goal in hurdle["goals"])
    hurdle["goals"] = [(x, *lane) for lane in lanes for x in (1.15, 1.85)]
    route_goal_ids(hurdle["goals"], far=False)
    config = stage / "config.json"
    config.write_text(json.dumps(source, indent=2) + "\n")
    fixed = load_scene_manifest(config)
    empty = replace(fixed, scenes=tuple(replace(scene, boxes=()) for scene in fixed.scenes))
    profile = {"version": "cat_source_v1"}
    fixed_dir = args.output / "fields"
    empty_dir = args.output / "empty-fields"
    foot_dir = args.output / "foot-fields"
    prepare_fields(fixed, fixed_dir, source_profile=profile)
    prepare_fields(empty, empty_dir, source_profile=profile)
    cache = load_field_cache(fixed_dir / "manifest.json", fixed, device="cpu", source_profile=profile)
    protocol = json.loads(args.protocol.read_text())
    prepare_foot_cache(cache, fixed, foot_dir, radius=protocol["foot_radius_m"])
    (stage / "assets.json").write_text(json.dumps(dict(
        fields=str(fixed_dir.resolve()), foot_fields=str(foot_dir.resolve()),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        original_config_sha256=hashlib.sha256(args.fixed_config.read_bytes()).hexdigest(),
        fixed_scene_identity=fixed.identity, empty_scene_identity=empty.identity,
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
