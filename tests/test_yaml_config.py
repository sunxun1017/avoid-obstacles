"""Commented YAML configuration and legacy JSON compatibility."""

import json
from pathlib import Path

import pytest
import yaml

from embodied_learning.data.field_scenes import load_scene_manifest


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_commented_yaml_preserves_scene_identity(tmp_path, suffix):
    source = Path("configs/experiments/g1_cat_source.json")
    payload = json.loads(source.read_text())
    path = tmp_path / ("experiment" + suffix)
    path.write_text("# 中文注释不会改变训练场景\n" + yaml.safe_dump(payload))
    assert load_scene_manifest(path) == load_scene_manifest(source)
    assert load_scene_manifest(path).identity == load_scene_manifest(source).identity


def test_yaml_rejects_executable_tags(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:builtins.print [unsafe]")
    with pytest.raises(ValueError):
        load_scene_manifest(path)


@pytest.mark.parametrize("name", ["g1_cat_source", "g1_field"])
def test_migrated_yaml_preserves_all_settings(name):
    from embodied_learning.data.config import load_experiment_config

    folder = Path("configs/experiments")
    assert load_experiment_config(folder / f"{name}.yaml") == json.loads(
        (folder / f"{name}.json").read_text()
    )


def test_cli_reads_yaml_settings(tmp_path):
    from embodied_learning.cli.field_training import main

    payload = json.loads(Path("configs/experiments/g1_cat_source.json").read_text())
    payload["robot"] = "unsupported_robot"
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="unitree_g1"):
        main(["prepare", "--config", str(path)])


@pytest.mark.parametrize("value", ["", "- a\n- b\n", "42"])
def test_yaml_requires_mapping(tmp_path, value):
    from embodied_learning.data.config import load_experiment_config

    path = tmp_path / "experiment.yaml"
    path.write_text(value)
    with pytest.raises(ValueError, match="mapping"):
        load_experiment_config(path)
