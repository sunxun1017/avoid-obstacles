"""Portable CLI preflight must fail before constructing a simulator."""

import json
import os
import subprocess
import sys

import pytest

from embodied_learning.cli.field_training import main, read_teachers


def test_cli_help_without_simulator():
    code = """
import sys
sys.modules['mjlab'] = None
sys.modules['torch'] = None
from embodied_learning.cli.field_training import main
main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert "prepare" in result.stdout and "dagger" in result.stdout


def test_missing_teacher_fails_early(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mjlab", None)
    manifest = tmp_path / "teachers.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="Missing teacher"):
        main(
            [
                "dagger",
                "--config",
                "configs/experiments/g1_field.json",
                "--output-dir",
                str(tmp_path),
                "--teachers",
                str(manifest),
            ]
        )


def test_invalid_config_fails_early(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mjlab", None)
    config = tmp_path / "bad.json"
    config.write_text('{"scenes": []}')
    with pytest.raises(ValueError):
        main(
            [
                "ppo",
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path),
                "--scene",
                "wall_detour",
            ]
        )


def test_relative_paths_portable(tmp_path, monkeypatch):
    folder = tmp_path / "portable"
    folder.mkdir()
    checkpoint = folder / "model.pt"
    checkpoint.write_bytes(b"not-loaded-here")
    manifest = folder / "teachers.json"
    manifest.write_text(json.dumps({"wall": "model.pt"}))
    monkeypatch.chdir(tmp_path)
    assert read_teachers(manifest, ("wall",)) == {"wall": checkpoint.resolve()}


def test_manifest_action_schema_checks(tmp_path):
    import torch

    from embodied_learning.cli.field_training import preflight_teachers

    manifest = {"wall_detour": tmp_path / "bad.pt"}
    torch.save(
        {"field_contract": {"schema": {"scene_hash": "bad"}}},
        manifest["wall_detour"],
    )
    with pytest.raises(ValueError, match="contract"):
        preflight_teachers(
            manifest, scene_hash="expected", scene_names=("wall_detour",)
        )
