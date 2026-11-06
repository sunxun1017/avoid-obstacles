"""Load experiment mappings from commented YAML or legacy JSON."""

import json
from pathlib import Path
from typing import Any


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Read a UTF-8 mapping; YAML uses safe_load, never executable tags."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        try:
            values = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML configuration {path}: {error}"
            ) from error
    elif path.suffix.lower() == ".json":
        values = json.loads(text)
    else:
        raise ValueError("Experiment configuration must be .yaml, .yml or .json.")
    if not isinstance(values, dict):
        raise ValueError("Experiment configuration must contain a mapping.")
    return values
