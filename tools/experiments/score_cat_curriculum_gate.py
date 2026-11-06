"""Score two 128-episode reports for the same CAT stage at update 600."""

import argparse
import json
from pathlib import Path

from embodied_learning.evaluation.cat_terminal_gate import gate_from_evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=2, type=Path)
    args = parser.parse_args()
    result = gate_from_evaluations([json.loads(path.read_text()) for path in args.reports])
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
