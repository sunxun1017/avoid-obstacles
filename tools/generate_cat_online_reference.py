"""Extract numerical deployment functions from pinned CAT source, without ROS.

Supply a directory containing the unmodified pf.py and octomap_bridge.py from
commit 866ba392f1c1e84b92ad75fa66550f26e8af8e48. Runtime needs neither file.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import skfmm
from scipy.ndimage import binary_closing, convolve

COMMIT = "866ba392f1c1e84b92ad75fa66550f26e8af8e48"
SOURCE_HASHES = {
    "pf.py": "cfc9d1f91b49f06e99d9b4683e76f48025bb09187452e49af506a3ad0df4c74b",
    "octomap_bridge.py": (
        "dcd51e20c2fd6860674b68b9d0d2bf1b0409f72c183d917a7927ae9850596c4a"
    ),
}


def generate(source: Path, output: Path) -> None:
    namespace = dict(
        np=np,
        skfmm=skfmm,
        PFConfig=object,
        binary_closing=binary_closing,
        convolve=convolve,
    )
    hashes = {}
    for name, selected in (
        (
            "pf.py",
            {
                "make_axes",
                "make_grid",
                "make_sdf",
                "grad3",
                "make_guidance_field_progressive",
                "make_pf_for_octomap",
            },
        ),
        ("octomap_bridge.py", {"closing_opening_padded"}),
    ):
        path = source / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[name] != SOURCE_HASHES[name]:
            raise ValueError(f"{name} does not match pinned upstream source.")
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in selected
        ]
        if {n.name for n in nodes} != selected:
            raise ValueError("Missing author functions.")
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    raw = np.zeros((32, 24, 35), dtype=bool)
    raw[14:18, 5:18, 0:27] = True
    raw[5:10, 7:16, 26:29] = True
    raw[24:27, 8:18, 2:4] = True
    raw[11, 11, 11] = True
    processed = namespace["closing_opening_padded"](raw)
    # Literal statements from OctomapHandler, after closing_opening_padded.
    processed[:, :, 20:] = np.cumsum(processed[:, :, 20:], axis=2) > 0
    processed[:, :, :5] = (
        np.cumsum(processed[:, :, :5][:, :, ::-1], axis=2)[:, :, ::-1] > 0
    )
    voxel = 0.04
    lower = np.array([-0.64, -0.48, 0.0], np.float32)
    goal = np.array([0.4, 0.24, 0.75], np.float32)
    cfg = SimpleNamespace(
        voxel=voxel,
        Nx=32,
        Ny=24,
        Nz=35,
        Lx=32 * voxel,
        Ly=24 * voxel,
        Lz=35 * voxel,
        origin_w=lower,
        goal_w=goal,
        goal_seed_r=0.12,
        v_max=0.6,
        k_decay=0.6,
    )
    sdf, bf, gf = namespace["make_pf_for_octomap"](cfg, processed)
    np.savez_compressed(
        output,
        raw=raw,
        processed=processed,
        sdf=sdf,
        bf=bf,
        gf=gf,
        origin=lower + voxel / 2,
        goal=goal,
        voxel=voxel,
    )
    output.with_suffix(".json").write_text(
        json.dumps(dict(commit=COMMIT, sha256=hashes), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
