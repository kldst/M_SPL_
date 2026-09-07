#!/usr/bin/env python3
"""Dump the fixed world camera HeatFormer's make_video.py fits for a sequence.

`compute_virtual_camera` is byte-identical in both repos, so the only thing that
makes the two videos look different is the data fed to the fit (and the frame the
geometry lives in). Rather than re-deriving the fit, this reproduces
HeatFormer's pass 1 exactly -- same `--camera_sample` subsampling, same
`camera_joints(root, head)` synthetic joints, same pred + GT vertex pool -- and
writes the resulting camera dict to JSON.

`infer_temporal_topk_mp4.py --camera-json <file> --world-frame` then renders from
that exact camera, so its output lines up with
`HeatFormer/output/videos_gt/<seq>/gt_pred_compare_3d.mp4`.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

HEATFORMER = Path("/train-data-3-hdd/yian/HeatFormer")

from infer_markerless_smpl_3d_gif import compute_virtual_camera  # noqa: E402

# make_video.py: "Head_top; SMPL/SMPL-X uses joint 15"
H36M_HEAD = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump HeatFormer's fitted world camera for one sequence."
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--pred-dir",
        default=str(HEATFORMER / "output/mamma_full_gt"),
        help="Directory of HeatFormer per-run prediction NPZs (verts_world).",
    )
    parser.add_argument(
        "--data-root",
        default="/train-data-3-hdd/yian/Multi_SMPL_0706/MAMMA_eval_dance/test",
    )
    parser.add_argument(
        "--sequence-log",
        default=None,
        help="Defaults to <data-root>/logs/process.log.",
    )
    parser.add_argument("--azimuth", type=float, default=35.0)
    parser.add_argument("--elevation", type=float, default=28.0)
    parser.add_argument("--fov", type=float, default=38.0)
    parser.add_argument("--camera-sample", type=int, default=8)
    parser.add_argument(
        "--pred-only",
        action="store_true",
        help="Fit on predictions alone, matching make_video.py --pred_only.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sequence_runs(log_path: Path, sequence: str) -> list[str]:
    """Resolve a sequence name to its runs_* ids via the dataset's own log."""
    import re

    pattern = re.compile(r"SUCCESS\s+(.+?):\s+(\d+) frames,.*?\(run_offset now (\d+)\)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        name, count_text, end_text = match.groups()
        if name != sequence:
            continue
        count, end = int(count_text), int(end_text)
        return [f"runs_{index:05d}" for index in range(end - count, end)]
    raise SystemExit(f"Sequence {sequence!r} not found in {log_path}")


def load_frame(pred_dir: str, run: str) -> list[dict]:
    """make_video.py load_frame()."""
    path = osp.join(pred_dir, run + ".npz")
    if not osp.exists(path):
        return []
    with np.load(path, allow_pickle=False) as d:
        names = sorted({k.split("/")[0] for k in d.files if "/" in k})
        return [
            {"verts": d[f"{n}/verts_world"], "joints": d[f"{n}/joints_world"]}
            for n in names
        ]


def load_gt(data_root: str, run: str) -> list[dict]:
    """make_video.py load_gt()."""
    mesh_dir = osp.join(data_root, "out_mesh", run)
    if not osp.isdir(mesh_dir):
        return []
    out = []
    for f in sorted(os.listdir(mesh_dir)):
        with np.load(osp.join(mesh_dir, f), allow_pickle=False) as d:
            out.append({"verts": d["verts"].astype(np.float32)})
    return out


def camera_joints(root: np.ndarray, head: np.ndarray) -> np.ndarray:
    """make_video.py camera_joints(): estimate_up_direction reads joints 0 and 15."""
    out = np.zeros((len(root), 16, 3), dtype=np.float64)
    out[:, 0], out[:, 15] = root, head
    return out


def main() -> None:
    args = parse_args()
    log_path = Path(
        args.sequence_log or (Path(args.data_root) / "logs/process.log")
    )
    runs = sequence_runs(log_path, args.sequence)

    # ---- HeatFormer make_video.py pass 1, verbatim ---------------------------
    verts_for_cam: list[np.ndarray] = []
    joints_for_cam: list[np.ndarray] = []
    for run in runs[:: max(1, args.camera_sample)]:
        people = load_frame(args.pred_dir, run)
        if not people:
            continue
        verts_for_cam.append(np.stack([p["verts"] for p in people]))
        joints_for_cam.append(
            camera_joints(
                np.stack([p["joints"][0] for p in people]),
                np.stack([p["joints"][H36M_HEAD] for p in people]),
            )
        )
        if not args.pred_only:
            gt = load_gt(args.data_root, run)
            if gt:
                verts_for_cam.append(np.stack([g["verts"] for g in gt]))
                joints_for_cam.append(
                    camera_joints(
                        np.stack([g["verts"].mean(0) for g in gt]),
                        np.stack(
                            [g["verts"][g["verts"][:, 2].argmax()] for g in gt]
                        ),
                    )
                )
    if not verts_for_cam:
        raise SystemExit(f"No geometry found for {args.sequence}")

    camera = compute_virtual_camera(
        verts_for_cam, joints_for_cam, args.azimuth, args.elevation, args.fov
    )

    payload = {
        "frame": "world",
        "source": "HeatFormer make_video.py pass-1 fit, reproduced verbatim",
        "sequence": args.sequence,
        "pred_dir": str(Path(args.pred_dir).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "runs": [runs[0], runs[-1]],
        "run_count": len(runs),
        "camera_sample": args.camera_sample,
        "pred_only": bool(args.pred_only),
        "azimuth_deg": args.azimuth,
        "elevation_deg": args.elevation,
        "fov_deg": args.fov,
        "camera": {
            key: (value.tolist() if isinstance(value, np.ndarray) else float(value))
            for key, value in camera.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\n[RESULT] {output}")


if __name__ == "__main__":
    main()
