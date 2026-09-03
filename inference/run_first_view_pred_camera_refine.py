#!/train-data-3-hdd/yian/conda/envs/mamma/bin/python
"""Compare cam0-only predicted-camera SMPL mask refine at sigma 0 and 5."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inference.infer_mamma_pred_camera_refine import (  # noqa: E402
    DEFAULT_SMPL, POSEGAM_PYTHON, REFINER, motion_stats, render_3d_scene,
)

DEFAULT_SOURCE = REPO / "inference/outputs/mamma_eval_dance_ck30_pred_camera_refine_225"
DEFAULT_OUTPUT = REPO / "inference/outputs/mamma_eval_dance_ck30_pred_camera_cam0_refine_225"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smpl-model", type=Path, default=DEFAULT_SMPL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_cam0_bundle(source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data["views"] = data["views"][:1]
    for key in ("mask_prob", "extrinsics", "intrinsics"):
        data[key] = data[key][:, :1]
    np.savez_compressed(target, **data)


def run_refine(bundle: Path, target: Path, sigma: float, args) -> None:
    subprocess.run([
        str(POSEGAM_PYTHON), str(REFINER),
        "--bundle", str(bundle), "--output", str(target),
        "--iterations", str(args.iterations), "--learning-rate", "0.025",
        "--max-correction-m", "0.6", "--anchor-weight", "0.015",
        "--temporal-weight", "0", "--smooth-sigma", str(sigma),
        "--log-every", str(args.log_every),
    ], cwd=REPO, check=True)


def main() -> int:
    args = parse_args()
    started = time.time()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    source_bundle = args.source / "pred_camera_refine_bundle.npz"
    bundle = args.output / "pred_camera_cam0_bundle.npz"
    if args.force or not bundle.is_file():
        build_cam0_bundle(source_bundle, bundle)

    results = {}
    for sigma in (0.0, 5.0):
        key = f"cam0_sigma{int(sigma)}"
        directory = args.output / key
        directory.mkdir(parents=True, exist_ok=True)
        tracks = directory / "refined_tracks.npz"
        if args.force or not tracks.is_file():
            run_refine(bundle, tracks, sigma, args)
        videos = render_3d_scene(
            args, bundle, tracks, directory,
            refined_title=f"PRED CAM0 / SIGMA {int(sigma)}",
            refined_subtitle="single predicted cam0 mask; 10-step translation refine",
        )
        summary = json.loads(tracks.with_suffix(".json").read_text())
        with np.load(tracks, allow_pickle=False) as archive:
            motion = motion_stats(archive["refined_translate"].astype(np.float32))
        results[key] = {
            "views": ["IOI_01"], "smooth_sigma_frames": sigma,
            "mean_mask_iou_original": summary["mean_mask_iou_original"],
            "mean_mask_iou_optimized_before_smoothing": summary["mean_mask_iou_optimized"],
            "mean_mask_iou_final": summary["mean_mask_iou_refined"],
            "accepted_frames": summary["accepted_frames"],
            "motion": motion, "tracks": str(tracks), "videos": videos,
        }
    manifest = {
        "design": "predicted cam0 only; same presence top-2 SMPL/masks; no GT or Hungarian",
        "frames": 225, "results": results, "elapsed_seconds": time.time() - started,
    }
    (args.output / "cam0_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
