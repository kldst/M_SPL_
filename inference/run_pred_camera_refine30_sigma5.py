#!/train-data-3-hdd/yian/conda/envs/mamma/bin/python
"""Render predicted-camera 30-step, sigma-5, axis-guarded mask refinement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inference.infer_mamma_pred_camera_refine import (  # noqa: E402
    DEFAULT_SMPL,
    motion_stats,
    render_3d_scene,
    render_comparison,
)
from inference.run_camera_smoothing_ablation import (  # noqa: E402
    apply_axis_guard,
    run_refiner,
)

DEFAULT_SOURCE = (
    REPO / "inference" / "outputs" / "mamma_eval_dance_ck30_pred_camera_refine_225"
)
DEFAULT_OUTPUT = (
    REPO / "inference" / "outputs"
    / "mamma_eval_dance_ck30_pred_camera_refine30_sigma5_xy5cm_z60cm_225"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smpl-model", type=Path, default=DEFAULT_SMPL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refined-title", default="PRED CAM / REFINE 30 / SIGMA 5")
    parser.add_argument("--refined-subtitle", default="XY 5cm / Z 60cm axis guard")
    parser.add_argument("--camera-note",
                        default="checkpoint pose_enc predicted camera; no GT camera")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    bundle = args.source / "pred_camera_refine_bundle.npz"
    images = args.source / "input_images"
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    if not images.is_dir():
        raise FileNotFoundError(images)

    unguarded = args.output / "refined_tracks_before_axis_guard.npz"
    guarded = args.output / "refined_tracks.npz"
    if args.force or not unguarded.is_file():
        run_refiner(bundle, unguarded, args.sigma, args)
    guard_summary = apply_axis_guard(unguarded, guarded)

    videos_3d = render_3d_scene(
        args,
        bundle,
        guarded,
        args.output,
        refined_title=args.refined_title,
        refined_subtitle=args.refined_subtitle,
    )
    reprojection = render_comparison(args, bundle, guarded, images, args.output)

    raw_summary = json.loads(unguarded.with_suffix(".json").read_text())
    with np.load(guarded, allow_pickle=False) as tracks:
        refined_motion = motion_stats(tracks["refined_translate"].astype(np.float32))
    manifest = {
        "camera": args.camera_note,
        "identity": "presence top-2 ordered by model slot ID; no GT assignment",
        "iterations": args.iterations,
        "smooth_sigma_frames": args.sigma,
        "axis_guard": guard_summary,
        "mean_mask_iou_original": raw_summary["mean_mask_iou_original"],
        "mean_mask_iou_optimized_before_smoothing": raw_summary["mean_mask_iou_optimized"],
        "mean_mask_iou_after_smoothing_before_axis_guard": raw_summary["mean_mask_iou_refined"],
        "motion_after_axis_guard": refined_motion,
        "tracks": str(guarded),
        "tracks_before_axis_guard": str(unguarded),
        "videos": {"3d": videos_3d, "reprojection": str(reprojection)},
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
