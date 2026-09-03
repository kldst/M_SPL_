#!/train-data-3-hdd/yian/conda/envs/mamma/bin/python
"""Run a controlled camera x smoothing ablation with guarded XYZ correction.

Every optimized correction is optionally Gaussian-smoothed by the shared
refiner, then constrained to X/Y +/-5 cm and Z +/-60 cm before rendering.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import infer_markerless_smpl_3d_gif as render3d  # noqa: E402
import infer_markerless_smpl_gif as common  # noqa: E402
from infer_temporal_smpl_mesh_hungarian_mp4 import (  # noqa: E402
    decode_params_at_mesh_translate,
    label_frame,
    load_refine_cameras,
)
from inference.infer_mamma_pred_camera_refine import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SMPL,
    POSEGAM_PYTHON,
    REFINER,
    close_video_writer,
    draw_mask_contours,
    draw_mesh_points,
    motion_stats,
    open_video_writer,
)
from training import smpl_body  # noqa: E402

DEFAULT_SOURCE = (
    REPO / "inference" / "outputs" / "mamma_eval_dance_ck30_pred_camera_refine_225"
)
DEFAULT_OUTPUT = (
    REPO / "inference" / "outputs" / "mamma_eval_dance_ck30_camera_smoothing_ablation_225"
)
XY_CORRECTION_LIMIT_M = 0.05
Z_CORRECTION_LIMIT_M = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smpl-model", type=Path, default=DEFAULT_SMPL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--reprojection-only", action="store_true",
        help="reuse guarded tracks and existing 3D videos; only render reprojection videos",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_gt_camera_bundle(source: Path, dataset_root: Path, target: Path) -> None:
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    frames = data["frames"].astype(str)
    views = data["views"].astype(str).tolist()
    extrinsics, intrinsics = [], []
    for index, frame in enumerate(frames):
        archive_path = dataset_root / "test" / "out_data" / f"{frame}.npz"
        image_paths = [dataset_root / "test" / "out_image" / frame / f"{view}.jpg" for view in views]
        E, K = load_refine_cameras(archive_path, views, image_paths, 518, 518)
        extrinsics.append(E.astype(np.float32))
        intrinsics.append(K.astype(np.float32))
        if (index + 1) % 50 == 0 or index + 1 == len(frames):
            print(f"[ablation] GT cameras {index + 1}/{len(frames)}", flush=True)
    data["extrinsics"] = np.stack(extrinsics)
    data["intrinsics"] = np.stack(intrinsics)
    np.savez_compressed(target, **data)


def run_refiner(bundle: Path, target: Path, sigma: float, args: argparse.Namespace) -> None:
    command = [
        str(POSEGAM_PYTHON), str(REFINER),
        "--bundle", str(bundle), "--output", str(target),
        "--iterations", str(args.iterations), "--learning-rate", "0.025",
        "--max-correction-m", "0.6", "--anchor-weight", "0.015",
        "--temporal-weight", "0", "--smooth-sigma", str(sigma),
        "--log-every", str(args.log_every),
    ]
    print("[ablation] RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def apply_axis_guard(source: Path, target: Path) -> dict[str, object]:
    """Clamp a refiner's final (already smoothed) correction by camera-0 axis."""
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    correction = data["translate_correction"].astype(np.float32).copy()
    before = correction.copy()
    correction[..., :2] = np.clip(
        correction[..., :2], -XY_CORRECTION_LIMIT_M, XY_CORRECTION_LIMIT_M
    )
    correction[..., 2] = np.clip(
        correction[..., 2], -Z_CORRECTION_LIMIT_M, Z_CORRECTION_LIMIT_M
    )
    data["translate_correction"] = correction
    data["refined_translate"] = data["pred_translate"].astype(np.float32) + correction
    np.savez_compressed(target, **data)
    changed = np.abs(before - correction) > 1e-7
    return {
        "xy_correction_limit_m": XY_CORRECTION_LIMIT_M,
        "z_correction_limit_m": Z_CORRECTION_LIMIT_M,
        "changed_values_xyz": changed.sum(axis=(0, 1)).astype(int).tolist(),
        "max_abs_correction_xyz_m": np.abs(correction).max(axis=(0, 1)).tolist(),
    }


def render_all(combos: dict[str, dict], base_bundle: Path, args: argparse.Namespace) -> dict:
    with np.load(base_bundle, allow_pickle=False) as bundle:
        original_vertices = bundle["pred_vertices"].astype(np.float32)
        faces = bundle["faces"].astype(np.int64)
        slots = bundle["matched_slots"].astype(np.int64)
        pose = bundle["pred_pose"].astype(np.float32)
        beta = bundle["pred_beta"].astype(np.float32)
        translate = bundle["pred_translate"].astype(np.float32)

    smpl_body._SMPL_MODEL_PATHS["neutral"] = str(args.smpl_model.resolve())
    device = torch.device(args.device)
    original_joints = []
    for index in range(len(original_vertices)):
        _, joints = decode_params_at_mesh_translate(pose[index], beta[index], translate[index], device)
        original_joints.append(joints)

    all_vertices = list(original_vertices)
    all_joints = list(original_joints)
    for combo in combos.values():
        with np.load(combo["tracks"], allow_pickle=False) as tracks:
            correction = tracks["translate_correction"].astype(np.float32)
        combo["correction"] = correction
        combo["vertices"] = original_vertices + correction[:, :, None, :]
        combo["joints"] = [joints + correction[i, :, None, :] for i, joints in enumerate(original_joints)]
        all_vertices.extend(list(combo["vertices"]))
        all_joints.extend(combo["joints"])

    camera = render3d.compute_virtual_camera(all_vertices, all_joints, 35.0, 28.0, 38.0)
    size = 640
    four_way = args.output / "four_way_refined_3d.mp4"
    writers = {"four_way": open_video_writer(four_way, args.fps, size * 2, size * 2)}
    outputs = {"four_way": str(four_way), "combinations": {}}
    for key, combo in combos.items():
        combo_dir = args.output / key
        combo_dir.mkdir(parents=True, exist_ok=True)
        refined_path = combo_dir / "refined_3d.mp4"
        compare_path = combo_dir / "original_vs_refined_3d.mp4"
        writers[f"{key}_refined"] = open_video_writer(refined_path, args.fps, size, size)
        writers[f"{key}_compare"] = open_video_writer(compare_path, args.fps, size * 2, size)
        outputs["combinations"][key] = {
            "refined": str(refined_path), "original_vs_refined": str(compare_path)
        }

    try:
        for index in range(len(original_vertices)):
            original = label_frame(
                render3d.render_mesh_frame(original_vertices[index], faces, slots[index], camera, size, size),
                "PRED ORIGINAL", "shared checkpoint SMPL", (90, 115, 255),
            )
            panels = {}
            for key, combo in combos.items():
                panel = label_frame(
                    render3d.render_mesh_frame(combo["vertices"][index], faces, slots[index], camera, size, size),
                    combo["title"], combo["subtitle"], combo["colour"],
                )
                panels[key] = panel
                for writer_key, image in (
                    (f"{key}_refined", panel),
                    (f"{key}_compare", np.hstack([original, panel])),
                ):
                    assert writers[writer_key].stdin is not None
                    writers[writer_key].stdin.write(np.ascontiguousarray(image).tobytes())
            grid = np.vstack([
                np.hstack([panels["gt_camera_sigma0"], panels["gt_camera_sigma5"]]),
                np.hstack([panels["pred_camera_sigma0"], panels["pred_camera_sigma5"]]),
            ])
            assert writers["four_way"].stdin is not None
            writers["four_way"].stdin.write(np.ascontiguousarray(grid).tobytes())
            if (index + 1) % args.log_every == 0 or index + 1 == len(original_vertices):
                print(f"[ablation] render 3D {index + 1}/{len(original_vertices)}", flush=True)
    finally:
        for key, process in writers.items():
            if key == "four_way":
                path = four_way
            else:
                combo_key, kind = key.rsplit("_", 1)
                path = Path(outputs["combinations"][combo_key]["refined" if kind == "refined" else "original_vs_refined"])
            close_video_writer(process, path)
    return outputs


def render_reprojections(combos: dict[str, dict], base_bundle: Path,
                         images_dir: Path, args: argparse.Namespace) -> dict:
    """Render each guarded mesh over its eight input views plus a 2x2 ablation."""
    with np.load(base_bundle, allow_pickle=False) as bundle:
        views = bundle["views"].astype(str).tolist()
        original_vertices = bundle["pred_vertices"].astype(np.float32)
        masks = bundle["mask_prob"].astype(np.float32)
    if len(views) != 8:
        raise ValueError(f"Reprojection mosaic requires eight views, got {len(views)}")

    for combo in combos.values():
        with np.load(combo["bundle"], allow_pickle=False) as bundle:
            combo["extrinsics"] = bundle["extrinsics"].astype(np.float32)
            combo["intrinsics"] = bundle["intrinsics"].astype(np.float32)
        with np.load(combo["tracks"], allow_pickle=False) as tracks:
            correction = tracks["translate_correction"].astype(np.float32)
        combo["refined_vertices"] = original_vertices + correction[:, :, None, :]

    cell_size = 256
    mosaic_width, mosaic_height = cell_size * 4, cell_size * 2
    four_way = args.output / "four_way_reprojection_refined_8view.mp4"
    writers = {
        "four_way": open_video_writer(
            four_way, args.fps, mosaic_width * 2, mosaic_height * 2
        )
    }
    outputs = {"four_way": str(four_way), "combinations": {}}
    for key in combos:
        target = args.output / key / "reprojection_refined_8view.mp4"
        writers[key] = open_video_writer(target, args.fps, mosaic_width, mosaic_height)
        outputs["combinations"][key] = str(target)

    mesh_colours = [(245, 220, 30), (245, 150, 30)]
    try:
        for index in range(len(original_vertices)):
            source_images = []
            for view in views:
                path = images_dir / f"{index:04d}_{view}.jpg"
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(path)
                source_images.append(image)
            panels = {}
            for key, combo in combos.items():
                cells = []
                for view_index, (view, source_image) in enumerate(zip(views, source_images)):
                    cell = source_image.copy()
                    draw_mask_contours(cell, masks[index, view_index])
                    draw_mesh_points(
                        cell, combo["refined_vertices"][index],
                        combo["extrinsics"][index, view_index],
                        combo["intrinsics"][index, view_index], mesh_colours,
                    )
                    cv2.rectangle(cell, (0, 0), (cell.shape[1], 27), (20, 20, 24), -1)
                    cv2.putText(
                        cell, f"{view} REFINED", (7, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (245, 245, 245), 2, cv2.LINE_AA,
                    )
                    cells.append(cv2.resize(
                        cell, (cell_size, cell_size), interpolation=cv2.INTER_AREA
                    ))
                panel = np.vstack([np.hstack(cells[:4]), np.hstack(cells[4:])])
                cv2.rectangle(panel, (0, 0), (mosaic_width, 28), (20, 20, 24), -1)
                cv2.putText(
                    panel, combo["title"] + " / XY 5cm / Z 60cm", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, combo["colour"], 2, cv2.LINE_AA,
                )
                panels[key] = panel
                assert writers[key].stdin is not None
                writers[key].stdin.write(np.ascontiguousarray(panel).tobytes())
            grid = np.vstack([
                np.hstack([panels["gt_camera_sigma0"], panels["gt_camera_sigma5"]]),
                np.hstack([panels["pred_camera_sigma0"], panels["pred_camera_sigma5"]]),
            ])
            assert writers["four_way"].stdin is not None
            writers["four_way"].stdin.write(np.ascontiguousarray(grid).tobytes())
            if (index + 1) % args.log_every == 0 or index + 1 == len(original_vertices):
                print(
                    f"[ablation] render reprojection {index + 1}/{len(original_vertices)}",
                    flush=True,
                )
    finally:
        for key, process in writers.items():
            target = four_way if key == "four_way" else Path(outputs["combinations"][key])
            close_video_writer(process, target)
    return outputs


def main() -> int:
    args = parse_args()
    started = time.time()
    args.source = args.source.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    predicted_bundle = args.source / "pred_camera_refine_bundle.npz"
    if not predicted_bundle.is_file():
        raise FileNotFoundError(predicted_bundle)
    gt_bundle = args.output / "gt_camera_refine_bundle.npz"
    if args.force or not gt_bundle.is_file():
        build_gt_camera_bundle(predicted_bundle, args.dataset_root, gt_bundle)

    specs = {
        "gt_camera_sigma0": (gt_bundle, 0.0),
        "gt_camera_sigma5": (gt_bundle, 5.0),
        "pred_camera_sigma0": (predicted_bundle, 0.0),
        "pred_camera_sigma5": (predicted_bundle, 5.0),
    }
    styles = {
        "gt_camera_sigma0": ("GT CAMERA / SIGMA 0", "raw correction; XY 5cm / Z 60cm", (110, 235, 125)),
        "gt_camera_sigma5": ("GT CAMERA / SIGMA 5", "Gaussian correction; XY 5cm / Z 60cm", (90, 210, 245)),
        "pred_camera_sigma0": ("PRED CAMERA / SIGMA 0", "raw correction; XY 5cm / Z 60cm", (110, 130, 255)),
        "pred_camera_sigma5": ("PRED CAMERA / SIGMA 5", "Gaussian correction; XY 5cm / Z 60cm", (255, 220, 80)),
    }
    combos = {}
    for key, (bundle, sigma) in specs.items():
        combo_dir = args.output / key
        combo_dir.mkdir(parents=True, exist_ok=True)
        tracks = combo_dir / "refined_tracks.npz"
        unguarded_tracks = combo_dir / "refined_tracks_before_axis_guard.npz"
        if args.force:
            run_refiner(bundle, unguarded_tracks, sigma, args)
        elif not unguarded_tracks.is_file():
            if tracks.is_file():
                # Preserve legacy ablation tracks before replacing their final
                # correction with the guarded version.
                shutil.copy2(tracks, unguarded_tracks)
                legacy_summary = tracks.with_suffix(".json")
                if legacy_summary.is_file():
                    shutil.copy2(legacy_summary, unguarded_tracks.with_suffix(".json"))
            else:
                run_refiner(bundle, unguarded_tracks, sigma, args)
        guard = apply_axis_guard(unguarded_tracks, tracks)
        title, subtitle, colour = styles[key]
        combos[key] = {
            "bundle": bundle, "tracks": tracks, "unguarded_tracks": unguarded_tracks,
            "sigma": sigma, "axis_guard": guard,
            "title": title, "subtitle": subtitle, "colour": colour,
        }

    manifest_path = args.output / "ablation_manifest.json"
    if args.reprojection_only and manifest_path.is_file():
        videos = json.loads(manifest_path.read_text()).get("videos", {})
    else:
        videos = render_all(combos, predicted_bundle, args)
    videos["reprojection"] = render_reprojections(
        combos, predicted_bundle, args.source / "input_images", args
    )
    metrics = {}
    for key, combo in combos.items():
        summary = json.loads(Path(combo["unguarded_tracks"]).with_suffix(".json").read_text())
        with np.load(combo["tracks"], allow_pickle=False) as tracks:
            refined = tracks["refined_translate"].astype(np.float32)
        metrics[key] = {
            "camera": "gt calibration" if key.startswith("gt_") else "model pose_enc",
            "smooth_sigma_frames": combo["sigma"],
            "axis_guard": combo["axis_guard"],
            "mean_mask_iou_original": summary["mean_mask_iou_original"],
            "mean_mask_iou_optimized_before_smoothing": summary["mean_mask_iou_optimized"],
            "mean_mask_iou_after_smoothing_before_axis_guard": summary["mean_mask_iou_refined"],
            "accepted_frames": summary["accepted_frames"],
            "motion": motion_stats(refined),
            "tracks": str(combo["tracks"]),
            "tracks_before_axis_guard": str(combo["unguarded_tracks"]),
        }
    manifest = {
        "design": (
            "same presence top-2 slots, SMPL, masks, and optimizer; no GT Hungarian; "
            "final camera-0 correction constrained by axis"
        ),
        "gt_usage": "GT calibration only in gt_camera rows; no GT body/identity assignment",
        "axis_guard": {
            "xy_correction_limit_m": XY_CORRECTION_LIMIT_M,
            "z_correction_limit_m": Z_CORRECTION_LIMIT_M,
            "order": "optimize, Gaussian-smooth when sigma > 0, then clamp by axis",
        },
        "frames": 225, "fps": args.fps, "metrics": metrics, "videos": videos,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
