#!/train-data-3-hdd/yian/conda/envs/mamma/bin/python
"""GT-free MAMMA temporal inference with predicted-camera mask refinement.

The model supplies SMPL, presence, person masks, extrinsics, and intrinsics.
Presence top-2 slots are refined frame by frame; GT cameras and GT Hungarian
assignment are never loaded. Pose/shape/root rotation and predicted cameras stay
fixed while a per-person XYZ translation correction is optimized against all
eight predicted masks. No Gaussian post-filter is applied.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import infer_markerless_smpl_3d_gif as render3d  # noqa: E402
import infer_markerless_smpl_gif as common  # noqa: E402
from infer_temporal_smpl_mesh_hungarian_mp4 import (  # noqa: E402
    causal_window_indices,
    decode_params_at_mesh_translate,
    label_frame,
    select_prediction_frame,
)
from training import smpl_body  # noqa: E402
from training.smpl_body import _get_smpl_model  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

DEFAULT_DATASET = REPO.parent / "Multi_SMPL_0706" / "MAMMA_eval_dance"
DEFAULT_CHECKPOINT = REPO / "model" / "root" / "checkpoint_30.pt"
DEFAULT_OUTPUT = REPO / "inference" / "outputs" / "mamma_eval_dance_ck30_pred_camera_refine"
DEFAULT_SMPL = (
    REPO.parent / "Multi_SMPL_0706" / "smpl_models"
    / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
)
REFINER = REPO / "debug" / "0901_meeting" / "refine_mesh_translate_from_masks.py"
POSEGAM_PYTHON = Path("/train-data-3-hdd/yian/conda/envs/posegam/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smpl-model", type=Path, default=DEFAULT_SMPL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=225)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def relative_to_camera0(extrinsics: torch.Tensor) -> np.ndarray:
    """Convert predicted OpenCV world-to-camera matrices to the SMPL cam0 gauge."""
    count = extrinsics.shape[0]
    homogeneous = torch.eye(4, dtype=torch.float32, device=extrinsics.device).repeat(count, 1, 1)
    homogeneous[:, :3, :4] = extrinsics.float()
    relative = homogeneous @ torch.linalg.inv(homogeneous[0])
    return relative[:, :3, :4].detach().cpu().numpy().astype(np.float32)


def valid_bundle(path: Path, frame_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "pred_pose", "pred_beta", "pred_translate", "pred_vertices",
                "mask_prob", "extrinsics", "intrinsics", "matched_slots",
            }
            return required.issubset(data.files) and len(data["pred_translate"]) == frame_count
    except Exception:
        return False


def build_bundle(args: argparse.Namespace, bundle_path: Path, images_dir: Path) -> dict:
    device = torch.device(args.device)
    smpl_body._SMPL_MODEL_PATHS["neutral"] = str(args.smpl_model.resolve())
    frame_dirs = common.discover_frames(args.dataset_root.resolve(), args.split, args.max_frames)
    first_images = common.list_frame_images(frame_dirs[0])
    if len(first_images) < 8:
        raise ValueError(f"Expected at least eight views, got {[path.stem for path in first_images]}")
    input_indices = list(range(8))
    views = [first_images[index].stem for index in input_indices]

    model, _, incompatible = common.load_model(
        args.config, args.checkpoint.resolve(), device, keep_person_mask=True
    )
    model.eval()
    faces = np.asarray(_get_smpl_model(device, "neutral").faces, dtype=np.int32)
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "pred_pose", "pred_beta", "pred_translate", "pred_vertices",
            "mask_prob", "extrinsics", "intrinsics", "matched_slots", "presence",
        )
    }
    images_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for frame_index, frame_dir in enumerate(frame_dirs):
        window = causal_window_indices(frame_index, args.clip_length)
        image_paths = [
            str(frame_images[index])
            for source_index in window
            for frame_images in [common.list_frame_images(frame_dirs[source_index])]
            for index in input_indices
        ]
        inputs = load_and_preprocess_images(image_paths).to(device)
        smpl_inputs = {
            "temporal_num_frames": torch.tensor([args.clip_length], device=device),
            "views_per_frame": torch.tensor([len(views)], device=device),
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            prediction = model(inputs, smpl_inputs=smpl_inputs)
        current = select_prediction_frame(prediction, args.clip_length - 1)
        logits = current.get("smpl_presence_logits")
        if logits is None:
            raise KeyError("Checkpoint did not return smpl_presence_logits")
        presence = common.stable_sigmoid(logits[0].float().cpu().numpy()).astype(np.float32)
        # Sort the selected IDs so array position stays stable when only their score rank flips.
        slots = np.sort(np.argsort(-presence, kind="stable")[:2]).astype(np.int64)

        mask_logits = current.get("person_mask_logits")
        if mask_logits is None:
            raise KeyError("Checkpoint did not return person_mask_logits")
        probability = torch.sigmoid(mask_logits[0].float())
        if probability.shape[-2:] != (128, 128):
            probability = F.interpolate(
                probability.flatten(0, 1).unsqueeze(1),
                size=(128, 128), mode="bilinear", align_corners=False,
            ).squeeze(1).reshape(probability.shape[0], probability.shape[1], 128, 128)

        pose_enc = current.get("pose_enc")
        if pose_enc is None:
            raise KeyError("Checkpoint camera head did not return pose_enc")
        predicted_e, predicted_k = pose_encoding_to_extri_intri(
            pose_enc.float(), image_size_hw=inputs.shape[-2:]
        )
        if predicted_e.shape[1] != len(views):
            raise ValueError(f"Expected {len(views)} current cameras, got {tuple(predicted_e.shape)}")
        relative_e = relative_to_camera0(predicted_e[0])
        intrinsic = predicted_k[0].detach().cpu().numpy().astype(np.float32)

        vertices, _ = render3d.decode_people(current, slots, device, avg_scale=1.0)
        arrays["pred_pose"].append(current["smpl_pose"][0, slots].float().cpu().numpy())
        arrays["pred_beta"].append(current["smpl_beta"][0, slots].float().cpu().numpy())
        arrays["pred_translate"].append(current["mesh_translate"][0, slots].float().cpu().numpy())
        arrays["pred_vertices"].append(vertices.astype(np.float32))
        arrays["mask_prob"].append(probability[:, slots].cpu().numpy().astype(np.float16))
        arrays["extrinsics"].append(relative_e)
        arrays["intrinsics"].append(intrinsic)
        arrays["matched_slots"].append(slots)
        arrays["presence"].append(presence[slots])

        current_images = inputs[-len(views):]
        for view_index, view in enumerate(views):
            cv2.imwrite(
                str(images_dir / f"{frame_index:04d}_{view}.jpg"),
                common.tensor_image_to_bgr(current_images[view_index]),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

        del prediction, current, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (frame_index + 1) % args.log_every == 0 or frame_index + 1 == len(frame_dirs):
            rate = (frame_index + 1) / max(time.time() - started, 1e-6)
            eta = (len(frame_dirs) - frame_index - 1) / max(rate, 1e-6)
            print(f"[pred-camera] inference {frame_index + 1}/{len(frame_dirs)} ETA={eta/60:.1f} min", flush=True)

    np.savez_compressed(
        bundle_path,
        frames=np.asarray([path.name for path in frame_dirs]),
        views=np.asarray(views),
        input_hw=np.asarray([518, 518], dtype=np.int64),
        faces=faces,
        **{key: np.stack(value) for key, value in arrays.items()},
    )
    return {
        "frames": len(frame_dirs),
        "views": views,
        "camera_source": "checkpoint pose_enc; converted to cam0-relative OpenCV extrinsics",
        "identity": "per-frame presence top-2, ordered by model slot ID; no GT assignment",
        "gt_data_loaded": False,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "inference_seconds": time.time() - started,
    }


def run_refine(args: argparse.Namespace, bundle: Path, refined: Path) -> None:
    command = [
        str(POSEGAM_PYTHON), str(REFINER),
        "--bundle", str(bundle), "--output", str(refined),
        "--iterations", str(args.iterations),
        "--learning-rate", str(args.learning_rate),
        "--max-correction-m", "0.6", "--anchor-weight", "0.015",
        "--temporal-weight", "0", "--smooth-sigma", "0",
        "--log-every", str(args.log_every),
    ]
    print("[pred-camera] RUN refine: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def project(vertices: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    camera = vertices @ extrinsic[:, :3].T + extrinsic[:, 3]
    depth = camera[:, 2]
    uvw = camera @ intrinsic.T
    uv = uvw[:, :2] / np.maximum(depth[:, None], 1e-6)
    return uv, depth


def draw_mesh_points(image: np.ndarray, vertices: np.ndarray, extrinsic: np.ndarray,
                     intrinsic: np.ndarray, colours: list[tuple[int, int, int]]) -> None:
    height, width = image.shape[:2]
    for person, colour in zip(vertices, colours):
        uv, depth = project(person[::12], extrinsic, intrinsic)
        valid = (depth > 1e-3) & np.isfinite(uv).all(axis=1)
        for x, y in uv[valid]:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                cv2.circle(image, (xi, yi), 1, colour, -1, cv2.LINE_AA)


def draw_mask_contours(image: np.ndarray, probability: np.ndarray) -> None:
    colours = [(80, 240, 80), (80, 200, 255)]
    height, width = image.shape[:2]
    for person, colour in zip(probability, colours):
        binary = (person >= 0.5).astype(np.uint8) * 255
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, colour, 1, cv2.LINE_AA)


def render_comparison(args: argparse.Namespace, bundle_path: Path, refined_path: Path,
                      images_dir: Path, output: Path) -> Path:
    with np.load(bundle_path, allow_pickle=False) as bundle, np.load(refined_path, allow_pickle=False) as tracks:
        frames = bundle["frames"].astype(str)
        views = bundle["views"].astype(str).tolist()
        original_vertices = bundle["pred_vertices"].astype(np.float32)
        correction = tracks["translate_correction"].astype(np.float32)
        refined_vertices = original_vertices + correction[:, :, None, :]
        masks = bundle["mask_prob"].astype(np.float32)
        extrinsics = bundle["extrinsics"].astype(np.float32)
        intrinsics = bundle["intrinsics"].astype(np.float32)

    frame_dir = output / "comparison_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cell_size = 256
    for index, _ in enumerate(frames):
        original_cells, refined_cells = [], []
        for view_index, view in enumerate(views):
            image = cv2.imread(str(images_dir / f"{index:04d}_{view}.jpg"), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(images_dir / f"{index:04d}_{view}.jpg")
            original, refined = image.copy(), image.copy()
            draw_mask_contours(original, masks[index, view_index])
            draw_mask_contours(refined, masks[index, view_index])
            draw_mesh_points(original, original_vertices[index], extrinsics[index, view_index],
                             intrinsics[index, view_index], [(30, 30, 245), (160, 30, 245)])
            draw_mesh_points(refined, refined_vertices[index], extrinsics[index, view_index],
                             intrinsics[index, view_index], [(245, 220, 30), (245, 150, 30)])
            for panel, title in ((original, f"{view} ORIGINAL"), (refined, f"{view} REFINED")):
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (20, 20, 24), -1)
                cv2.putText(panel, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (245, 245, 245), 2, cv2.LINE_AA)
            original_cells.append(cv2.resize(original, (cell_size, cell_size), interpolation=cv2.INTER_AREA))
            refined_cells.append(cv2.resize(refined, (cell_size, cell_size), interpolation=cv2.INTER_AREA))
        original_grid = np.vstack([np.hstack(original_cells[:4]), np.hstack(original_cells[4:])])
        refined_grid = np.vstack([np.hstack(refined_cells[:4]), np.hstack(refined_cells[4:])])
        gap = np.full((original_grid.shape[0], 4, 3), 64, dtype=np.uint8)
        cv2.imwrite(str(frame_dir / f"{index:04d}.png"), np.hstack([original_grid, gap, refined_grid]))
        if (index + 1) % args.log_every == 0 or index + 1 == len(frames):
            print(f"[pred-camera] render {index + 1}/{len(frames)}", flush=True)

    video = output / "pred_camera_original_vs_refined_8view.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps:g}",
        "-i", str(frame_dir / "%04d.png"), "-c:v", "libx264", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ], check=True)
    return video


def motion_stats(track: np.ndarray) -> dict:
    step = np.linalg.norm(np.diff(track, axis=0), axis=-1)
    acceleration = np.linalg.norm(track[2:] - 2 * track[1:-1] + track[:-2], axis=-1)
    return {
        "mean_step_m": float(step.mean()),
        "p95_step_m": float(np.percentile(step, 95)),
        "max_step_m": float(step.max()),
        "mean_acceleration_m": float(acceleration.mean()),
    }


def open_video_writer(path: Path, fps: float, width: int, height: int) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def close_video_writer(process: subprocess.Popen, path: Path) -> None:
    assert process.stdin is not None
    process.stdin.close()
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed ({code}): {path}")


def render_3d_scene(args: argparse.Namespace, bundle_path: Path, refined_path: Path,
                    output: Path, refined_title: str = "PRED MASK-REFINED",
                    refined_subtitle: str = "predicted camera; 10 steps; no Gaussian") -> dict[str, str]:
    """Render GT-free original/refined SMPL scenes with one fixed virtual camera."""
    with np.load(bundle_path, allow_pickle=False) as bundle, np.load(refined_path, allow_pickle=False) as tracks:
        faces = bundle["faces"].astype(np.int64)
        original_vertices = bundle["pred_vertices"].astype(np.float32)
        slots = bundle["matched_slots"].astype(np.int64)
        pose = tracks["pred_pose"].astype(np.float32)
        beta = tracks["pred_beta"].astype(np.float32)
        original_translate = tracks["pred_translate"].astype(np.float32)
        correction = tracks["translate_correction"].astype(np.float32)
    refined_vertices = original_vertices + correction[:, :, None, :]

    smpl_body._SMPL_MODEL_PATHS["neutral"] = str(args.smpl_model.resolve())
    device = torch.device(args.device)
    original_joints, refined_joints = [], []
    for index in range(len(original_vertices)):
        _, joints = decode_params_at_mesh_translate(
            pose[index], beta[index], original_translate[index], device
        )
        original_joints.append(joints)
        refined_joints.append(joints + correction[index, :, None, :])
    camera = render3d.compute_virtual_camera(
        list(original_vertices) + list(refined_vertices),
        original_joints + refined_joints,
        35.0, 28.0, 38.0,
    )

    size = 640
    comparison = output / "pred_camera_original_vs_refined_3d.mp4"
    refined_only = output / "pred_camera_refined_3d.mp4"
    writers = {
        "comparison": open_video_writer(comparison, args.fps, size * 2, size),
        "refined": open_video_writer(refined_only, args.fps, size, size),
    }
    try:
        for index in range(len(original_vertices)):
            track_ids = slots[index]
            original_panel = label_frame(
                render3d.render_mesh_frame(
                    original_vertices[index], faces, track_ids, camera, size, size
                ),
                "PRED ORIGINAL", "checkpoint mesh translation", (90, 115, 255),
            )
            refined_panel = label_frame(
                render3d.render_mesh_frame(
                    refined_vertices[index], faces, track_ids, camera, size, size
                ),
                refined_title, refined_subtitle, (255, 220, 80),
            )
            frames = {
                "comparison": np.hstack([original_panel, refined_panel]),
                "refined": refined_panel,
            }
            for key, image in frames.items():
                assert writers[key].stdin is not None
                writers[key].stdin.write(np.ascontiguousarray(image).tobytes())
            if (index + 1) % args.log_every == 0 or index + 1 == len(original_vertices):
                print(f"[pred-camera] render 3D {index + 1}/{len(original_vertices)}", flush=True)
    finally:
        close_video_writer(writers["comparison"], comparison)
        close_video_writer(writers["refined"], refined_only)
    return {"comparison": str(comparison), "refined": str(refined_only)}


def plot_corrections(refined_path: Path, target: Path) -> dict:
    with np.load(refined_path, allow_pickle=False) as data:
        original = data["pred_translate"].astype(np.float32)
        refined = data["refined_translate"].astype(np.float32)
        correction = data["translate_correction"].astype(np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    for person, colour in enumerate(("#00a7b5", "#376bd1")):
        axes[0].plot(np.linalg.norm(correction[:, person], axis=-1), color=colour, label=f"slot position {person}")
        axes[1].plot(np.linalg.norm(np.diff(original[:, person], axis=0), axis=-1),
                     color=colour, alpha=.35, label=f"original {person}")
        axes[1].plot(np.linalg.norm(np.diff(refined[:, person], axis=0), axis=-1),
                     color=colour, label=f"refined {person}")
    axes[0].set_title("Raw per-frame translation correction (no Gaussian filter)")
    axes[1].set_title("Frame-to-frame translation step")
    for axis in axes:
        axis.set_xlabel("frame"); axis.set_ylabel("metres"); axis.grid(alpha=.25); axis.legend()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return {"original": motion_stats(original), "refined": motion_stats(refined)}


def main() -> int:
    args = parse_args()
    started = time.time()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_count = len(common.discover_frames(args.dataset_root, args.split, args.max_frames))
    bundle = args.output / "pred_camera_refine_bundle.npz"
    images = args.output / "input_images"
    refined = args.output / "pred_camera_refined_tracks.npz"

    if args.force or not valid_bundle(bundle, frame_count):
        inference_summary = build_bundle(args, bundle, images)
        (args.output / "inference_manifest.json").write_text(
            json.dumps(inference_summary, indent=2) + "\n", encoding="utf-8"
        )
    else:
        inference_summary = json.loads((args.output / "inference_manifest.json").read_text())
        print(f"[pred-camera] reuse bundle: {bundle}", flush=True)

    if args.force or not refined.is_file():
        run_refine(args, bundle, refined)
    else:
        print(f"[pred-camera] reuse refined tracks: {refined}", flush=True)

    video = render_comparison(args, bundle, refined, images, args.output)
    videos_3d = render_3d_scene(args, bundle, refined, args.output)
    motion = plot_corrections(refined, args.output / "translation_corrections.png")
    refine_summary = json.loads(refined.with_suffix(".json").read_text())
    manifest = {
        **inference_summary,
        "checkpoint": str(args.checkpoint.resolve()),
        "bundle": str(bundle),
        "refined_tracks": str(refined),
        "comparison_video": str(video),
        "scene_3d_videos": videos_3d,
        "iterations": args.iterations,
        "temporal_weight": 0.0,
        "smooth_sigma_frames": 0.0,
        "warm_start": "previous correction remapped by predicted model slot ID",
        "mean_mask_iou_original": refine_summary["mean_mask_iou_original"],
        "mean_mask_iou_refined": refine_summary["mean_mask_iou_refined"],
        "accepted_frames": refine_summary["accepted_frames"],
        "motion": motion,
        "elapsed_seconds_this_run": time.time() - started,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
