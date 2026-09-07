#!/usr/bin/env python3
"""Run SMPL-only inference on organized MAMMA frames and make a mesh GIF.

The script intentionally does not load GT SMPL parameters or GT meshes.  Four
dataset cameras are sampled once, the model predicts cameras and SMPL meshes,
and one of those cameras is used as a fixed observation view for every GIF
frame.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent
TRAINING_DIR = REPO_DIR / "training"
for module_path in (REPO_DIR, TRAINING_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# Compatibility for old SMPL/chumpy model files.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for alias, value in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if alias not in np.__dict__:
        setattr(np, alias, value)
if not hasattr(np, "unicode"):
    np.unicode = str

from render_mesh_projection_cpu import (  # noqa: E402
    _project_mesh_to_frames,
    _render_one_frame,
)
from training.smpl_body import _decode_smpl_batch, _get_smpl_model  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


MESH_COLORS_RGBA = np.asarray(
    [
        (230, 57, 70, 255),
        (29, 126, 214, 255),
        (42, 157, 143, 255),
        (244, 162, 97, 255),
        (131, 56, 236, 255),
        (233, 196, 106, 255),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer predicted SMPL meshes for MAMMA frames and save a fixed-camera GIF."
    )
    parser.add_argument("--config", default="mamma_mask_dpt")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_DIR / "model/no_avg/checkpoint_47.pt"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_DIR / "MAMMA_markerless_multiple_people"),
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--num-input-views", type=int, default=4)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Render the K slots with highest presence probability; use 0 for threshold mode.",
    )
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_DIR / "outputs/no_avg_checkpoint47_markerless_first100_gif"),
    )
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def run_sort_key(path: Path) -> tuple[int, object]:
    match = re.fullmatch(r"runs_(\d+)", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def discover_frames(dataset_root: Path, split: str, max_frames: int) -> list[Path]:
    image_root = dataset_root / split / "out_image"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    frames = sorted(
        (
            path
            for path in image_root.iterdir()
            if path.is_dir() and path.name.startswith("runs_")
        ),
        key=run_sort_key,
    )
    if not frames:
        raise FileNotFoundError(f"No runs_* directories found under {image_root}")
    if max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    return frames[:max_frames]


def list_frame_images(frame_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"No images found in {frame_dir}")
    return images


def load_model(
    config_name: str,
    checkpoint_path: Path,
    device: torch.device,
    keep_person_mask: bool = False,
    keep_camera: bool = True,
):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(TRAINING_DIR / "config"),
    ):
        cfg = compose(config_name=config_name)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = instantiate(cfg.model, _recursive_=False)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    incompatible = model.load_state_dict(state_dict, strict=False)

    active_smpl_heads = [
        name
        for name in (
            "smpl_head",
            "smpl_multi_query_head",
            "smpl_multi_query_trans_head",
            "smpl_multi_query_trans_rot_head",
        )
        if getattr(model, name, None) is not None
    ]
    if not active_smpl_heads:
        raise RuntimeError(f"Config {config_name} has no active SMPL head")
    critical_missing = [
        key
        for key in incompatible.missing_keys
        if any(key.startswith(f"{head}.") for head in active_smpl_heads)
    ]
    if critical_missing:
        raise RuntimeError(
            "Checkpoint/config SMPL-head mismatch; first missing key: "
            f"{critical_missing[0]}"
        )

    # The SMPL head is required.  Dense outputs are unnecessary here.
    disabled_heads = [
        "depth_head",
        "point_head",
        "track_head",
        "smpl_dense_landmark_head",
    ]
    if not keep_person_mask:
        disabled_heads.append("person_mask_head")
    if not keep_camera:
        # 216M params / 825MiB that callers who never read `pose_enc` do not
        # need. Dropping it cannot change any SMPL output: the camera head is a
        # sibling of the SMPL head over the same aggregator tokens, and the
        # cached temporal path never invokes it at all.
        disabled_heads.append("camera_head")
    for head_name in disabled_heads:
        if hasattr(model, head_name):
            setattr(model, head_name, None)

    model.eval().to(device)
    return model, cfg, incompatible


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def choose_slots(
    probabilities: np.ndarray,
    top_k: int,
    threshold: float,
) -> np.ndarray:
    if top_k > 0:
        return np.argsort(-probabilities, kind="stable")[: min(top_k, probabilities.size)]
    return np.flatnonzero(probabilities >= threshold)


def decode_and_place_mesh(
    predictions: dict,
    slots: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = predictions["smpl_pose"][0, slots].float()
    betas = predictions["smpl_beta"][0, slots].float()
    translations = predictions.get("mesh_translate")
    if translations is None:
        translations = predictions.get("smpl_trans")
    if translations is None:
        raise KeyError("Model produced neither mesh_translate nor smpl_trans")
    translations = translations[0, slots].float().reshape(len(slots), 3)

    zeros = torch.zeros((len(slots), 3), dtype=torch.float32, device=device)
    joints, vertices = _decode_smpl_batch(
        pose_aa=poses.to(device=device, dtype=torch.float32),
        betas=betas.to(device=device, dtype=torch.float32),
        trans=zeros,
        genders=["neutral"] * len(slots),
        use_mamma=False,
    )
    # checkpoint_47 predicts mesh_rot in pose[:3], so decoded bodies are already
    # oriented in camera-0 coordinates. Re-anchor each decoded root to its
    # predicted mesh translation. scale_by_extrinsics=False means no scale divide.
    vertices = vertices - joints[:, :1, :] + translations[:, None, :]
    vertices_np = vertices.detach().cpu().numpy().astype(np.float32)

    neutral_model = _get_smpl_model(device, "neutral")
    base_faces = np.asarray(neutral_model.faces, dtype=np.int64)
    vertex_count = vertices_np.shape[1]
    all_faces = np.concatenate(
        [base_faces + person_index * vertex_count for person_index in range(len(slots))],
        axis=0,
    )
    all_vertices = vertices_np.reshape(-1, 3)
    vertex_colors = np.concatenate(
        [
            np.repeat(
                MESH_COLORS_RGBA[person_index % len(MESH_COLORS_RGBA)][None],
                vertex_count,
                axis=0,
            )
            for person_index in range(len(slots))
        ],
        axis=0,
    )
    return all_vertices, all_faces, vertex_colors


def tensor_image_to_bgr(image: torch.Tensor) -> np.ndarray:
    rgb = (
        image.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    return (rgb[:, :, ::-1] * 255.0).round().astype(np.uint8)


def save_gif(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("No rendered frames are available for GIF creation")
    duration_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))
    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs = discover_frames(dataset_root, args.dataset_split, args.max_frames)

    first_images = list_frame_images(frame_dirs[0])
    if args.num_input_views < 1 or args.num_input_views > len(first_images):
        raise ValueError(
            f"--num-input-views must be within [1, {len(first_images)}]"
        )
    rng = random.Random(args.seed)
    input_indices = sorted(rng.sample(range(len(first_images)), args.num_input_views))
    render_input_position = rng.randrange(len(input_indices))
    render_dataset_index = input_indices[render_input_position]
    render_camera_name = first_images[render_dataset_index].stem

    model, cfg, incompatible = load_model(
        args.config,
        checkpoint_path,
        device,
    )
    scale_by_extrinsics = bool(
        OmegaConf.select(cfg, "scale_by_extrinsics", default=True)
    )
    if scale_by_extrinsics:
        raise RuntimeError(
            "This script's direct placement is intended for no-avg-scale checkpoints "
            "(scale_by_extrinsics=False)."
        )

    print(f"[GIF] checkpoint={checkpoint_path}")
    print(f"[GIF] dataset={dataset_root / args.dataset_split}")
    print(f"[GIF] frames={len(frame_dirs)} input_indices={input_indices}")
    print(
        f"[GIF] fixed_render_camera={render_camera_name} "
        f"(input position {render_input_position}, seed={args.seed})"
    )
    print(
        f"[GIF] selection={'top-' + str(args.top_k) if args.top_k > 0 else 'threshold'} "
        f"missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )

    rendered_paths: list[Path] = []
    manifest_frames = []
    started = time.time()
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if autocast_enabled and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )

    for frame_number, frame_dir in enumerate(frame_dirs):
        all_images = list_frame_images(frame_dir)
        if max(input_indices) >= len(all_images):
            raise IndexError(
                f"{frame_dir.name} only has {len(all_images)} cameras; "
                f"index {max(input_indices)} was requested"
            )
        image_paths = [str(all_images[index]) for index in input_indices]
        images = load_and_preprocess_images(image_paths).to(device)

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                predictions = model(images)

        if "pose_enc" not in predictions:
            raise KeyError("Camera head did not produce pose_enc")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:],
        )
        logits = predictions.get("smpl_presence_logits")
        probabilities = (
            stable_sigmoid(logits[0].float().cpu().numpy())
            if logits is not None
            else np.ones(predictions["smpl_pose"].shape[1], dtype=np.float64)
        )
        slots = choose_slots(probabilities, args.top_k, args.presence_threshold)

        base_path = frames_dir / f"{frame_number:04d}_{frame_dir.name}_base.png"
        output_path = frames_dir / f"{frame_number:04d}_{frame_dir.name}.png"
        cv2.imwrite(str(base_path), tensor_image_to_bgr(images[render_input_position]))

        if slots.size:
            vertices, faces, vertex_colors = decode_and_place_mesh(
                predictions,
                slots,
                device,
            )
            projected, depth = _project_mesh_to_frames(
                vertices,
                extrinsic[0, render_input_position].float().cpu().numpy()[None],
                intrinsic[0, render_input_position].float().cpu().numpy()[None],
            )
            face_colors_bgr = vertex_colors[faces[:, 0]][:, [2, 1, 0]]
            _render_one_frame(
                str(base_path),
                str(output_path),
                projected[0],
                depth[0],
                faces,
                face_colors_bgr,
            )
        else:
            cv2.imwrite(
                str(output_path),
                tensor_image_to_bgr(images[render_input_position]),
            )
        base_path.unlink(missing_ok=True)

        rendered = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        label = (
            f"{frame_dir.name}  cam={render_camera_name}  "
            f"slots={','.join(map(str, slots.tolist()))}"
        )
        cv2.putText(
            rendered,
            label,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            label,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path), rendered)
        rendered_paths.append(output_path)
        manifest_frames.append(
            {
                "frame_index": frame_number,
                "run": frame_dir.name,
                "selected_slots": slots.tolist(),
                "selected_probabilities": [
                    float(probabilities[index]) for index in slots
                ],
                "rendered_png": str(output_path),
            }
        )

        del predictions, images
        torch.cuda.empty_cache()
        if (frame_number + 1) % max(1, args.log_every) == 0:
            elapsed = time.time() - started
            rate = (frame_number + 1) / max(elapsed, 1e-6)
            eta = (len(frame_dirs) - frame_number - 1) / max(rate, 1e-6)
            print(
                f"[GIF] {frame_number + 1}/{len(frame_dirs)} {frame_dir.name} "
                f"slots={slots.tolist()} rate={rate:.3f} frame/s "
                f"ETA={eta / 60.0:.1f} min"
            )

    gif_path = output_dir / "no_avg_checkpoint47_first100_pred_smpl.gif"
    save_gif(rendered_paths, gif_path, args.fps)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "config": args.config,
        "dataset": str(dataset_root / args.dataset_split),
        "uses_gt_mesh": False,
        "uses_gt_smpl": False,
        "camera_source": "model-predicted camera",
        "seed": args.seed,
        "input_camera_indices": input_indices,
        "input_camera_names": [
            first_images[index].stem for index in input_indices
        ],
        "fixed_render_camera_index": render_dataset_index,
        "fixed_render_camera_name": render_camera_name,
        "top_k": args.top_k,
        "presence_threshold": args.presence_threshold,
        "fps": args.fps,
        "frame_count": len(rendered_paths),
        "gif": str(gif_path),
        "elapsed_seconds": time.time() - started,
        "frames": manifest_frames,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[RESULT] GIF={gif_path}")
    print(f"[RESULT] frames={frames_dir}")
    print(f"[RESULT] manifest={manifest_path}")


if __name__ == "__main__":
    main()
