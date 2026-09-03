#!/usr/bin/env python3
"""Evaluate CK30 + 10-step predicted-mask translation refine on MAMMA Dance.

Each causal sample is evaluated exactly like the working demo: all 3 temporal
frames x 8 views are passed through the complete model together.  Geometry and
metrics use MAMMA's 10475-vertex SMPL-X model.  Refinement only changes XYZ
mesh translation; pose, shape, and root rotation remain frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "debug/0901_meeting"))
# nvdiffrast is shared from the PoseGAM environment; both environments use
# Python 3.10 and the same CUDA/PyTorch ABI on this machine.
sys.path.append("/train-data-3-hdd/yian/conda/envs/posegam/lib/python3.10/site-packages")

import eval_mamma_dance_mpjpe as base
import eval_mamma_dance_temporal_mpjpe_vpe as temporal
import infer_markerless_smpl_gif as common
from refine_mesh_translate_from_masks import (
    MultiPersonRenderer, clean_masks, mask_loss, smooth_correction, target_distance,
)
from infer_temporal_smpl_mesh_hungarian_mp4 import (
    gt_mesh_hungarian, select_prediction_frame,
)
from render_markerless_gt_smpl_3d_video import load_gt_smpl
from training import smpl_body
from training.data.dataset_util import (
    crop_image_depth_and_intrinsic_by_pp,
    resize_image_depth_and_intrinsic,
)
from training.smpl_body import (
    _decode_smpl_batch, compute_gt_mesh_rot, compute_gt_mesh_translate,
)


CHECKPOINT = REPO / "model/root/checkpoint_30.pt"
DATASET = Path("/train-data-3-hdd/yian/Multi_SMPL_0706/MAMMA_eval_dance")
SMPLX_MODEL = (
    REPO.parent / "Multi_SMPL_0706/body_models/smplx_locked_head/"
    "smplx/SMPLX_NEUTRAL.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=REPO / "eval/eval_results/checkpoint30_mask_refine10_absolute")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def gaussian_guard(raw: np.ndarray) -> np.ndarray:
    correction = smooth_correction(raw.astype(np.float32), 5.0)
    correction[..., :2] = np.clip(correction[..., :2], -0.05, 0.05)
    correction[..., 2] = np.clip(correction[..., 2], -0.60, 0.60)
    return correction


def valid_cache(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                len(data["pred_joints"]) == expected
                and data["raw_correction"].shape == (expected, 2, 3)
                and "preprocess" in data.files
                and str(data["preprocess"].item()) == "full_window_3x8_smplx"
            )
    except Exception:
        return False


def decode_mamma_at_mesh_translate(
    poses: np.ndarray,
    betas: np.ndarray,
    mesh_translate: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the same MAMMA SMPL-X geometry used by the 225-frame demo."""
    count = int(len(poses))
    with torch.inference_mode():
        pose_t = torch.as_tensor(poses, dtype=torch.float32, device=device)
        beta_t = torch.as_tensor(betas, dtype=torch.float32, device=device)
        translate_t = torch.as_tensor(mesh_translate, dtype=torch.float32, device=device)
        joints, vertices = _decode_smpl_batch(
            pose_aa=pose_t,
            betas=beta_t,
            trans=torch.zeros((count, 3), dtype=torch.float32, device=device),
            genders=["neutral"] * count,
            use_mamma=True,
        )
        offsets = translate_t.reshape(count, 3) - joints[:, 0]
        vertices = vertices + offsets[:, None]
        joints = joints + offsets[:, None]
    return (
        vertices.detach().cpu().numpy().astype(np.float32),
        joints.detach().cpu().numpy().astype(np.float32),
    )


def load_gt_mamma_in_prediction_gauge(
    archive_path: Path,
    camera_name: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Load GT in camera-0 gauge and decode it with MAMMA SMPL-X."""
    person_ids, poses, betas, translations, _ = load_gt_smpl(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        extrinsics = np.asarray(
            archive[f"cam_param_min/{camera_name}/extrinsics.worldToCamera12"],
            dtype=np.float32,
        ).reshape(3, 4)
    pose = torch.as_tensor(poses, dtype=torch.float32, device=device).unsqueeze(0)
    beta = torch.as_tensor(betas, dtype=torch.float32, device=device).unsqueeze(0)
    trans = torch.as_tensor(translations, dtype=torch.float32, device=device).unsqueeze(0)
    batch = {
        "smpl_pose": pose,
        "smpl_beta": beta,
        "smpl_trans": trans,
        "smpl_gender": torch.full((1, len(person_ids)), 2, dtype=torch.long, device=device),
        "raw_extrinsics": torch.as_tensor(extrinsics, device=device).reshape(1, 1, 3, 4),
        "avg_scale": torch.ones(1, dtype=torch.float32, device=device),
    }
    with torch.inference_mode():
        mesh_rot = compute_gt_mesh_rot(batch)
        mesh_translate = compute_gt_mesh_translate(
            batch, normalize_cam=True, use_mamma=True,
        )
    gauge_pose = pose.clone()
    gauge_pose[..., :3] = mesh_rot
    return decode_mamma_at_mesh_translate(
        gauge_pose[0].cpu().numpy(), beta[0].cpu().numpy(),
        mesh_translate[0].cpu().numpy(), device,
    )


def preprocess_like_training(
    image_path: str,
    intrinsic: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """Apply the trainer's principal-point crop/resize to a single RGB view."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise OSError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    intrinsic = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3).copy()
    image, _, intrinsic, _ = crop_image_depth_and_intrinsic_by_pp(
        image, None, intrinsic, np.asarray(image.shape[:2]), strict=False,
    )
    original_size = np.asarray(image.shape[:2])
    target = np.asarray([518, 518])
    image, _, intrinsic, _ = resize_image_depth_and_intrinsic(
        image, None, intrinsic, target, original_size,
        safe_bound=4, rescale_aug=False,
    )
    image, _, intrinsic, _ = crop_image_depth_and_intrinsic_by_pp(
        image, None, intrinsic, target, strict=True,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return tensor, intrinsic.astype(np.float32)


def load_training_window(
    dataset: Path,
    frames: list[Path],
    window_indices: list[int],
    image_ids: list[int],
    camera_names: list[str],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Load 3x8 trainer-style images plus current-frame refine cameras."""
    tensors = []
    current_intrinsics = None
    current_extrinsics = None
    for window_position, frame_index in enumerate(window_indices):
        frame_dir = frames[frame_index]
        image_paths = base.select_frame_images(frame_dir, image_ids)
        archive_path = dataset / "test/out_data" / f"{frame_dir.name}.npz"
        frame_intrinsics, frame_extrinsics = [], []
        with np.load(archive_path, allow_pickle=False) as archive:
            for camera_name, image_path in zip(camera_names, image_paths):
                intrinsic = np.asarray(
                    archive[f"cam_param_min/{camera_name}/intrinsics.K_flat9"],
                    dtype=np.float32,
                ).reshape(3, 3)
                tensor, processed_intrinsic = preprocess_like_training(
                    image_path, intrinsic,
                )
                tensors.append(tensor)
                frame_intrinsics.append(processed_intrinsic)
                frame_extrinsics.append(np.asarray(
                    archive[f"cam_param_min/{camera_name}/extrinsics.worldToCamera12"],
                    dtype=np.float32,
                ).reshape(3, 4))
        if window_position == len(window_indices) - 1:
            current_intrinsics = np.stack(frame_intrinsics)
            current_extrinsics = np.stack(frame_extrinsics)

    homogeneous = np.tile(np.eye(4, dtype=np.float32), (len(camera_names), 1, 1))
    homogeneous[:, :3] = current_extrinsics
    relative = homogeneous @ np.linalg.inv(homogeneous[0])
    return torch.stack(tensors), relative[:, :3], current_intrinsics


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    cache_root = output / "sequence_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    segments = temporal.parse_sequence_segments(dataset / "test/logs/process.log")
    if args.max_sequences is not None:
        segments = segments[:args.max_sequences]
    if args.max_frames is not None:
        remaining = max(0, int(args.max_frames))
        limited_segments = []
        for segment in segments:
            if remaining == 0:
                break
            count = min(remaining, segment["end"] - segment["start"] + 1)
            limited_segments.append({
                **segment,
                "end": segment["start"] + count - 1,
            })
            remaining -= count
        segments = limited_segments
    if not segments:
        raise ValueError("No frames selected for evaluation")
    frames = base.discover_frames(dataset, "test")
    views = [f"IOI_{index:02d}" for index in range(1, 9)]
    image_ids = list(range(8))
    if not SMPLX_MODEL.is_file():
        raise FileNotFoundError(f"MAMMA SMPL-X model not found: {SMPLX_MODEL}")
    smpl_body._SMPLX_MODEL_PATHS["neutral"] = str(SMPLX_MODEL.resolve())
    smpl_body._SMPLX_MODEL_CACHE.clear()

    pending = []
    for segment in segments:
        count = segment["end"] - segment["start"] + 1
        cache = cache_root / f"{segment['name']}.npz"
        if args.force or not valid_cache(cache, count):
            pending.append(segment)
    model = None
    renderer = None
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    if pending:
        model, _, incompatible = common.load_model(
            "mamma_harmony4d_mask_dpt", args.checkpoint.resolve(), device,
            keep_person_mask=True,
        )
        with np.load(SMPLX_MODEL, allow_pickle=True) as body_archive:
            faces = np.asarray(body_archive["f"], dtype=np.int32)
            vertices_per_person = int(np.asarray(body_archive["v_template"]).shape[0])
        renderer = MultiPersonRenderer(
            faces, 2, vertices_per_person, 128, 128, device,
        )
        print(
            f"[eval] checkpoint missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}", flush=True,
        )

    started_all = time.time()
    for sequence_number, segment in enumerate(segments, 1):
        start, end = segment["start"], segment["end"]
        count = end - start + 1
        cache_path = cache_root / f"{segment['name']}.npz"
        if not args.force and valid_cache(cache_path, count):
            print(f"[eval] reuse {sequence_number}/{len(segments)} {segment['name']}", flush=True)
            continue

        previous = torch.zeros((2, 3), dtype=torch.float32, device=device)
        previous2 = previous.clone()
        pred_joints_all, gt_joints_all, pred_translate_all = [], [], []
        raw_all, slots_all, original_iou_all, optimized_iou_all = [], [], [], []
        orientation_flip = None
        sequence_started = time.time()

        for local_index, global_index in enumerate(range(start, end + 1)):
            frame_dir = frames[global_index]
            # Exact causal demo route: run the full aggregator on 3 x 8 images.
            window_indices = [
                max(start, index)
                for index in range(global_index - 2, global_index + 1)
            ]
            images, extrinsics, intrinsics = load_training_window(
                dataset, frames, window_indices, image_ids, views,
            )
            images = images.to(device)
            metadata = {
                "temporal_num_frames": torch.tensor([3], dtype=torch.long, device=device),
                "views_per_frame": torch.tensor([8], dtype=torch.long, device=device),
            }
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                all_predictions = model(images, smpl_inputs=metadata)
            prediction = select_prediction_frame(all_predictions, 2)
            mask_logits = prediction.get("person_mask_logits")
            if mask_logits is None:
                raise RuntimeError("Model did not return person_mask_logits")
            probabilities = torch.sigmoid(mask_logits[0].float())
            if probabilities.shape[-2:] != (128, 128):
                probabilities = F.interpolate(
                    probabilities.flatten(0, 1).unsqueeze(1), size=(128, 128),
                    mode="bilinear", align_corners=False,
                ).squeeze(1).reshape(8, -1, 128, 128)

            pose_all = prediction["smpl_pose"][0].float().cpu().numpy()
            beta_all = prediction["smpl_beta"][0].float().cpu().numpy()
            translate_all = prediction["mesh_translate"][0].float().cpu().numpy()
            pred_vertices_all, pred_joints_all_slots = decode_mamma_at_mesh_translate(
                pose_all, beta_all, translate_all, device,
            )
            archive = dataset / "test/out_data" / f"{frame_dir.name}.npz"
            gt_vertices, gt_joints = load_gt_mamma_in_prediction_gauge(
                archive, views[0], device,
            )
            selected, _ = gt_mesh_hungarian(pred_vertices_all, gt_vertices)
            pose, beta, translate = (
                pose_all[selected], beta_all[selected], translate_all[selected]
            )
            pred_vertices = pred_vertices_all[selected]
            pred_joints = pred_joints_all_slots[selected]
            target_np = clean_masks(
                probabilities[:, selected].cpu().numpy(), 0.5,
            )
            # DPT masks are 128x128 while trainer-preprocessed RGB is 518x518.
            intrinsics[:, 0, :] *= 128.0 / 518.0
            intrinsics[:, 1, :] *= 128.0 / 518.0
            vertices_t = torch.as_tensor(pred_vertices, device=device)
            extrinsics_t = torch.as_tensor(extrinsics, device=device)
            intrinsics_t = torch.as_tensor(intrinsics, device=device)

            def render(delta: torch.Tensor) -> torch.Tensor:
                rendered = []
                for view_index in range(8):
                    matrix = torch.eye(4, dtype=torch.float32, device=device)
                    matrix[:3, :4] = extrinsics_t[view_index]
                    rendered.append(renderer(vertices_t, delta, matrix, intrinsics_t[view_index]))
                result = torch.stack(rendered)
                return torch.flip(result, dims=(-2,)) if orientation_flip else result

            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            distance = torch.as_tensor(target_distance(target_np), device=device)
            valid = target.sum(dim=(-2, -1)) >= 24
            zero = torch.zeros((2, 3), dtype=torch.float32, device=device)
            if orientation_flip is None:
                with torch.no_grad():
                    normal = render(zero)
                    normal_loss, _ = mask_loss(normal, target, distance, valid)
                    flipped_loss, _ = mask_loss(torch.flip(normal, dims=(-2,)), target, distance, valid)
                orientation_flip = bool(flipped_loss < normal_loss)
                print(f"[eval] {segment['name']} flip_y={orientation_flip}", flush=True)

            if bool(valid.any()):
                with torch.no_grad():
                    original_render = render(zero)
                    original_loss, original_terms = mask_loss(
                        original_render, target, distance, valid,
                    )
                delta = (previous.detach().clone() if local_index else zero).requires_grad_(True)
                optimizer = torch.optim.Adam([delta], lr=0.025)
                best_delta, best_loss = zero.clone(), float(original_loss)
                best_iou = float(original_terms["iou"])
                for _ in range(args.iterations):
                    optimizer.zero_grad(set_to_none=True)
                    rendered = render(delta)
                    data_loss, terms = mask_loss(rendered, target, distance, valid)
                    total = data_loss + 0.015 * (delta / 0.6).square().mean()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_([delta], 5.0)
                    if float(data_loss.detach()) < best_loss:
                        best_loss = float(data_loss.detach())
                        best_delta = delta.detach().clone()
                        best_iou = float(terms["iou"].detach())
                    optimizer.step()
                    with torch.no_grad():
                        delta.clamp_(-0.6, 0.6)
                chosen = best_delta if best_loss + 1e-4 < float(original_loss) else zero
                original_iou = float(original_terms["iou"])
            else:
                chosen = previous.detach().clone()
                original_iou = best_iou = float("nan")
            previous2, previous = previous, chosen.detach()
            pred_joints_all.append(pred_joints)
            gt_joints_all.append(gt_joints)
            pred_translate_all.append(translate)
            raw_all.append(chosen.detach().cpu().numpy())
            slots_all.append(selected)
            original_iou_all.append(original_iou)
            optimized_iou_all.append(best_iou)

            del images, all_predictions, prediction, mask_logits, probabilities
            completed = local_index + 1
            if completed % args.log_every == 0 or completed == count:
                elapsed = time.time() - sequence_started
                global_done = global_index + 1
                eta = (len(frames) - global_done) * elapsed / completed / 60.0
                print(
                    f"[eval] seq {sequence_number}/{len(segments)} "
                    f"{completed}/{count} global={global_done}/{len(frames)} "
                    f"ETA~{eta:.1f} min", flush=True,
                )

        np.savez_compressed(
            cache_path,
            pred_joints=np.stack(pred_joints_all).astype(np.float32),
            gt_joints=np.stack(gt_joints_all).astype(np.float32),
            pred_translate=np.stack(pred_translate_all).astype(np.float32),
            raw_correction=np.stack(raw_all).astype(np.float32),
            selected_slots=np.stack(slots_all).astype(np.int64),
            original_iou=np.asarray(original_iou_all, dtype=np.float32),
            optimized_iou=np.asarray(optimized_iou_all, dtype=np.float32),
            global_indices=np.arange(start, end + 1, dtype=np.int64),
            preprocess=np.asarray("full_window_3x8_smplx"),
        )
        print(f"[eval] cached {cache_path}", flush=True)

    rows = []
    original_sum = refined_sum = root_aligned_sum = 0.0
    joint_count = 0
    for segment in segments:
        cache_path = cache_root / f"{segment['name']}.npz"
        with np.load(cache_path, allow_pickle=False) as data:
            pred_joints = data["pred_joints"].astype(np.float32)
            gt_joints = data["gt_joints"].astype(np.float32)
            raw = data["raw_correction"].astype(np.float32)
            indices = data["global_indices"].astype(np.int64)
        correction = gaussian_guard(raw)
        refined_joints = pred_joints + correction[:, :, None]
        original_error = np.linalg.norm(pred_joints - gt_joints, axis=-1) * 1000.0
        refined_error = np.linalg.norm(refined_joints - gt_joints, axis=-1) * 1000.0
        pred_ra = pred_joints - pred_joints[:, :, :1]
        gt_ra = gt_joints - gt_joints[:, :, :1]
        root_error = np.linalg.norm(pred_ra - gt_ra, axis=-1) * 1000.0
        original_sum += float(original_error.sum())
        refined_sum += float(refined_error.sum())
        root_aligned_sum += float(root_error.sum())
        joint_count += int(refined_error.size)
        for local, global_index in enumerate(indices):
            rows.append({
                "frame": f"runs_{global_index:05d}", "sequence": segment["name"],
                "absolute_mpjpe_original_mm": float(original_error[local].mean()),
                "absolute_mpjpe_refined_mm": float(refined_error[local].mean()),
                "root_aligned_mpjpe_mm": float(root_error[local].mean()),
                "mean_correction_m": float(np.linalg.norm(correction[local], axis=-1).mean()),
            })

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "method": "checkpoint_30 + predicted-mask translation refine",
        "inference_route": "full causal window: 3 frames x 8 views jointly",
        "body_model": "MAMMA SMPL-X (10475 vertices)",
        "checkpoint": str(args.checkpoint.resolve()), "views": views,
        "frames": len(rows), "sequences": len(segments), "people": len(rows) * 2,
        "iterations": args.iterations, "smooth_sigma_frames": 5.0,
        "xy_guard_m": 0.05, "z_guard_m": 0.60,
        "absolute_mpjpe_original_mm": original_sum / joint_count,
        "absolute_mpjpe_refined_mm": refined_sum / joint_count,
        "root_aligned_mpjpe_mm": root_aligned_sum / joint_count,
        "per_frame_csv": str(csv_path), "elapsed_seconds_this_run": time.time() - started_all,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
