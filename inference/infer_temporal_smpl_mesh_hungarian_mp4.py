#!/usr/bin/env python3
"""Temporal SMPL visualization with GT-pinned Hungarian assignment.

All predicted slots are decoded to meshes. A predicted-slot x GT-person cost
matrix is built from corresponding-vertex 3D distance, then Hungarian matching
picks one unique predicted slot per GT identity. Presence top-k is never used.

Output frame t uses the last decoder position of the causal window
``[t-2, t-1, t]``; frame zero is repeated for the two warm-up windows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

# This module lives in inference/; its repo-root siblings (infer_markerless_*,
# eval_mamma_dance_mpjpe, training/, vggt/) are imported by plain name, so the
# repo root has to be importable before those imports run.
REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import infer_markerless_smpl_3d_gif as render3d
import infer_markerless_smpl_gif as common
from render_markerless_gt_smpl_3d_video import load_gt_smpl
from training import smpl_body
from training.smpl_body import (
    _decode_smpl_batch,
    _get_smpl_model,
    compute_gt_mesh_rot,
    compute_gt_mesh_translate,
)
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal SMPL MP4 with GT-mesh Hungarian assignment."
    )
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smpl-model", default=None)
    parser.add_argument("--max-frames", type=int, default=225)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Index of the first runs_* directory to use. With --max-frames this "
             "selects one MAMMA sequence out of the concatenated eval set; the causal "
             "window is padded from the first selected frame, so each sequence starts "
             "clean instead of leaking the previous take.",
    )
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--num-input-views", type=int, default=8)
    parser.add_argument("--input-indices", type=int, nargs="+", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--azimuth-deg", type=float, default=35.0)
    parser.add_argument("--elevation-deg", type=float, default=28.0)
    parser.add_argument("--fov-deg", type=float, default=38.0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--translate-refine-mask",
        action="store_true",
        help=(
            "Keep predicted pose/shape fixed and refine mesh_translate against "
            "the checkpoint's per-person masks in all input views."
        ),
    )
    parser.add_argument("--translate-refine-iters", type=int, default=30)
    parser.add_argument("--translate-refine-size", type=int, default=112)
    parser.add_argument("--translate-refine-chamfer-points", type=int, default=100)
    parser.add_argument("--translate-refine-lr", type=float, default=0.1)
    parser.add_argument(
        "--feature-cache",
        dest="feature_cache",
        action="store_true",
        default=True,
        help=(
            "Encode each frame's views with the aggregator exactly once and keep "
            "the previous clip_length-1 frames' final tokens in a rolling cache. "
            "The temporal path folds T into the batch axis, so per-timestep "
            "encoding is independent and the cached window is numerically the "
            "same input the full forward would build."
        ),
    )
    parser.add_argument(
        "--no-feature-cache",
        dest="feature_cache",
        action="store_false",
        help="Re-encode the whole causal window every frame (original path).",
    )
    return parser.parse_args()


_DR_CONTEXTS: dict[str, object] = {}


def _load_nvdiffrast():
    """Load the PoseGAM nvdiffrast install without replacing this env's torch."""
    try:
        import nvdiffrast.torch as dr  # type: ignore
        return dr
    except ModuleNotFoundError:
        import sys

        posegam_site = Path(
            "/train-data-3-hdd/yian/conda/envs/posegam/lib/python3.10/site-packages"
        )
        if not posegam_site.is_dir():
            raise RuntimeError(
                "Mask translation refinement needs nvdiffrast. The PoseGAM "
                f"installation was not found at {posegam_site}."
            )
        # Append (do not prepend): torch must continue to come from the mamma env.
        sys.path.append(str(posegam_site))
        import nvdiffrast.torch as dr  # type: ignore
        return dr


def _dr_context(device: torch.device):
    key = str(device)
    if key not in _DR_CONTEXTS:
        dr = _load_nvdiffrast()
        _DR_CONTEXTS[key] = dr.RasterizeCudaContext(device=device)
    return _DR_CONTEXTS[key]


def preprocess_intrinsics(
    intrinsic: np.ndarray,
    original_width: int,
    original_height: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Apply load_and_preprocess_images(mode='crop') resize/crop to K."""
    resized_width = 518
    resized_height = round(original_height * (resized_width / original_width) / 14) * 14
    crop_y = max(0, (resized_height - 518) // 2)
    pad_y = max(0, (output_height - min(resized_height, 518)) // 2)
    pad_x = max(0, (output_width - resized_width) // 2)
    result = np.asarray(intrinsic, dtype=np.float32).copy()
    result[0, :] *= resized_width / float(original_width)
    result[1, :] *= resized_height / float(original_height)
    result[0, 2] += pad_x
    result[1, 2] += pad_y - crop_y
    return result


def load_refine_cameras(
    archive_path: Path,
    camera_names: list[str],
    image_paths: list[Path],
    mask_height: int,
    mask_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cam0-gauge world-to-view matrices and preprocessed intrinsics."""
    extrinsics, intrinsics = [], []
    with np.load(archive_path, allow_pickle=False) as archive:
        for camera_name, image_path in zip(camera_names, image_paths):
            extrinsics.append(
                np.asarray(
                    archive[f"cam_param_min/{camera_name}/extrinsics.worldToCamera12"],
                    dtype=np.float32,
                ).reshape(3, 4)
            )
            intrinsic = np.asarray(
                archive[f"cam_param_min/{camera_name}/intrinsics.K_flat9"],
                dtype=np.float32,
            ).reshape(3, 3)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Failed to read input image: {image_path}")
            intrinsics.append(
                preprocess_intrinsics(
                    intrinsic,
                    image.shape[1],
                    image.shape[0],
                    mask_width,
                    mask_height,
                )
            )

    homogeneous = np.tile(np.eye(4, dtype=np.float32), (len(extrinsics), 1, 1))
    homogeneous[:, :3] = np.stack(extrinsics)
    relative = homogeneous @ np.linalg.inv(homogeneous[0])
    return relative[:, :3], np.stack(intrinsics)


def render_multiview_silhouettes(
    local_vertices: torch.Tensor,
    mesh_translate: torch.Tensor,
    relative_extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    faces: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Differentiably render one cam0-gauge mesh into all OpenCV cameras."""
    dr = _load_nvdiffrast()
    vertices = local_vertices + mesh_translate.reshape(1, 3)
    rotation = relative_extrinsics[:, :3, :3]
    translation = relative_extrinsics[:, :3, 3]
    vertices_camera = torch.einsum("vij,nj->vni", rotation, vertices) + translation[:, None]
    z = vertices_camera[..., 2].clamp(min=1e-4)
    fx = intrinsics[:, 0, 0, None]
    fy = intrinsics[:, 1, 1, None]
    cx = intrinsics[:, 0, 2, None]
    cy = intrinsics[:, 1, 2, None]
    clip = torch.stack(
        [
            (2.0 * fx / width) * vertices_camera[..., 0]
            + (2.0 * cx / width - 1.0) * z,
            (2.0 * fy / height) * vertices_camera[..., 1]
            + (2.0 * cy / height - 1.0) * z,
            z,
            z,
        ],
        dim=-1,
    )
    rast, _ = dr.rasterize(
        _dr_context(mesh_translate.device),
        clip,
        faces,
        resolution=[height, width],
        grad_db=True,
    )
    hard_mask = (rast[..., -1:] > 0).to(clip.dtype)
    silhouette = dr.antialias(hard_mask, rast, clip, faces)[..., 0]
    # Antialias can produce tiny negative values and, for a triangle crossing a
    # near-plane singularity, an isolated NaN. Neither is meaningful occupancy.
    return torch.nan_to_num(silhouette, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def mask_iou(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (rendered * target).sum(dim=(-2, -1))
    union = rendered.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean()


def refine_mesh_translate_with_masks(
    vertices_at_translate: np.ndarray,
    initial_translate: np.ndarray,
    target_masks: torch.Tensor,
    relative_extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    faces: np.ndarray,
    device: torch.device,
    max_iters: int,
    refine_size: int,
    chamfer_points: int,
    learning_rate: float,
) -> tuple[np.ndarray, float, float]:
    """PoseGAM-style fixed-pose translation refinement, jointly over all views."""
    if device.type != "cuda":
        raise RuntimeError("nvdiffrast translation refinement requires CUDA")
    masks = target_masks.detach().float().to(device).clamp(0.0, 1.0)
    source_height, source_width = masks.shape[-2:]
    if refine_size > 0:
        scale = float(refine_size) / max(source_height, source_width)
        height = max(1, round(source_height * scale))
        width = max(1, round(source_width * scale))
        masks = F.interpolate(
            masks[:, None], size=(height, width), mode="bilinear", align_corners=False
        )[:, 0]
    else:
        height, width = source_height, source_width
        scale = 1.0

    camera_k = torch.as_tensor(intrinsics, device=device, dtype=torch.float32).clone()
    camera_k[:, 0, :] *= scale
    camera_k[:, 1, :] *= scale
    camera_e = torch.as_tensor(relative_extrinsics, device=device, dtype=torch.float32)
    face_t = torch.as_tensor(faces, device=device, dtype=torch.int32)
    initial_t = torch.as_tensor(initial_translate, device=device, dtype=torch.float32)
    local_vertices = torch.as_tensor(
        vertices_at_translate - np.asarray(initial_translate)[None],
        device=device,
        dtype=torch.float32,
    )
    translate = initial_t.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([translate], lr=learning_rate)

    coords_y = torch.arange(height, device=device, dtype=torch.float32)
    coords_x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
    target_samples = []
    for view_mask in masks:
        weights = view_mask.flatten()
        count = min(chamfer_points, int((weights > 0.05).sum().item()))
        if count > 0 and weights.sum() > 0:
            indices = torch.multinomial(weights / weights.sum(), count, replacement=False)
            target_samples.append((coords[indices], weights[indices]))
        else:
            target_samples.append(None)

    with torch.no_grad():
        initial_render = render_multiview_silhouettes(
            local_vertices, translate, camera_e, camera_k, face_t, height, width
        )
        initial_iou = float(mask_iou(initial_render, masks).item())
    best_iou = initial_iou
    best_translate = translate.detach().clone()
    stale_checks = 0
    previous_check_iou = initial_iou

    for iteration in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        rendered = render_multiview_silhouettes(
            local_vertices, translate, camera_e, camera_k, face_t, height, width
        )
        iou = mask_iou(rendered, masks)
        chamfer = torch.zeros((), device=device)
        valid_views = 0
        if chamfer_points > 0:
            for view_index, target_sample in enumerate(target_samples):
                if target_sample is None:
                    continue
                render_weights = rendered[view_index].flatten()
                count = min(chamfer_points, int((render_weights > 0).sum().item()))
                if count <= 0 or render_weights.sum() <= 0:
                    continue
                indices = torch.multinomial(
                    render_weights / render_weights.sum(), count, replacement=False
                )
                render_coords = coords[indices]
                target_coords, target_weights = target_sample
                distances = torch.cdist(render_coords, target_coords)
                temperature = 0.1
                render_to_target = (
                    torch.softmax(-distances / temperature, dim=1) * distances
                ).sum(dim=1)
                target_to_render = (
                    torch.softmax(-distances.T / temperature, dim=1) * distances.T
                ).sum(dim=1)
                chamfer = chamfer + (
                    0.6
                    * (render_to_target * render_weights[indices]).sum()
                    / (render_weights[indices].sum() + 1e-8)
                    + 0.4
                    * (target_to_render * target_weights).sum()
                    / (target_weights.sum() + 1e-8)
                ) / 100.0
                valid_views += 1
        if valid_views:
            chamfer = chamfer / valid_views
        loss = 0.6 * (1.0 - iou) + 0.4 * chamfer
        if not loss.requires_grad:
            break
        current_iou = float(iou.detach().item())
        if current_iou > best_iou:
            best_iou = current_iou
            best_translate = translate.detach().clone()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 10 == 0:
            if current_iou - previous_check_iou < 1e-4:
                stale_checks += 1
            else:
                stale_checks = 0
            previous_check_iou = current_iou
            if stale_checks >= 2:
                break
    return best_translate.cpu().numpy().astype(np.float32), initial_iou, best_iou


def select_prediction_frame(predictions: dict, index: int) -> dict:
    selected = {}
    for key, value in predictions.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > index:
            selected[key] = value[index : index + 1]
        else:
            selected[key] = value
    return selected


def causal_window_indices(frame_index: int, clip_length: int) -> list[int]:
    return [
        max(0, index)
        for index in range(frame_index - clip_length + 1, frame_index + 1)
    ]


class AggregatorTokenCache:
    """Rolling cache of final aggregator tokens for the previous frames.

    ``VGGT.forward`` folds the temporal axis into the batch axis before the
    aggregator (``[B,T*V,...] -> [B*T,V,...]``), so global attention never
    crosses a timestep and each frame's tokens depend only on that frame's
    views.  Encoding frame ``t`` once and replaying the stored tokens for
    ``t-1`` and ``t-2`` therefore reconstructs exactly the tensor the
    full-window forward would have produced, at 1/clip_length the aggregator
    and JPEG-decode cost.
    """

    def __init__(self, clip_length: int) -> None:
        if clip_length < 1:
            raise ValueError("clip_length must be >= 1")
        self.clip_length = int(clip_length)
        self._tokens: list[torch.Tensor] = []

    def clear(self) -> None:
        self._tokens.clear()

    def window(self, current: torch.Tensor) -> torch.Tensor:
        """Build the [T*B, V, N, C] causal window ending at ``current``.

        Warm-up frames repeat the oldest cached frame, which is what
        ``causal_window_indices`` does by clamping negative indices to the
        first frame of the run.
        """
        history = list(self._tokens)
        padded = [history[0] if history else current] * (
            self.clip_length - 1 - len(history)
        )
        window = padded + history + [current]
        if len(window) != self.clip_length:
            raise AssertionError((len(window), self.clip_length))
        return torch.cat(window, dim=0)

    def append(self, current: torch.Tensor) -> None:
        self._tokens.append(current)
        del self._tokens[: max(0, len(self._tokens) - (self.clip_length - 1))]


def cached_temporal_forward(
    model,
    cache: AggregatorTokenCache,
    images: torch.Tensor,
    smpl_inputs: dict,
    want_person_mask: bool,
) -> dict:
    """One aggregator pass over the current frame + cached temporal head.

    ``images`` holds only the current frame's views, shaped [V,3,H,W].  The
    returned dict carries the head outputs for the whole cached window, in the
    same layout ``VGGT.forward`` returns, so ``select_prediction_frame`` picks
    the current frame the same way in both paths.
    """
    if images.ndim == 4:
        images = images.unsqueeze(0)
    current_tokens, patch_start_idx, _ = model.aggregator(images)
    current_final = current_tokens[-1]
    window_features = [cache.window(current_final)]
    with torch.cuda.amp.autocast(enabled=False):
        predictions = dict(
            model.smpl_multi_query_trans_rot_head(
                window_features,
                patch_start_idx=patch_start_idx,
                smpl_inputs=smpl_inputs,
            )
        )
        if want_person_mask:
            if model.person_mask_head is None:
                raise RuntimeError("The loaded model has no person-mask head")
            if model.person_mask_head_type != "dpt":
                raise RuntimeError(
                    "Cached mask decoding supports the DPT mask head only, got "
                    f"{model.person_mask_head_type}"
                )
            # Masks are only consumed for the current frame, so decode the last
            # window position against the tokens just encoded for it.
            predictions["person_mask_logits"] = model.person_mask_head(
                predictions["person_tokens"][-1:],
                current_tokens,
                images=images,
                patch_start_idx=patch_start_idx,
            )
    cache.append(current_final.detach())
    return predictions


def load_gt_mesh_in_prediction_gauge(
    archive_path: Path,
    camera_name: str,
    device: torch.device,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Decode GT in the camera-0 mesh_rot/mesh_translate prediction gauge."""
    person_ids, poses, betas, translations, _ = load_gt_smpl(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        extrinsics = np.asarray(
            archive[f"cam_param_min/{camera_name}/extrinsics.worldToCamera12"],
            dtype=np.float32,
        ).reshape(3, 4)

    pose = torch.as_tensor(poses, device=device, dtype=torch.float32).unsqueeze(0)
    beta = torch.as_tensor(betas, device=device, dtype=torch.float32).unsqueeze(0)
    trans = torch.as_tensor(
        translations, device=device, dtype=torch.float32
    ).unsqueeze(0)
    batch = {
        "smpl_pose": pose,
        "smpl_beta": beta,
        "smpl_trans": trans,
        "smpl_gender": torch.full(
            (1, len(person_ids)), 2, device=device, dtype=torch.long
        ),
        "raw_extrinsics": torch.as_tensor(
            extrinsics, device=device, dtype=torch.float32
        ).reshape(1, 1, 3, 4),
        "avg_scale": torch.ones(1, device=device, dtype=torch.float32),
    }
    with torch.inference_mode():
        mesh_rot = compute_gt_mesh_rot(batch)
        mesh_translate = compute_gt_mesh_translate(
            batch, normalize_cam=True, use_mamma=False
        )
        gauge_pose = pose.clone()
        gauge_pose[..., :3] = mesh_rot
        flat_pose = gauge_pose.reshape(-1, gauge_pose.shape[-1])
        flat_beta = beta.reshape(-1, beta.shape[-1])
        zero_trans = torch.zeros(
            (len(person_ids), 3), device=device, dtype=torch.float32
        )
        joints, vertices = _decode_smpl_batch(
            pose_aa=flat_pose,
            betas=flat_beta,
            trans=zero_trans,
            genders=["neutral"] * len(person_ids),
            use_mamma=False,
        )
        offsets = mesh_translate.reshape(-1, 3) - joints[:, 0, :]
        vertices = vertices + offsets[:, None, :]
        joints = joints + offsets[:, None, :]
    return (
        person_ids,
        vertices.detach().cpu().numpy().astype(np.float32),
        joints.detach().cpu().numpy().astype(np.float32),
        gauge_pose[0].detach().cpu().numpy().astype(np.float32),
        beta[0].detach().cpu().numpy().astype(np.float32),
        mesh_translate[0].detach().cpu().numpy().astype(np.float32),
    )


def gt_mesh_hungarian(
    predicted_vertices: np.ndarray,
    gt_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign one unique predicted mesh to every GT mesh."""
    if predicted_vertices.shape[1:] != gt_vertices.shape[1:]:
        raise ValueError(
            "Predicted and GT SMPL topology differ: "
            f"{predicted_vertices.shape} vs {gt_vertices.shape}"
        )
    delta = (
        predicted_vertices[:, None].astype(np.float64)
        - gt_vertices[None].astype(np.float64)
    )
    cost = np.linalg.norm(delta, axis=-1).mean(axis=-1)
    predicted_indices, gt_indices = linear_sum_assignment(cost)
    if len(gt_indices) != len(gt_vertices):
        raise RuntimeError(
            f"Hungarian matched {len(gt_indices)}/{len(gt_vertices)} GT people"
        )
    slot_by_gt = np.full(len(gt_vertices), -1, dtype=np.int64)
    slot_by_gt[gt_indices] = predicted_indices
    if np.any(slot_by_gt < 0) or len(np.unique(slot_by_gt)) != len(slot_by_gt):
        raise RuntimeError(f"Invalid one-to-one Hungarian result: {slot_by_gt}")
    return slot_by_gt, cost


def decode_params_at_mesh_translate(
    poses: np.ndarray,
    betas: np.ndarray,
    mesh_translate: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SMPL pose/shape, then anchor each pelvis to mesh_translate."""
    count = int(len(poses))
    with torch.inference_mode():
        pose_t = torch.as_tensor(poses, device=device, dtype=torch.float32)
        beta_t = torch.as_tensor(betas, device=device, dtype=torch.float32)
        translate_t = torch.as_tensor(
            mesh_translate, device=device, dtype=torch.float32
        ).reshape(count, 3)
        joints, vertices = _decode_smpl_batch(
            pose_aa=pose_t,
            betas=beta_t,
            trans=torch.zeros((count, 3), device=device, dtype=torch.float32),
            genders=["neutral"] * count,
            use_mamma=False,
        )
        offsets = translate_t - joints[:, 0, :]
        vertices = vertices + offsets[:, None, :]
        joints = joints + offsets[:, None, :]
    return (
        vertices.detach().cpu().numpy().astype(np.float32),
        joints.detach().cpu().numpy().astype(np.float32),
    )


def label_frame(image: np.ndarray, title: str, subtitle: str, color: tuple) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 52), (31, 27, 27), -1)
    cv2.putText(
        labeled, title, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        labeled, subtitle, (14, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
        (190, 190, 190), 1, cv2.LINE_AA,
    )
    return labeled


def open_raw_mp4_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> subprocess.Popen:
    """Open an ffmpeg stdin writer for constant-frame-rate H.264."""
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ],
        stdin=subprocess.PIPE,
    )


def close_raw_mp4_writer(writer: subprocess.Popen, output_path: Path) -> None:
    assert writer.stdin is not None
    writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed ({return_code}) for {output_path}")


def input_mosaic(
    frame_dir: Path,
    input_indices: list[int],
    width: int,
    height: int,
) -> np.ndarray:
    paths = common.list_frame_images(frame_dir)
    selected = [paths[index] for index in input_indices]
    columns = 4 if len(selected) >= 4 else max(1, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    tile_width = max(1, width // columns)
    tile_height = max(1, height // rows)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for position, path in enumerate(selected):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Failed to read input image: {path}")
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(
            image,
            (
                max(1, round(image.shape[1] * scale)),
                max(1, round(image.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(position, columns)
        x0 = column * tile_width + (tile_width - resized.shape[1]) // 2
        y0 = row * tile_height + (tile_height - resized.shape[0]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cv2.putText(
            canvas,
            path.stem,
            (column * tile_width + 6, row * tile_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def encode_mp4(frames_dir: Path, output_path: Path, fps: float) -> None:
    """Encode browser-friendly constant-frame-rate H.264."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(frames_dir / "%04d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    if args.clip_length != 3:
        raise ValueError("This checkpoint was trained with clip_length=3")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    device = torch.device(args.device)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    pure_frames_dir = output_dir / "pred_frames"
    side_frames_dir = output_dir / "frames"
    pure_frames_dir.mkdir(parents=True, exist_ok=True)
    side_frames_dir.mkdir(parents=True, exist_ok=True)

    if args.smpl_model is not None:
        smpl_model_path = Path(args.smpl_model).expanduser().resolve()
        if not smpl_model_path.is_file():
            raise FileNotFoundError(f"Neutral SMPL model not found: {smpl_model_path}")
        smpl_body._SMPL_MODEL_PATHS["neutral"] = str(smpl_model_path)

    frame_dirs = common.discover_frames(
        dataset_root, args.dataset_split, args.start_frame + args.max_frames
    )[args.start_frame:]
    if not frame_dirs:
        raise ValueError(
            f"--start-frame {args.start_frame} is past the end of the dataset"
        )
    first_images = common.list_frame_images(frame_dirs[0])
    input_indices = (
        list(args.input_indices)
        if args.input_indices is not None
        else list(range(args.num_input_views))
    )
    if not input_indices or len(set(input_indices)) != len(input_indices):
        raise ValueError("Input camera indices must be non-empty and unique")
    if min(input_indices) < 0 or max(input_indices) >= len(first_images):
        raise ValueError(
            f"Input camera indices must be within [0, {len(first_images) - 1}]"
        )
    camera0_name = first_images[input_indices[0]].stem

    model, cfg, incompatible = common.load_model(
        args.config,
        checkpoint_path,
        device,
        keep_person_mask=args.translate_refine_mask,
    )
    if not bool(getattr(model, "use_temporal_smpl_head", False)):
        raise RuntimeError("Loaded config does not enable the temporal SMPL head")
    config_clip_length = int(
        OmegaConf.select(cfg, "temporal_training.clip_length", default=-1)
    )
    config_clip_stride = int(
        OmegaConf.select(cfg, "temporal_training.clip_stride", default=-1)
    )
    scale_by_extrinsics = bool(
        OmegaConf.select(cfg, "scale_by_extrinsics", default=True)
    )
    if config_clip_length != args.clip_length:
        raise RuntimeError(
            f"Config clip length is {config_clip_length}, requested {args.clip_length}"
        )
    if scale_by_extrinsics:
        raise RuntimeError("This GT gauge helper expects scale_by_extrinsics=False")

    print(f"[GT-HUNGARIAN] checkpoint={checkpoint_path}")
    print(f"[GT-HUNGARIAN] dataset={dataset_root / args.dataset_split}")
    print(
        f"[GT-HUNGARIAN] frames={len(frame_dirs)} causal_window={args.clip_length} "
        f"inference_stride=1 views={input_indices} fps={args.fps:g}"
    )
    print(
        f"[GT-HUNGARIAN] missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)} "
        "selection=all predicted slots -> GT mesh Hungarian"
    )

    all_vertices: list[np.ndarray] = []
    all_joints: list[np.ndarray] = []
    all_pred_gt_translate_vertices: list[np.ndarray] = []
    all_pred_gt_translate_joints: list[np.ndarray] = []
    all_gt_pose_pred_translate_vertices: list[np.ndarray] = []
    all_gt_pose_pred_translate_joints: list[np.ndarray] = []
    all_gt_vertices: list[np.ndarray] = []
    all_gt_joints: list[np.ndarray] = []
    all_color_ids: list[np.ndarray] = []
    pred_pose_tracks: list[np.ndarray] = []
    pred_beta_tracks: list[np.ndarray] = []
    pred_translate_tracks: list[np.ndarray] = []
    pred_translate_raw_tracks: list[np.ndarray] = []
    gt_pose_tracks: list[np.ndarray] = []
    gt_beta_tracks: list[np.ndarray] = []
    gt_translate_tracks: list[np.ndarray] = []
    records: list[dict] = []
    color_id_by_person: dict[str, int] = {}
    started = time.time()
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if autocast_enabled and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    faces = np.asarray(_get_smpl_model(device, "neutral").faces, dtype=np.int64)

    token_cache = AggregatorTokenCache(args.clip_length) if args.feature_cache else None
    inference_seconds = 0.0

    for frame_index, frame_dir in enumerate(frame_dirs):
        window_indices = causal_window_indices(frame_index, args.clip_length)
        if token_cache is not None:
            # Only the current frame is encoded; the window's earlier positions
            # come from the cache, so their JPEGs are never decoded again.
            image_paths = [
                str(common.list_frame_images(frame_dir)[index])
                for index in input_indices
            ]
        else:
            image_paths = []
            for window_index in window_indices:
                frame_images = common.list_frame_images(frame_dirs[window_index])
                image_paths.extend(str(frame_images[index]) for index in input_indices)
        images = load_and_preprocess_images(image_paths).to(device)
        smpl_inputs = {
            "temporal_num_frames": torch.tensor(
                [args.clip_length], device=device, dtype=torch.long
            ),
            "views_per_frame": torch.tensor(
                [len(input_indices)], device=device, dtype=torch.long
            ),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started = time.time()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            if token_cache is not None:
                predictions = cached_temporal_forward(
                    model,
                    token_cache,
                    images,
                    smpl_inputs,
                    want_person_mask=args.translate_refine_mask,
                )
            else:
                predictions = model(images, smpl_inputs=smpl_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.time() - inference_started
        frame_predictions = select_prediction_frame(predictions, args.clip_length - 1)

        num_slots = int(frame_predictions["smpl_pose"].shape[1])
        all_slots = np.arange(num_slots, dtype=np.int64)
        predicted_vertices, predicted_joints = render3d.decode_people(
            frame_predictions, all_slots, device, avg_scale=1.0
        )
        archive_path = (
            dataset_root / args.dataset_split / "out_data" / f"{frame_dir.name}.npz"
        )
        (
            person_ids,
            gt_vertices,
            gt_joints,
            gt_pose_rot,
            gt_beta,
            gt_translate,
        ) = load_gt_mesh_in_prediction_gauge(
            archive_path, camera0_name, device
        )
        slot_by_gt, cost = gt_mesh_hungarian(predicted_vertices, gt_vertices)
        matched_vertices = predicted_vertices[slot_by_gt]
        matched_joints = predicted_joints[slot_by_gt]
        matched_pred_pose = (
            frame_predictions["smpl_pose"][0, slot_by_gt]
            .float().detach().cpu().numpy().astype(np.float32)
        )
        matched_pred_beta = (
            frame_predictions["smpl_beta"][0, slot_by_gt]
            .float().detach().cpu().numpy().astype(np.float32)
        )
        matched_pred_translate = (
            frame_predictions["mesh_translate"][0, slot_by_gt]
            .float().detach().cpu().numpy().astype(np.float32)
        )
        matched_pred_translate_raw = matched_pred_translate.copy()
        refine_initial_ious = None
        refine_final_ious = None
        refine_elapsed_s = 0.0
        if args.translate_refine_mask:
            mask_logits = frame_predictions.get("person_mask_logits")
            if mask_logits is None:
                raise RuntimeError(
                    "--translate-refine-mask was requested, but the model did not "
                    "produce person_mask_logits"
                )
            predicted_masks = torch.sigmoid(mask_logits[0].float())
            if predicted_masks.ndim != 4:
                raise RuntimeError(
                    "Expected person_mask_logits[0] to have shape (views, slots, H, W), "
                    f"got {tuple(predicted_masks.shape)}"
                )
            current_images = common.list_frame_images(frame_dir)
            selected_image_paths = [current_images[index] for index in input_indices]
            selected_camera_names = [path.stem for path in selected_image_paths]
            relative_extrinsics, refine_intrinsics = load_refine_cameras(
                archive_path,
                selected_camera_names,
                selected_image_paths,
                int(predicted_masks.shape[-2]),
                int(predicted_masks.shape[-1]),
            )
            refine_initial_ious = []
            refine_final_ious = []
            refined_translates = []
            refine_started = time.perf_counter()
            for gt_index, slot in enumerate(slot_by_gt):
                refined_translate, initial_iou, final_iou = refine_mesh_translate_with_masks(
                    matched_vertices[gt_index],
                    matched_pred_translate[gt_index],
                    predicted_masks[:, int(slot)],
                    relative_extrinsics,
                    refine_intrinsics,
                    faces,
                    device,
                    args.translate_refine_iters,
                    args.translate_refine_size,
                    args.translate_refine_chamfer_points,
                    args.translate_refine_lr,
                )
                refined_translates.append(refined_translate)
                refine_initial_ious.append(initial_iou)
                refine_final_ious.append(final_iou)
            refine_elapsed_s = time.perf_counter() - refine_started
            matched_pred_translate = np.stack(refined_translates)
            matched_vertices, matched_joints = decode_params_at_mesh_translate(
                matched_pred_pose, matched_pred_beta, matched_pred_translate, device
            )
        pred_gt_translate_vertices, pred_gt_translate_joints = (
            decode_params_at_mesh_translate(
                matched_pred_pose, matched_pred_beta, gt_translate, device
            )
        )
        gt_pose_pred_translate_vertices, gt_pose_pred_translate_joints = (
            decode_params_at_mesh_translate(
                gt_pose_rot, gt_beta, matched_pred_translate, device
            )
        )

        for person_id in person_ids:
            if person_id not in color_id_by_person:
                color_id_by_person[person_id] = len(color_id_by_person)
        color_ids = np.asarray(
            [color_id_by_person[person_id] for person_id in person_ids], dtype=np.int64
        )
        logits = frame_predictions.get("smpl_presence_logits")
        probabilities = (
            common.stable_sigmoid(logits[0].float().cpu().numpy())
            if logits is not None
            else np.ones(num_slots, dtype=np.float64)
        )

        all_vertices.append(matched_vertices)
        all_joints.append(matched_joints)
        all_pred_gt_translate_vertices.append(pred_gt_translate_vertices)
        all_pred_gt_translate_joints.append(pred_gt_translate_joints)
        all_gt_pose_pred_translate_vertices.append(gt_pose_pred_translate_vertices)
        all_gt_pose_pred_translate_joints.append(gt_pose_pred_translate_joints)
        all_gt_vertices.append(gt_vertices)
        all_gt_joints.append(gt_joints)
        all_color_ids.append(color_ids)
        pred_pose_tracks.append(matched_pred_pose)
        pred_beta_tracks.append(matched_pred_beta)
        pred_translate_tracks.append(matched_pred_translate)
        pred_translate_raw_tracks.append(matched_pred_translate_raw)
        gt_pose_tracks.append(gt_pose_rot)
        gt_beta_tracks.append(gt_beta)
        gt_translate_tracks.append(gt_translate)
        records.append(
            {
                "frame_index": frame_index,
                "run": frame_dir.name,
                "causal_window": [frame_dirs[index].name for index in window_indices],
                "gt_archive": str(archive_path),
                "gt_person_ids": person_ids,
                "matched_predicted_slots": slot_by_gt.tolist(),
                "matched_presence_probabilities": [
                    float(probabilities[slot]) for slot in slot_by_gt
                ],
                "matched_mean_vertex_distances": [
                    float(cost[slot_by_gt[gt_index], gt_index])
                    for gt_index in range(len(person_ids))
                ],
                "hungarian_cost_matrix": cost.tolist(),
                "color_ids": color_ids.tolist(),
                "mask_translate_refine": (
                    {
                        "initial_mean_iou": float(np.mean(refine_initial_ious)),
                        "final_mean_iou": float(np.mean(refine_final_ious)),
                        "initial_ious": refine_initial_ious,
                        "final_ious": refine_final_ious,
                        "raw_mesh_translate": matched_pred_translate_raw.tolist(),
                        "refined_mesh_translate": matched_pred_translate.tolist(),
                        "elapsed_seconds": refine_elapsed_s,
                    }
                    if refine_initial_ious is not None
                    else None
                ),
            }
        )
        del predictions, frame_predictions, images
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (frame_index + 1) % max(1, args.log_every) == 0 or frame_index + 1 == len(frame_dirs):
            elapsed = time.time() - started
            rate = (frame_index + 1) / max(elapsed, 1e-6)
            eta = (len(frame_dirs) - frame_index - 1) / max(rate, 1e-6)
            print(
                f"[GT-HUNGARIAN] inference {frame_index + 1}/{len(frame_dirs)} "
                f"ETA={eta / 60.0:.1f} min"
                + (
                    f" refine_IoU={np.mean(refine_initial_ious):.3f}"
                    f"->{np.mean(refine_final_ious):.3f}"
                    if refine_initial_ious is not None
                    else ""
                )
            )

    pred_pose_np = np.stack(pred_pose_tracks)
    pred_beta_np = np.stack(pred_beta_tracks)
    pred_translate_np = np.stack(pred_translate_tracks)
    pred_translate_raw_np = np.stack(pred_translate_raw_tracks)
    gt_pose_np = np.stack(gt_pose_tracks)
    gt_beta_np = np.stack(gt_beta_tracks)
    gt_translate_np = np.stack(gt_translate_tracks)
    tracks_path = output_dir / "gt_translate_tracks.npz"
    np.savez_compressed(
        tracks_path,
        pred_pose=pred_pose_np,
        pred_beta=pred_beta_np,
        pred_translate=pred_translate_np,
        pred_translate_raw=pred_translate_raw_np,
        gt_pose_rot=gt_pose_np,
        gt_beta=gt_beta_np,
        gt_translate=gt_translate_np,
    )

    camera_vertices = [
        np.concatenate(parts, axis=0)
        for parts in zip(
            all_vertices,
            all_pred_gt_translate_vertices,
            all_gt_pose_pred_translate_vertices,
            all_gt_vertices,
        )
    ]
    camera_joints = [
        np.concatenate(parts, axis=0)
        for parts in zip(
            all_joints,
            all_pred_gt_translate_joints,
            all_gt_pose_pred_translate_joints,
            all_gt_joints,
        )
    ]
    camera = render3d.compute_virtual_camera(
        camera_vertices,
        camera_joints,
        args.azimuth_deg,
        args.elevation_deg,
        args.fov_deg,
    )
    comparison_paths = {
        "pred_3d.mp4": output_dir / "pred_3d.mp4",
        "pred_gt_translate_3d.mp4": output_dir / "pred_gt_translate_3d.mp4",
        "gt_pose_pred_translate_3d.mp4": output_dir / "gt_pose_pred_translate_3d.mp4",
        "gt_3d.mp4": output_dir / "gt_3d.mp4",
        "gt_translate_compare_3d.mp4": output_dir / "gt_translate_compare_3d.mp4",
        "gt_pred_identity_compare_3d.mp4": output_dir / "gt_pred_identity_compare_3d.mp4",
    }
    writers = {
        "pred": open_raw_mp4_writer(
            comparison_paths["pred_3d.mp4"], args.fps, args.width, args.height
        ),
        "pred_gt": open_raw_mp4_writer(
            comparison_paths["pred_gt_translate_3d.mp4"], args.fps, args.width, args.height
        ),
        "gt_pred": open_raw_mp4_writer(
            comparison_paths["gt_pose_pred_translate_3d.mp4"], args.fps, args.width, args.height
        ),
        "gt": open_raw_mp4_writer(
            comparison_paths["gt_3d.mp4"], args.fps, args.width, args.height
        ),
        "grid": open_raw_mp4_writer(
            comparison_paths["gt_translate_compare_3d.mp4"],
            args.fps,
            args.width * 2 + 4,
            args.height * 2 + 4,
        ),
        "identity": open_raw_mp4_writer(
            comparison_paths["gt_pred_identity_compare_3d.mp4"],
            args.fps,
            args.width * 2,
            args.height,
        ),
    }
    try:
        for frame_index, color_ids in enumerate(all_color_ids):
            pred_raw = render3d.render_mesh_frame(
                all_vertices[frame_index], faces, color_ids, camera, args.width, args.height
            )
            pred_gt_raw = render3d.render_mesh_frame(
                all_pred_gt_translate_vertices[frame_index],
                faces, color_ids, camera, args.width, args.height,
            )
            gt_pred_raw = render3d.render_mesh_frame(
                all_gt_pose_pred_translate_vertices[frame_index],
                faces, color_ids, camera, args.width, args.height,
            )
            gt_raw = render3d.render_mesh_frame(
                all_gt_vertices[frame_index], faces, color_ids, camera, args.width, args.height
            )
            pred = label_frame(
                pred_raw,
                "PRED" + (" + MASK TRANSLATE REFINE" if args.translate_refine_mask else ""),
                (
                    "predicted pose + mask-refined mesh_translate"
                    if args.translate_refine_mask
                    else "predicted pose + predicted mesh_translate"
                ),
                (90, 115, 255),
            )
            pred_gt = label_frame(
                pred_gt_raw, "PRED POSE + GT TRANSLATE",
                "predicted pose, GT mesh_translate", (80, 235, 255),
            )
            gt_pred = label_frame(
                gt_pred_raw, "GT POSE + PRED TRANSLATE",
                "GT pose, predicted mesh_translate", (95, 195, 255),
            )
            gt = label_frame(
                gt_raw, "GT", "GT pose + GT mesh_translate (reference)", (105, 235, 125)
            )
            vertical_gap = np.full((args.height, 4, 3), 34, dtype=np.uint8)
            horizontal_gap = np.full((4, args.width * 2 + 4, 3), 34, dtype=np.uint8)
            top = np.concatenate([pred, vertical_gap, pred_gt], axis=1)
            bottom = np.concatenate([gt_pred, vertical_gap, gt], axis=1)
            grid = np.concatenate([top, horizontal_gap, bottom], axis=0)
            identity = np.concatenate([gt, pred], axis=1)

            frames_to_write = {
                "pred": pred,
                "pred_gt": pred_gt,
                "gt_pred": gt_pred,
                "gt": gt,
                "grid": grid,
                "identity": identity,
            }
            for key, frame in frames_to_write.items():
                assert writers[key].stdin is not None
                writers[key].stdin.write(np.ascontiguousarray(frame).tobytes())

            mosaic = input_mosaic(
                frame_dirs[frame_index], input_indices, args.width, args.height
            )
            side_by_side = np.concatenate([mosaic, pred_raw], axis=1)
            if not cv2.imwrite(str(pure_frames_dir / f"{frame_index:04d}.png"), pred_raw):
                raise OSError(f"Failed to write pure frame {frame_index}")
            if not cv2.imwrite(str(side_frames_dir / f"{frame_index:04d}.png"), side_by_side):
                raise OSError(f"Failed to write side-by-side frame {frame_index}")
            if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(frame_dirs):
                print(f"[GT-HUNGARIAN] comparison render {frame_index + 1}/{len(frame_dirs)}")
    finally:
        for key, writer in writers.items():
            close_raw_mp4_writer(writer, comparison_paths[
                {
                    "pred": "pred_3d.mp4",
                    "pred_gt": "pred_gt_translate_3d.mp4",
                    "gt_pred": "gt_pose_pred_translate_3d.mp4",
                    "gt": "gt_3d.mp4",
                    "grid": "gt_translate_compare_3d.mp4",
                    "identity": "gt_pred_identity_compare_3d.mp4",
                }[key]
            ])

    fps_tag = f"{args.fps:g}fps"
    pure_path = output_dir / f"pred_gt_mesh_hungarian_3d_{fps_tag}.mp4"
    side_path = output_dir / f"input_and_pred_gt_mesh_hungarian_3d_{fps_tag}.mp4"
    encode_mp4(pure_frames_dir, pure_path, args.fps)
    encode_mp4(side_frames_dir, side_path, args.fps)

    manifest = {
        "checkpoint": str(checkpoint_path),
        "config": args.config,
        "checkpoint_missing_keys": incompatible.missing_keys,
        "checkpoint_unexpected_keys": incompatible.unexpected_keys,
        "dataset": str(dataset_root / args.dataset_split),
        "uses_gt_smpl_for_assignment": True,
        "uses_presence_topk": False,
        "predicted_slot_count": len(records[0]["hungarian_cost_matrix"]),
        "temporal_clip_length": args.clip_length,
        "temporal_alignment": "causal sliding window; current frame is final decoder position",
        "temporal_inference_stride": 1,
        "aggregator_feature_cache": {
            "enabled": bool(args.feature_cache),
            "cached_frames": args.clip_length - 1 if args.feature_cache else 0,
            "note": (
                "The temporal path folds T into the batch axis, so the aggregator "
                "sees one timestep at a time and cached tokens are bit-comparable "
                "to re-encoding. Each frame is encoded and JPEG-decoded once "
                "instead of clip_length times."
                if args.feature_cache
                else "Disabled: the whole causal window is re-encoded every frame."
            ),
        },
        "model_inference_seconds": inference_seconds,
        "mean_model_inference_seconds_per_frame": (
            inference_seconds / max(len(frame_dirs), 1)
        ),
        "training_clip_start_stride": config_clip_stride,
        "input_camera_indices": input_indices,
        "input_camera_names": [first_images[index].stem for index in input_indices],
        "identity_assignment": (
            "all predicted slots x GT people Hungarian assignment on mean "
            "corresponding-vertex distance in the camera0 SMPL mesh gauge"
        ),
        "mask_translate_refinement": {
            "enabled": args.translate_refine_mask,
            "target": "checkpoint person_mask_logits (sigmoid), all input views",
            "fixed_parameters": ["smpl_pose", "smpl_beta"],
            "optimized_parameter": "mesh_translate in camera0 gauge",
            "iterations": args.translate_refine_iters,
            "refine_size": args.translate_refine_size,
            "chamfer_points": args.translate_refine_chamfer_points,
            "learning_rate": args.translate_refine_lr,
            "mean_initial_iou": (
                float(np.mean([
                    frame["mask_translate_refine"]["initial_mean_iou"]
                    for frame in records
                ]))
                if args.translate_refine_mask
                else None
            ),
            "mean_final_iou": (
                float(np.mean([
                    frame["mask_translate_refine"]["final_mean_iou"]
                    for frame in records
                ]))
                if args.translate_refine_mask
                else None
            ),
        },
        "person_color_ids": color_id_by_person,
        "frame_count": len(frame_dirs),
        "fps": args.fps,
        "codec": "H.264, yuv420p, constant frame rate",
        "resolution": [args.width, args.height],
        "outputs": {
            "pred_gt_mesh_hungarian_3d": str(pure_path),
            "input_and_pred_gt_mesh_hungarian_3d": str(side_path),
            **{name: str(path) for name, path in comparison_paths.items()},
            "gt_translate_tracks.npz": str(tracks_path),
            "pred_frames": str(pure_frames_dir),
            "side_by_side_frames": str(side_frames_dir),
        },
        "elapsed_seconds": time.time() - started,
        "frames": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    gt_translate_manifest_path = output_dir / "gt_translate_manifest.json"
    gt_translate_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[RESULT] {pure_path}")
    print(f"[RESULT] {side_path}")
    for path in comparison_paths.values():
        print(f"[RESULT] {path}")
    print(f"[RESULT] {tracks_path}")
    print(f"[RESULT] {manifest_path}")
    print(f"[RESULT] {gt_translate_manifest_path}")


if __name__ == "__main__":
    main()
