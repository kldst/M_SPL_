#!/usr/bin/env python3
"""Cached causal T=3 / V=8 temporal inference with GT-free top-k slot selection.

`infer_temporal_smpl_mesh_hungarian_mp4.py` picks which predicted slots to show by matching
them against the GT meshes, and optionally refines mesh_translate against the predicted
masks.  This variant does neither:

  * slots are chosen by the model's own presence logits (`--top-k`, default 2),
  * `mesh_translate` is used exactly as predicted -- no mask refinement,

so the right-hand panel is what the checkpoint outputs on its own.  GT is still rendered on
the left for comparison, and is used for two things only: colouring the predicted people to
match the GT panel on the first frame, and the reported error.

    python infer_temporal_topk_mp4.py \
        --checkpoint model/root/checkpoint_30.pt \
        --dataset-root /train-data-3-hdd/yian/Multi_SMPL_0706/MAMMA_eval_dance \
        --output-dir debug/ck30_topk2/<sequence> --start-frame 0 --max-frames 225
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
from infer_temporal_smpl_mesh_hungarian_mp4 import (
    AggregatorTokenCache,
    cached_temporal_forward,
    causal_window_indices,
    close_raw_mp4_writer,
    label_frame,
    load_gt_mesh_in_prediction_gauge,
    open_raw_mp4_writer,
    select_prediction_frame,
)
from training.smpl_body import _get_smpl_model
from vggt.utils.load_fn import load_and_preprocess_images



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument("--checkpoint", default=str(REPO_DIR / "model/root/checkpoint_30.pt"))
    parser.add_argument(
        "--dataset-root", default="/train-data-3-hdd/yian/Multi_SMPL_0706/MAMMA_eval_dance"
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--smpl-model",
        default="/train-data-3-hdd/yian/Multi_SMPL_0706/smpl_models/"
        "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
    )

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=225)
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--num-input-views", type=int, default=8)
    parser.add_argument("--input-indices", type=int, nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--dedup",
        default="none",
        choices=("none", "nms3d", "temporal", "nms3d_temporal", "mask"),
        help=(
            "How to stop two slots latching onto the SAME person. 'none' takes the top-k "
            "presence slots as-is. 'nms3d' rejects a candidate whose pelvis is within "
            "--nms-dist of an already accepted one. 'temporal' first continues the tracks "
            "accepted in the previous frame. 'mask' rejects a candidate whose predicted "
            "person mask overlaps an accepted one by more than --mask-iou (needs the mask "
            "head, so the DPT trunk runs every frame)."
        ),
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=6,
        help="Slots considered before de-duplication (ignored when --dedup none).",
    )
    parser.add_argument("--nms-dist", type=float, default=0.25, help="metres")
    parser.add_argument("--mask-iou", type=float, default=0.5)
    parser.add_argument("--track-dist", type=float, default=0.5, help="metres")

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--azimuth-deg", type=float, default=35.0)
    parser.add_argument("--elevation-deg", type=float, default=28.0)
    parser.add_argument("--fov-deg", type=float, default=38.0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--pred-subtitle",
        default=None,
        help="Second line of the prediction panel. Defaults to '<ckpt dir>/<ckpt stem>, "
             "T=<clip>, V=<views>, top-<k>, no refine'.",
    )
    parser.add_argument(
        "--world-frame",
        action="store_true",
        help=(
            "Render in dataset world coordinates instead of the camera-0 gauge. "
            "Predictions and GT are mapped back with the inverse gauge transform "
            "X_world = (X_gauge * avg_scale - t0) @ R0, a rigid motion, so every "
            "reported metric is unchanged. Required to share a camera with "
            "HeatFormer, whose geometry is already world-frame."
        ),
    )
    parser.add_argument(
        "--camera-json",
        default=None,
        help=(
            "Use a fixed camera from this JSON (see inference/dump_heatformer_camera.py) "
            "instead of fitting one to this clip. A world-frame camera needs "
            "--world-frame."
        ),
    )
    parser.add_argument(
        "--feature-cache",
        dest="feature_cache",
        action="store_true",
        default=True,
        help=(
            "Encode each frame's views with the aggregator exactly once and "
            "replay the previous frames' final tokens from a rolling cache. The "
            "temporal path folds T into the batch axis, so this is the same "
            "window the full forward builds at 1/clip_length the cost."
        ),
    )
    parser.add_argument(
        "--no-feature-cache",
        dest="feature_cache",
        action="store_false",
        help="Re-encode the whole causal window every frame (original path).",
    )
    return parser.parse_args()


def mask_iou_matrix(mask_probs, slots, threshold=0.5):
    """Mean-over-views IoU between the binarised predicted masks of each slot pair."""
    binary = (mask_probs[:, slots] > threshold).reshape(mask_probs.shape[0], len(slots), -1)
    binary = binary.float()
    inter = torch.einsum("vah,vbh->vab", binary, binary)
    area = binary.sum(-1)
    union = area[:, :, None] + area[:, None, :] - inter
    iou = torch.where(union > 0, inter / union.clamp(min=1e-6), torch.zeros_like(inter))
    return iou.mean(0).cpu().numpy()


def gauge_to_world(points, extrinsics, avg_scale=1.0):
    """Invert normalize_joints_world_to_batch_gauge: X_world = (X_g*s - t0) @ R0.

    The gauge maps world into camera-0 coordinates as X_g = (X_w @ R0.T + t0)/s,
    so this is its exact inverse (R0 is a rotation, R0.T @ R0 = I). Being rigid,
    it leaves every joint/vertex distance -- and therefore every metric in this
    script -- untouched; only the rendering frame changes.
    """
    rotation = np.asarray(extrinsics, dtype=np.float64)[:, :3]
    translation = np.asarray(extrinsics, dtype=np.float64)[:, 3]
    shifted = np.asarray(points, dtype=np.float64) * float(avg_scale) - translation
    return (shifted @ rotation).astype(np.float32)


def read_camera0_extrinsics(archive_path, camera_name):
    """cam0 worldToCamera as (3,4); the gauge these predictions live in."""
    with np.load(archive_path, allow_pickle=False) as archive:
        return np.asarray(
            archive[f"cam_param_min/{camera_name}/extrinsics.worldToCamera12"],
            dtype=np.float64,
        ).reshape(3, 4)


def load_camera_json(path, world_frame):
    """Rehydrate a camera dict dumped by dump_heatformer_camera.py."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    camera = payload["camera"] if "camera" in payload else payload
    frame = payload.get("frame", "world")
    if frame == "world" and not world_frame:
        raise SystemExit(
            f"{path} holds a world-frame camera; pass --world-frame so the "
            "geometry is mapped into the same frame."
        )
    vector_keys = ("eye", "target", "up", "horizontal_x", "horizontal_z", "floor_point")
    rehydrated = {
        key: np.asarray(camera[key], dtype=np.float64) for key in vector_keys
    }
    rehydrated["grid_extent"] = float(camera["grid_extent"])
    rehydrated["fov_deg"] = float(camera["fov_deg"])
    return rehydrated, payload


def select_slots(probabilities, args, decode, previous_tracks, mask_probs=None):
    """Choose which person slots to keep, without ever looking at the GT.

    Returns the chosen slot indices.  Candidates are always visited in order of decreasing
    presence, so with --dedup none this is exactly the old top-k behaviour.
    """
    order = np.argsort(-probabilities, kind="stable")
    if args.dedup == "none":
        return order[: min(args.top_k, probabilities.size)]

    candidates = order[: min(args.candidates, probabilities.size)]
    _, cand_joints = decode(candidates)                     # (C, J, 3)
    roots = cand_joints[:, 0, :]
    iou = (
        mask_iou_matrix(mask_probs, candidates, args.presence_threshold)
        if args.dedup == "mask" and mask_probs is not None
        else None
    )

    def conflicts(i, accepted):
        for j in accepted:
            if iou is not None:
                if iou[i, j] > args.mask_iou:
                    return True
            elif float(np.linalg.norm(roots[i] - roots[j])) < args.nms_dist:
                return True
        return False

    accepted: list[int] = []
    if args.dedup in ("temporal", "nms3d_temporal") and previous_tracks is not None:
        # continue last frame's people first: for each track take the nearest candidate
        for track_root in previous_tracks:
            best, best_d = None, args.track_dist
            for i in range(len(candidates)):
                if i in accepted:
                    continue
                d = float(np.linalg.norm(roots[i] - track_root))
                if d < best_d and not conflicts(i, accepted):
                    best, best_d = i, d
            if best is not None:
                accepted.append(best)
            if len(accepted) == args.top_k:
                break

    for i in range(len(candidates)):                        # fill up by presence
        if len(accepted) == args.top_k:
            break
        if i not in accepted and not conflicts(i, accepted):
            accepted.append(i)

    for i in range(len(candidates)):                        # never return fewer than top-k
        if len(accepted) == args.top_k:
            break
        if i not in accepted:
            accepted.append(i)

    return candidates[np.array(accepted, dtype=np.int64)]


def colour_ids(pred_joints, gt_joints, previous, colour_of_person, person_ids):
    """Stable colours: GT by person id, prediction by frame-to-frame pelvis tracking.

    The very first frame seeds the predicted tracks from the nearest GT root so that the two
    panels agree; after that the prediction is tracked on its own.
    """
    for pid in person_ids:
        colour_of_person.setdefault(pid, len(colour_of_person))
    gt_ids = np.array([colour_of_person[pid] for pid in person_ids], dtype=np.int64)

    pred_roots = pred_joints[:, 0, :]
    if previous is None:
        cost = np.linalg.norm(pred_roots[:, None] - gt_joints[None, :, 0, :], axis=-1)
        ids = np.full(len(pred_roots), -1, dtype=np.int64)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            ids[r] = gt_ids[c]
        spare = (i for i in range(100) if i not in set(ids.tolist()))
        ids = np.array([i if i >= 0 else next(spare) for i in ids], dtype=np.int64)
    else:
        cost = np.linalg.norm(pred_roots[:, None] - previous["roots"][None], axis=-1)
        ids = np.full(len(pred_roots), -1, dtype=np.int64)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] < 1.0:
                ids[r] = previous["ids"][c]
        spare = (i for i in range(100) if i not in set(ids.tolist()))
        ids = np.array([i if i >= 0 else next(spare) for i in ids], dtype=np.int64)
    return ids, gt_ids, {"roots": pred_roots, "ids": ids}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    smpl_path = Path(args.smpl_model).expanduser().resolve()
    if not smpl_path.is_file():
        raise FileNotFoundError(f"Neutral SMPL model not found: {smpl_path}")
    import training.smpl_body as smpl_body

    smpl_body._SMPL_MODEL_PATHS["neutral"] = str(smpl_path)

    frame_dirs = common.discover_frames(
        dataset_root, args.dataset_split, args.start_frame + args.max_frames
    )[args.start_frame :]
    if not frame_dirs:
        raise ValueError(f"--start-frame {args.start_frame} is past the end of the dataset")

    first_images = common.list_frame_images(frame_dirs[0])
    input_indices = (
        list(args.input_indices)
        if args.input_indices is not None
        else list(range(args.num_input_views))
    )
    camera0_name = first_images[input_indices[0]].stem

    model, cfg, incompatible = common.load_model(
        args.config, Path(args.checkpoint).expanduser().resolve(), device,
        keep_person_mask=args.dedup == "mask",
        # This script reads only SMPL pose/beta/translate/presence (+ masks for
        # --dedup mask); it never uses `pose_enc`. Freeing the camera head's
        # 825MiB keeps the run inside a partly occupied GPU without changing a
        # single predicted value.
        keep_camera=False,
    )
    if not bool(getattr(model, "use_temporal_smpl_head", False)):
        raise RuntimeError("Loaded config does not enable the temporal SMPL head")
    config_clip_length = int(OmegaConf.select(cfg, "temporal_training.clip_length", default=-1))
    if config_clip_length != args.clip_length:
        raise RuntimeError(
            f"Config clip length is {config_clip_length}, requested {args.clip_length}"
        )

    print(f"[TOPK] checkpoint={args.checkpoint}")
    print(
        f"[TOPK] frames={len(frame_dirs)} (start={args.start_frame}) "
        f"causal_window={args.clip_length} views={input_indices} top_k={args.top_k} "
        "refine=off selection=presence logits (no GT)"
    )
    print(
        f"[TOPK] missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )

    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if autocast_enabled and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    faces = np.asarray(_get_smpl_model(device, "neutral").faces, dtype=np.int64)

    all_pred_v, all_pred_j, all_gt_v, all_gt_j = [], [], [], []
    all_pred_ids, all_gt_ids, presence_kept = [], [], []
    colour_of_person: dict[str, int] = {}
    previous = None
    previous_tracks = None
    token_cache = AggregatorTokenCache(args.clip_length) if args.feature_cache else None
    inference_seconds = 0.0
    started = time.time()

    for frame_index, frame_dir in enumerate(frame_dirs):
        window_indices = causal_window_indices(frame_index, args.clip_length)
        if token_cache is not None:
            # Cached mode encodes (and JPEG-decodes) only the current frame; the
            # window's earlier positions come from the cache.
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
            "temporal_num_frames": torch.tensor([args.clip_length], device=device, dtype=torch.long),
            "views_per_frame": torch.tensor([len(input_indices)], device=device, dtype=torch.long),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started = time.time()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled
        ):
            if token_cache is not None:
                predictions = cached_temporal_forward(
                    model,
                    token_cache,
                    images,
                    smpl_inputs,
                    want_person_mask=args.dedup == "mask",
                )
            else:
                predictions = model(images, smpl_inputs=smpl_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.time() - inference_started
        frame_predictions = select_prediction_frame(predictions, args.clip_length - 1)

        logits = frame_predictions.get("smpl_presence_logits")
        probabilities = (
            common.stable_sigmoid(logits[0].float().cpu().numpy())
            if logits is not None
            else np.ones(frame_predictions["smpl_pose"].shape[1], dtype=np.float64)
        )
        def decode(sel):
            return render3d.decode_people(frame_predictions, sel, device, avg_scale=1.0)

        mask_probs = None
        if args.dedup == "mask":
            mask_logits = frame_predictions.get("person_mask_logits")
            if mask_logits is None:
                raise RuntimeError("--dedup mask needs person_mask_logits from the model")
            mask_probs = torch.sigmoid(mask_logits[0].float())      # (views, slots, H, W)

        slots = select_slots(probabilities, args, decode, previous_tracks, mask_probs)
        presence_kept.append(probabilities[slots].tolist())

        pred_v, pred_j = decode(slots)
        previous_tracks = pred_j[:, 0, :].copy()
        archive = dataset_root / args.dataset_split / "out_data" / f"{frame_dir.name}.npz"
        person_ids, gt_v, gt_j, _, _, _ = load_gt_mesh_in_prediction_gauge(
            archive, camera0_name, device
        )

        ids, gt_ids, previous = colour_ids(pred_j, gt_j, previous, colour_of_person, person_ids)
        gt_v = np.asarray(gt_v, dtype=np.float32)
        gt_j = np.asarray(gt_j, dtype=np.float32)
        if args.world_frame:
            # Map out of the cam0 gauge only for storage/rendering: slot
            # selection, tracking and colour ids above all run on distances,
            # which this rigid transform preserves.
            extrinsics = read_camera0_extrinsics(archive, camera0_name)
            pred_v = gauge_to_world(pred_v, extrinsics)
            pred_j = gauge_to_world(pred_j, extrinsics)
            gt_v = gauge_to_world(gt_v, extrinsics)
            gt_j = gauge_to_world(gt_j, extrinsics)
        all_pred_v.append(pred_v)
        all_pred_j.append(pred_j)
        all_gt_v.append(gt_v)
        all_gt_j.append(gt_j)
        all_pred_ids.append(ids)
        all_gt_ids.append(gt_ids)

        if (frame_index + 1) % max(1, args.log_every) == 0 or frame_index + 1 == len(frame_dirs):
            elapsed = time.time() - started
            rate = (frame_index + 1) / max(elapsed, 1e-6)
            print(
                f"[TOPK] inference {frame_index + 1}/{len(frame_dirs)} "
                f"({rate:.2f} fps, ETA {(len(frame_dirs) - frame_index - 1) / max(rate, 1e-6) / 60:.1f} min)",
                flush=True,
            )

    camera_payload = None
    if args.camera_json:
        camera, camera_payload = load_camera_json(args.camera_json, args.world_frame)
        print(
            f"[TOPK] fixed camera from {args.camera_json} "
            f"(frame={camera_payload.get('frame', 'world')}, "
            f"source={camera_payload.get('source', 'n/a')})",
            flush=True,
        )
    else:
        camera = render3d.compute_virtual_camera(
            all_pred_v + all_gt_v, all_pred_j + all_gt_j,
            args.azimuth_deg, args.elevation_deg, args.fov_deg,
        )

    ckpt = Path(args.checkpoint)
    subtitle = args.pred_subtitle or (
        f"{ckpt.parent.name}/{ckpt.stem}, T={args.clip_length}, V={len(input_indices)}, "
        f"top-{args.top_k}, no refine"
    )

    compare_path = output_dir / "gt_pred_topk_compare_3d.mp4"
    pred_path = output_dir / "pred_topk_3d.mp4"
    writers = {
        "compare": open_raw_mp4_writer(compare_path, args.fps, args.width * 2, args.height),
        "pred": open_raw_mp4_writer(pred_path, args.fps, args.width, args.height),
    }
    errors, root_errors = [], []
    matched_pred_j, matched_gt_j = [], []
    try:
        for index in range(len(all_pred_v)):
            pred_panel = label_frame(
                render3d.render_mesh_frame(
                    all_pred_v[index], faces, all_pred_ids[index], camera, args.width, args.height
                ),
                "PRED", subtitle, (80, 200, 255),
            )
            gt_panel = label_frame(
                render3d.render_mesh_frame(
                    all_gt_v[index], faces, all_gt_ids[index], camera, args.width, args.height
                ),
                "GT", "GT pose + GT mesh_translate", (110, 235, 130),
            )
            writers["compare"].stdin.write(
                np.ascontiguousarray(np.hstack([gt_panel, pred_panel])).tobytes()
            )
            writers["pred"].stdin.write(np.ascontiguousarray(pred_panel).tobytes())

            # pred <-> GT matched on the joints themselves (report only)
            pred_j = all_pred_j[index][:, :24]
            gt_j = all_gt_j[index][:, :24]
            cost = np.linalg.norm(pred_j[:, None] - gt_j[None], axis=-1).mean(-1)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                errors.append(float(cost[r, c]) * 1000)
                # root-aligned: put each body's own pelvis at the origin first
                pr = pred_j[r] - pred_j[r][:1]
                gj = gt_j[c] - gt_j[c][:1]
                root_errors.append(float(np.linalg.norm(pr - gj, axis=-1).mean()) * 1000)
                matched_pred_j.append(pred_j[r])
                matched_gt_j.append(gt_j[c])
            if (index + 1) % 100 == 0 or index + 1 == len(all_pred_v):
                print(f"[TOPK] render {index + 1}/{len(all_pred_v)}", flush=True)
    finally:
        close_raw_mp4_writer(writers["compare"], compare_path)
        close_raw_mp4_writer(writers["pred"], pred_path)

    # how often did two kept slots land on the same person?
    duplicate_frames = 0
    off = 0
    for joints in all_pred_j:
        if len(joints) > 1:
            d = np.linalg.norm(joints[:, None, 0] - joints[None, :, 0], axis=-1)
            np.fill_diagonal(d, np.inf)
            duplicate_frames += int(d.min() < args.nms_dist)

    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": args.config,
        "start_frame": args.start_frame,
        "frame_count": len(frame_dirs),
        "clip_length": args.clip_length,
        "input_indices": input_indices,
        "top_k": args.top_k,
        "dedup": args.dedup,
        "candidates": args.candidates if args.dedup != "none" else args.top_k,
        "duplicate_frames": duplicate_frames,
        "frames": len(all_pred_j),
        "seconds_per_frame": round((time.time() - started) / max(len(frame_dirs), 1), 3),
        "render_frame": "world" if args.world_frame else "camera0_gauge",
        "camera_source": (
            {"fixed_json": str(Path(args.camera_json).resolve()),
             "sequence": (camera_payload or {}).get("sequence"),
             "origin": (camera_payload or {}).get("source")}
            if args.camera_json
            else {"fitted_to_this_clip": True,
                  "azimuth_deg": args.azimuth_deg,
                  "elevation_deg": args.elevation_deg,
                  "fov_deg": args.fov_deg}
        ),
        "aggregator_feature_cache": {
            "enabled": bool(args.feature_cache),
            "cached_frames": args.clip_length - 1 if args.feature_cache else 0,
        },
        "model_inference_seconds": round(inference_seconds, 2),
        "model_seconds_per_frame": round(
            inference_seconds / max(len(frame_dirs), 1), 4
        ),
        "pred_subtitle": subtitle,
        "translate_refine": False,
        "selection": "presence_topk",
        "absolute_mpjpe_mm": round(float(np.mean(errors)), 2) if errors else None,
        "root_aligned_mpjpe_mm": round(float(np.mean(root_errors)), 2) if root_errors else None,
        "matched_people": len(errors),
        "mean_presence_of_kept_slots": round(float(np.mean(presence_kept)), 4),
        "elapsed_seconds": round(time.time() - started, 1),
        "videos": [str(compare_path), str(pred_path)],
    }
    # every matched (frame, person) pair, so any other metric can be recomputed later
    # without re-running inference
    np.savez_compressed(
        output_dir / "matched_joints.npz",
        pred_joints=np.stack(matched_pred_j).astype(np.float32),
        gt_joints=np.stack(matched_gt_j).astype(np.float32),
        people_per_frame=np.array([len(v) for v in all_pred_j], dtype=np.int32),
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[RESULT] {compare_path}")
    print(f"[RESULT] {pred_path}")
    print(
        f"[RESULT] absolute MPJPE {manifest['absolute_mpjpe_mm']} mm, "
        f"root-aligned MPJPE {manifest['root_aligned_mpjpe_mm']} mm, "
        f"duplicate frames {duplicate_frames}/{len(all_pred_j)}, "
        f"mean presence {manifest['mean_presence_of_kept_slots']}"
    )


if __name__ == "__main__":
    main()
