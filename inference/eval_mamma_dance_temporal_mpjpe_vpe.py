#!/usr/bin/env python3
"""Evaluate the cached causal T=3/V=8 temporal model on MAMMA dance.

The evaluator encodes each frame only once.  The final aggregator features of
the previous two frames are cached and passed, together with the current
frame, to the temporal SMPL head.  At a sequence boundary the cache is reset;
the first two causal windows are padded by repeating the first frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

import eval_mamma_dance_mpjpe as base
import training.smpl_body as smpl_body
from training.loss import _decode_smpl_batch
from vggt.utils.load_fn import load_and_preprocess_images




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cached causal T=3/V=8 temporal MPJPE and VPE evaluation."
    )
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument(
        "--inference-mode",
        choices=("cached_temporal", "framewise"),
        default="cached_temporal",
        help="Use cached causal T=3 features or independent 8-view frames.",
    )
    parser.add_argument(
        "--checkpoint", default=str(REPO_DIR / "model/root/checkpoint_30.pt")
    )
    parser.add_argument(
        "--dataset-root",
        default="/train-data-3-hdd/yian/Multi_SMPL_0706/MAMMA_eval_dance",
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument(
        "--smpl-model-dir",
        default="/train-data-3-hdd/yian/Multi_SMPL_0706/smpl_models",
        help="Directory containing the gendered 6890-vertex SMPL .pkl files.",
    )
    parser.add_argument(
        "--image-ids",
        default="0 1 2 3 4 5 6 7",
        help="Zero-based indices into the sorted camera images (must be 8 views).",
    )
    parser.add_argument(
        "--output",
        default=str(
            REPO_DIR
            / "eval/eval_results/root_checkpoint_30_mamma_eval_dance_temporal_summary.json"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=3,
        help=(
            "Frames of JPEG/GT decoding to keep read ahead on worker threads. "
            "Preprocessing costs about as much wall time as the cached forward "
            "and uses the CPU instead of the GPU, so reading ahead takes it off "
            "the critical path. 0 disables prefetching."
        ),
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=3,
        help="Worker threads decoding prefetched frames.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def parse_sequence_segments(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        raise FileNotFoundError(f"Dataset sequence log not found: {log_path}")
    pattern = re.compile(
        r"SUCCESS\s+(.+?):\s+(\d+) frames,.*?\(run_offset now (\d+)\)"
    )
    segments = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        name, count_text, end_text = match.groups()
        count, end = int(count_text), int(end_text)
        segments.append({"name": name, "start": end - count, "end": end - 1})
    if not segments:
        raise RuntimeError(f"No successful sequence segments found in {log_path}")
    for previous, current in zip(segments, segments[1:]):
        if previous["end"] + 1 != current["start"]:
            raise RuntimeError(f"Non-contiguous sequence log near {current['name']}")
    return segments


def decode_body(
    poses: np.ndarray,
    betas: np.ndarray,
    genders: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(poses.shape[0])
    if count == 0:
        return np.empty((0, 24, 3)), np.empty((0, 0, 3))
    pose_tensor = torch.as_tensor(poses, dtype=torch.float32, device=device)
    beta_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device)
    zero_trans = torch.zeros((count, 3), dtype=torch.float32, device=device)
    joints, vertices = _decode_smpl_batch(
        pose_aa=pose_tensor,
        betas=beta_tensor,
        trans=zero_trans,
        genders=genders,
        use_mamma=False,
    )
    if joints is None or vertices is None:
        raise RuntimeError("SMPL decoder returned no joints or vertices")
    return (
        joints[:, :24].detach().cpu().numpy().astype(np.float64),
        vertices.detach().cpu().numpy().astype(np.float64),
    )


def configure_smpl_models(model_dir: Path) -> None:
    paths = {
        "female": model_dir / "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
        "male": model_dir / "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
        "neutral": model_dir / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SMPL model files: {missing}")
    smpl_body._SMPL_MODEL_PATHS.update(
        {gender: str(path.resolve()) for gender, path in paths.items()}
    )
    smpl_body._SMPL_MODEL_CACHE.clear()


class FramePrefetcher:
    """Decode each frame's views and GT archive ahead of the GPU.

    Order is preserved: ``next()`` returns frames in the order they were
    given, and a frame that fails to decode raises at the position the
    sequential code would have failed, so the caller's per-frame error
    handling is unchanged.
    """

    def __init__(
        self,
        frames: list[Path],
        image_ids: list[int],
        data_root: Path,
        depth: int,
        workers: int,
    ) -> None:
        self._frames = list(frames)
        self._image_ids = list(image_ids)
        self._data_root = Path(data_root)
        self._depth = max(1, int(depth))
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="frame-prefetch"
        )
        self._pending: deque = deque()
        self._next_index = 0

    def _decode(self, frame_dir: Path) -> tuple[list[str], torch.Tensor, dict]:
        image_paths = base.select_frame_images(frame_dir, self._image_ids)
        gt = base.load_frame_gt(
            self._data_root / f"{frame_dir.name}.npz", Path(image_paths[0]).stem
        )
        return image_paths, load_and_preprocess_images(image_paths), gt

    def _fill(self) -> None:
        while len(self._pending) < self._depth and self._next_index < len(self._frames):
            self._pending.append(
                self._pool.submit(self._decode, self._frames[self._next_index])
            )
            self._next_index += 1

    def next(self) -> tuple[list[str], torch.Tensor, dict]:
        self._fill()
        if not self._pending:
            raise IndexError("Prefetch queue is exhausted")
        future = self._pending.popleft()
        self._fill()
        return future.result()

    def close(self) -> None:
        self._pending.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)


def cached_temporal_forward(
    model,
    cached_tokens: deque,
    images: torch.Tensor,
    metadata: dict[str, torch.Tensor],
) -> tuple[dict, torch.Tensor]:
    current_features, patch_start_idx, _ = model.aggregator(images)
    current_final = current_features[-1]
    history = list(cached_tokens)
    if not history:
        history = [current_final, current_final]
    elif len(history) == 1:
        history = [history[0], history[0]]
    window_features = [torch.cat([history[-2], history[-1], current_final], dim=0)]
    with torch.cuda.amp.autocast(enabled=False):
        outputs = model.smpl_multi_query_trans_rot_head(
            window_features,
            patch_start_idx=patch_start_idx,
            smpl_inputs=metadata,
        )
    predictions = {
        key: value[-1:] if torch.is_tensor(value) else value
        for key, value in outputs.items()
    }
    return predictions, current_final.detach()


def evaluate_prediction(
    predictions: dict,
    gt: dict,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    poses = predictions["smpl_pose"][0].float().cpu().numpy()
    betas = predictions["smpl_beta"][0].float().cpu().numpy()
    logits = predictions.get("smpl_presence_logits")
    probabilities = (
        base.stable_sigmoid(logits[0].float().cpu().numpy())
        if logits is not None
        else np.ones(poses.shape[0], dtype=np.float64)
    )
    gt_count = int(gt["pose"].shape[0])
    selected = np.argsort(-probabilities, kind="stable")[: min(gt_count, len(probabilities))]
    row = {
        "gt_people": gt_count,
        "selected_people": int(len(selected)),
        "matched_people": 0,
        "max_presence_probability": float(probabilities.max()),
        "mpjpe_mm": float("nan"),
        "vpe_mm": float("nan"),
        "matched_pred_slots": "",
        "matched_gt_people": "",
    }
    if not len(selected) or not gt_count:
        return row, np.empty(0), np.empty(0)

    pred_joints, pred_vertices = decode_body(
        poses[selected], betas[selected], ["neutral"] * len(selected), device
    )
    gt_joints, gt_vertices = decode_body(
        gt["pose"], gt["beta"], gt["genders"], device
    )

    # Remove pelvis translation from both joints and mesh vertices.  This gives
    # root-aligned MPJPE and pelvis-aligned VPE in corresponding SMPL topology.
    pred_vertices = pred_vertices - pred_joints[:, :1, :]
    gt_vertices = gt_vertices - gt_joints[:, :1, :]
    pred_joints = base.root_align(pred_joints)
    gt_joints = base.root_align(gt_joints)

    # This checkpoint's mesh_rot/root pose is predicted in camera-0 coordinates.
    rotation0 = gt["camera0_extrinsic"][:, :3]
    gt_joints = gt_joints @ rotation0.T
    gt_vertices = gt_vertices @ rotation0.T

    pairwise_cost = np.linalg.norm(
        pred_joints[:, None] - gt_joints[None, :], axis=-1
    ).mean(axis=-1)
    pred_indices, gt_indices = linear_sum_assignment(pairwise_cost)
    joint_errors = np.linalg.norm(
        pred_joints[pred_indices] - gt_joints[gt_indices], axis=-1
    ) * 1000.0
    vertex_errors = np.linalg.norm(
        pred_vertices[pred_indices] - gt_vertices[gt_indices], axis=-1
    ) * 1000.0
    row.update(
        matched_people=int(len(pred_indices)),
        mpjpe_mm=float(joint_errors.mean()),
        vpe_mm=float(vertex_errors.mean()),
        matched_pred_slots=" ".join(str(int(selected[i])) for i in pred_indices),
        matched_gt_people=" ".join(str(int(i)) for i in gt_indices),
    )
    return row, joint_errors.reshape(-1), vertex_errors.reshape(-1)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    image_ids = base.parse_image_ids(args.image_ids)
    if len(image_ids) != 8:
        raise ValueError("This checkpoint evaluation requires exactly 8 views")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    configure_smpl_models(Path(args.smpl_model_dir).expanduser().resolve())
    split_root = dataset_root / args.dataset_split
    data_root = split_root / "out_data"
    frames = base.discover_frames(dataset_root, args.dataset_split)
    discovered_frames = len(frames)
    segments = parse_sequence_segments(split_root / "logs/process.log")
    if segments[-1]["end"] + 1 != discovered_frames:
        raise RuntimeError(
            f"Sequence log covers {segments[-1]['end'] + 1} frames, "
            f"but discovered {discovered_frames}"
        )
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be >= 1")
        frames = frames[: args.max_frames]

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model, cfg, incompatible = base.load_model(args.config, checkpoint, device)
    if args.inference_mode == "cached_temporal" and not getattr(
        model, "use_temporal_smpl_head", False
    ):
        raise RuntimeError(f"Config {args.config} does not enable the temporal SMPL head")
    if model.smpl_multi_query_trans_rot_head is None:
        raise RuntimeError("Temporal SMPL trans-rot head is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = True

    output_json = Path(args.output).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_json.with_suffix(".csv")
    metadata = {
        "temporal_num_frames": torch.tensor([3], dtype=torch.long, device=device),
        "views_per_frame": torch.tensor([8], dtype=torch.long, device=device),
    }
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )

    print(f"[EVAL] checkpoint={checkpoint}", flush=True)
    print(f"[EVAL] dataset={split_root}", flush=True)
    print(
        f"[EVAL] frames={len(frames)}/{discovered_frames} "
        f"mode={args.inference_mode} views={image_ids} segments={len(segments)} "
        f"selection=topk_gt",
        flush=True,
    )
    print(
        f"[EVAL] checkpoint missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )

    fieldnames = [
        "frame", "sequence", "sequence_frame", "status", "views", "gt_people",
        "selected_people", "matched_people", "max_presence_probability",
        "mpjpe_mm", "vpe_mm", "matched_pred_slots", "matched_gt_people",
        "inference_seconds", "total_seconds", "error",
    ]
    totals = {"failed": 0, "gt": 0, "selected": 0, "matched": 0}
    joint_sum = vertex_sum = 0.0
    joint_count = vertex_count = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    token_cache: deque = deque(maxlen=2)
    segment_index = 0
    prefetcher = (
        FramePrefetcher(
            frames, image_ids, data_root, args.prefetch_depth, args.prefetch_workers
        )
        if args.prefetch_depth > 0
        else None
    )

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for position, frame_dir in enumerate(frames):
            while position > segments[segment_index]["end"]:
                segment_index += 1
            segment = segments[segment_index]
            if position == segment["start"]:
                token_cache.clear()
            frame_started = time.perf_counter()
            row = {
                "frame": frame_dir.name,
                "sequence": segment["name"],
                "sequence_frame": position - segment["start"],
                "status": "ok",
                "views": 8,
                "error": "",
            }
            try:
                if prefetcher is not None:
                    _, images_cpu, gt = prefetcher.next()
                else:
                    image_paths = base.select_frame_images(frame_dir, image_ids)
                    gt = base.load_frame_gt(
                        data_root / f"{frame_dir.name}.npz", Path(image_paths[0]).stem
                    )
                    images_cpu = load_and_preprocess_images(image_paths)
                images = images_cpu.unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_started = time.perf_counter()
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=device.type == "cuda",
                ):
                    if args.inference_mode == "cached_temporal":
                        predictions, current_token = cached_temporal_forward(
                            model, token_cache, images, metadata
                        )
                    else:
                        predictions = model(images)
                        current_token = None
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row["inference_seconds"] = time.perf_counter() - infer_started
                inference_seconds += row["inference_seconds"]
                if current_token is not None:
                    token_cache.append(current_token)
                measured, joint_errors, vertex_errors = evaluate_prediction(
                    predictions, gt, device
                )
                row.update(measured)
                joint_sum += float(joint_errors.sum())
                joint_count += int(joint_errors.size)
                vertex_sum += float(vertex_errors.sum())
                vertex_count += int(vertex_errors.size)
                totals["gt"] += int(row["gt_people"])
                totals["selected"] += int(row["selected_people"])
                totals["matched"] += int(row["matched_people"])
                del images, images_cpu, predictions
            except Exception as exc:
                if args.fail_fast:
                    raise
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                totals["failed"] += 1
                # A missing feature would invalidate subsequent temporal windows.
                token_cache.clear()
                print(f"[EVAL][WARN] {frame_dir.name}: {row['error']}", flush=True)
            row["total_seconds"] = time.perf_counter() - frame_started
            writer.writerow(row)
            csv_file.flush()

            completed = position + 1
            if args.log_every > 0 and (
                completed == 1 or completed % args.log_every == 0 or completed == len(frames)
            ):
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-9)
                eta_minutes = (len(frames) - completed) / max(rate, 1e-9) / 60.0
                print(
                    f"[EVAL] {completed}/{len(frames)} {frame_dir.name} "
                    f"MPJPE={row.get('mpjpe_mm', float('nan')):.3f} "
                    f"VPE={row.get('vpe_mm', float('nan')):.3f} "
                    f"rate={rate:.3f} frame/s ETA={eta_minutes:.1f} min",
                    flush=True,
                )

    if prefetcher is not None:
        prefetcher.close()
    elapsed = time.perf_counter() - started
    summary = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "dataset": str(split_root),
        "image_ids": image_ids,
        "view_names": [f"IOI_{index + 1:02d}" for index in image_ids],
        "inference_mode": args.inference_mode,
        "temporal_frames": 3 if args.inference_mode == "cached_temporal" else 1,
        "feature_cache": args.inference_mode == "cached_temporal",
        "frame_prefetch": {
            "depth": args.prefetch_depth,
            "workers": args.prefetch_workers,
            "note": (
                "JPEG decode and GT archive loading run on worker threads ahead "
                "of the GPU; this changes wall time only, not any prediction."
            ),
        },
        "causal_padding": (
            "repeat first frame within each sequence"
            if args.inference_mode == "cached_temporal"
            else None
        ),
        "selection_mode": "topk_gt",
        "discovered_frames": discovered_frames,
        "evaluated_frames": len(frames),
        "successful_frames": len(frames) - totals["failed"],
        "failed_frames": totals["failed"],
        "sequence_count": len(segments),
        "sequences": segments,
        "gt_people": totals["gt"],
        "selected_predicted_people": totals["selected"],
        "matched_people": totals["matched"],
        "mpjpe_root_aligned_mm": joint_sum / joint_count if joint_count else None,
        "vpe_pelvis_aligned_mm": vertex_sum / vertex_count if vertex_count else None,
        "joint_error_samples": joint_count,
        "vertex_error_samples": vertex_count,
        "joint_count_per_person": 24,
        "vertex_count_per_person": vertex_count // max(totals["matched"], 1),
        "model_inference_seconds": inference_seconds,
        "mean_model_inference_seconds_per_frame": inference_seconds / max(len(frames), 1),
        "elapsed_seconds": elapsed,
        "mean_end_to_end_seconds_per_frame": elapsed / max(len(frames), 1),
        "per_frame_csv": str(output_csv),
        "metric_note": (
            "Top-k slots (k=GT people) are Hungarian-matched by root-aligned first-24-"
            "joint error. MPJPE uses those 24 SMPL joints; VPE uses every SMPL vertex. "
            "Both subtract joint-0 pelvis translation. GT is rotated into camera-0 "
            "orientation for this mesh_rot checkpoint. Values are millimetres."
        ),
        "config_scale_by_extrinsics": bool(
            OmegaConf.select(cfg, "scale_by_extrinsics", default=True)
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[RESULT] MPJPE={summary['mpjpe_root_aligned_mm']:.6f} mm", flush=True)
    print(f"[RESULT] VPE={summary['vpe_pelvis_aligned_mm']:.6f} mm", flush=True)
    print(f"[RESULT] summary={output_json}", flush=True)
    print(f"[RESULT] per_frame={output_csv}", flush=True)


if __name__ == "__main__":
    main()
