#!/usr/bin/env python3
"""Benchmark online T=3/V=8 inference with a two-frame feature cache.

The end-to-end latency starts before loading the eight current-frame JPEGs and
ends after SMPL pose/beta/translation/presence outputs have reached CPU memory.
Model construction/checkpoint loading and metric/GT decoding are reported
separately and are not part of per-frame inference latency.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


def run_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.rsplit("_", 1)[-1]), path.name
    except ValueError:
        return sys.maxsize, path.name


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def cached_head_forward(model, final_tokens, patch_start_idx, metadata):
    history = list(final_tokens)
    current = history[-1]
    if len(history) == 1:
        temporal_tokens = [current, current, current]
    elif len(history) == 2:
        temporal_tokens = [history[0], history[0], history[1]]
    else:
        temporal_tokens = history[-3:]
    features = [torch.cat(temporal_tokens, dim=0)]
    with torch.cuda.amp.autocast(enabled=False):
        outputs = model.smpl_multi_query_trans_rot_head(
            features,
            patch_start_idx=patch_start_idx,
            smpl_inputs=metadata,
        )
    return {
        key: value[-1:] if torch.is_tensor(value) else value
        for key, value in outputs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO / "model/root/checkpoint_30.pt"
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(
            "/train-data-3-hdd/yian/Multi_SMPL_0706/"
            "MAMMA_eval_dance/test/out_image"
        ),
    )
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--benchmark-frames", type=int, default=100)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--include-person-mask", action="store_true",
        help="Also run the checkpoint's person-mask decoder for the current 8 views.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "eval/eval_results/checkpoint_30_cached_inference_time.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    frame_dirs = sorted(
        (path for path in args.image_root.iterdir() if path.is_dir()), key=run_key
    )
    required = 1 + args.warmup_frames + args.benchmark_frames
    frame_dirs = frame_dirs[args.start_frame : args.start_frame + required]
    if len(frame_dirs) != required:
        raise RuntimeError(f"Need {required} frames, found {len(frame_dirs)}")

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "training/config")
    ):
        cfg = compose(config_name=args.config)
    load_started = time.perf_counter()
    model = instantiate(cfg.model, _recursive_=False)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    # The requested endpoint is SMPL parameters. These auxiliary heads do not
    # feed the temporal SMPL head and would measure different output products.
    disabled_heads = [
        "camera_head", "depth_head", "point_head", "track_head",
        "smpl_dense_landmark_head",
    ]
    if not args.include_person_mask:
        disabled_heads.append("person_mask_head")
    for name in disabled_heads:
        if hasattr(model, name):
            setattr(model, name, None)
    model.eval().to(device)
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    del checkpoint, state
    gc.collect()

    metadata = {
        "temporal_num_frames": torch.tensor([3], device=device, dtype=torch.long),
        "views_per_frame": torch.tensor([args.views], device=device, dtype=torch.long),
    }
    cache: deque[torch.Tensor] = deque(maxlen=3)
    output_keys = (
        "smpl_pose", "smpl_beta", "mesh_translate", "mesh_rot",
        "smpl_presence_logits",
    )
    records = []
    cold_record = None
    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    for index, frame_dir in enumerate(frame_dirs):
        image_paths = sorted(frame_dir.glob("*.jpg"))[: args.views]
        if len(image_paths) != args.views:
            raise RuntimeError(f"{frame_dir} has {len(image_paths)} JPEGs")
        total_started = time.perf_counter()

        started = time.perf_counter()
        images_cpu = load_and_preprocess_images([str(path) for path in image_paths])
        preprocess_seconds = time.perf_counter() - started

        started = time.perf_counter()
        images = images_cpu.unsqueeze(0).to(device)
        torch.cuda.synchronize(device)
        transfer_seconds = time.perf_counter() - started

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            started = time.perf_counter()
            current_features, patch_start_idx, _ = model.aggregator(images)
            torch.cuda.synchronize(device)
            aggregator_seconds = time.perf_counter() - started
            cache.append(current_features[-1].detach())

            started = time.perf_counter()
            outputs = cached_head_forward(model, cache, patch_start_idx, metadata)
            torch.cuda.synchronize(device)
            temporal_head_seconds = time.perf_counter() - started

            person_mask_head_seconds = 0.0
            if args.include_person_mask:
                started = time.perf_counter()
                mask_logits = model.person_mask_head(
                    outputs["person_tokens"],
                    current_features,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                torch.cuda.synchronize(device)
                person_mask_head_seconds = time.perf_counter() - started

        started = time.perf_counter()
        outputs_cpu = {
            key: outputs[key].float().cpu() for key in output_keys if key in outputs
        }
        torch.cuda.synchronize(device)
        output_to_cpu_seconds = time.perf_counter() - started
        total_seconds = time.perf_counter() - total_started
        record = {
            "frame": frame_dir.name,
            "jpeg_preprocess_seconds": preprocess_seconds,
            "host_to_gpu_seconds": transfer_seconds,
            "current_frame_aggregator_seconds": aggregator_seconds,
            "cached_temporal_head_seconds": temporal_head_seconds,
            "person_mask_head_seconds": person_mask_head_seconds,
            "smpl_output_to_cpu_seconds": output_to_cpu_seconds,
            "model_forward_seconds": (
                aggregator_seconds + temporal_head_seconds + person_mask_head_seconds
            ),
            "input_jpegs_to_smpl_cpu_seconds": total_seconds,
        }
        if index == 0:
            cold_record = record
        elif index > args.warmup_frames:
            records.append(record)
        del images_cpu, images, current_features, outputs, outputs_cpu
        if args.include_person_mask:
            del mask_logits

    if len(records) != args.benchmark_frames:
        raise AssertionError((len(records), args.benchmark_frames))
    fields = [key for key in records[0] if key != "frame"]
    timing = {field: summarize([row[field] for row in records]) for field in fields}
    peak_allocated = torch.cuda.max_memory_allocated(device)
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "config": args.config,
        "temporal_frames": 3,
        "views_per_frame": args.views,
        "cache": "rolling final aggregator tokens for previous two frames",
        "output_endpoint": (
            "SMPL pose/beta/translation/presence in CPU memory plus person masks on GPU"
            if args.include_person_mask else
            "SMPL pose/beta/translation/presence in CPU memory"
        ),
        "person_mask_decoder_included": bool(args.include_person_mask),
        "model_load_seconds_excluded_from_latency": model_load_seconds,
        "cold_first_frame": cold_record,
        "warmup_frames_excluded": args.warmup_frames,
        "benchmark_frames": args.benchmark_frames,
        "first_benchmark_frame": records[0]["frame"],
        "last_benchmark_frame": records[-1]["frame"],
        "steady_state_seconds": timing,
        "model_baseline_allocated_mib": baseline_allocated / 2**20,
        "peak_allocated_mib": peak_allocated / 2**20,
        "peak_increment_over_model_mib": (peak_allocated - baseline_allocated) / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
