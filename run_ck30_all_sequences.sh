#!/usr/bin/env bash
# Run the cached causal T=3 / V=8 temporal model (model/root/checkpoint_30.pt) over EVERY
# sequence of the MAMMA dance eval set, one video per sequence.
#
# MAMMA_eval_dance/test is 18 separate takes concatenated into runs_00000..runs_06230, so
# each take is a separate --start-frame/--max-frames slice: that way the causal window is
# padded from the take's own first frame instead of leaking the previous take.
#
# Sequence boundaries come from the dataset's own log (the same source
# inference/eval_mamma_dance_temporal_mpjpe_vpe.py parses).
#
#   ./run_ck30_all_sequences.sh                        # all 18, resumable, MODE=topk
#   MODE=refine ./run_ck30_all_sequences.sh            # GT-Hungarian slots + mask translate refine
#   SEQUENCES=Basic_Whip ./run_ck30_all_sequences.sh   # only takes matching this substring
#
# MODE=topk    inference/infer_temporal_topk_mp4.py   -- slots from the model's presence logits,
#                                              mesh_translate used as predicted (no refine)
# MODE=refine  inference/infer_temporal_smpl_mesh_hungarian_mp4.py -- slots matched to the GT meshes,
#                                              mesh_translate refined against the masks
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PY=/train-data-3-hdd/yian/conda/envs/mamma/bin/python
DATASET=${DATASET:-/train-data-3-hdd/yian/Multi_SMPL_0706/MAMMA_eval_dance}
CHECKPOINT=${CHECKPOINT:-$REPO_DIR/model/root/checkpoint_30.pt}
CONFIG=${CONFIG:-mamma_harmony4d_mask_dpt}
MODE=${MODE:-topk}
case "$MODE" in
  topk)   SCRIPT=inference/infer_temporal_topk_mp4.py;              EXTRA=(--top-k "${TOP_K:-2}" --dedup "${DEDUP:-nms3d}") ;;
  refine) SCRIPT=inference/infer_temporal_smpl_mesh_hungarian_mp4.py; EXTRA=(--translate-refine-mask) ;;
  *) echo "unknown MODE=$MODE (use topk or refine)" >&2; exit 1 ;;
esac
OUT_ROOT=${OUT_ROOT:-$REPO_DIR/debug/ck30_all18_$MODE}
SMPL=${SMPL:-/train-data-3-hdd/yian/Multi_SMPL_0706/smpl_models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl}
LOG=$DATASET/test/logs/success.log
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PYTORCH3D_PROJECTION_PYTHON=${PYTORCH3D_PROJECTION_PYTHON:-$PY}

mkdir -p "$OUT_ROOT"

# "SUCCESS <name>: <n> frames, ... (run_offset now <end>)" -> name, start, count
mapfile -t ROWS < <(sed -nE 's/.*SUCCESS ([^:]+): ([0-9]+) frames.*run_offset now ([0-9]+).*/\1 \2 \3/p' "$LOG")
echo "[ck30] mode=$MODE script=$SCRIPT out=$OUT_ROOT"
echo "[ck30] ${#ROWS[@]} sequences from $LOG"

for row in "${ROWS[@]}"; do
  read -r name count end <<<"$row"
  start=$((end - count))
  if [[ -n "${SEQUENCES:-}" && "$name" != *"$SEQUENCES"* ]]; then continue; fi
  out="$OUT_ROOT/$name"
  if [[ -f "$out/manifest.json" ]]; then
    echo "[ck30] skip $name (already done)"
    continue
  fi
  echo "[ck30] $name: frames $start..$((end - 1)) ($count)"
  $PY "$SCRIPT" \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" \
      --dataset-root "$DATASET" --output-dir "$out" \
      --start-frame "$start" --max-frames "$count" \
      --num-input-views 8 --clip-length 3 \
      "${EXTRA[@]}" \
      --smpl-model "$SMPL" > "$out.log" 2>&1 \
    && echo "[ck30] done $name" \
    || echo "[ck30] FAILED $name (see $out.log)"
done
echo "[ck30] all sequences finished"
