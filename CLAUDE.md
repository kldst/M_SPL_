# CLAUDE.md — vggt_multi 0621_mamma demo

VGGT multi-person SMPL demo. Entry point is `demo_gradio_smpl_multi.py`, launched via
`run_demo_mamma_dance.sh` (conda env `mamma`, checkpoint
`training/logs/0621_mamma/ckpts/checkpoint_step_10000.pt` — the base `ckpt/model.pt`
has NO SMPL head weights). Model = `Aggregator` (24× alternating frame/global attention)
+ `smpl_multi_query_trans_rot_head` (`smpl_num_people` person queries cross-attend to patch
tokens; config default now **5**, was 20). Two optional auxiliary heads
(`enable_smpl_dense_landmark` / `enable_person_mask`) reuse those person tokens — see
"Dense-landmark + per-person-mask heads" below.

> **2026-07 mask 改版**：pixel-level DPT mask head + Hungarian mask cost 的完整
> 動機/實作/實測紀錄在 **`CLAUDE_MASK_DPT.md`**（config `mamma_mask_dpt.yaml`）。

## Known bugs & gotchas

### 1. GT camera lookup: image stem vs. archive view-id mismatch (FIXED)

**Symptom:** ticking **Use GT** crashes with
`FileNotFoundError: GT camera params for '0000_runs_00000_IOI_01' not found in .../out_data/runs_00000.npz`.
Also breaks the non-GT path (see gauge note below): no SMPL mesh, no projection, only
cameras render.

**Root cause:** for the *organized* dataset (`MAMMA_eval_dance`, GT in `out_data/*.npz`),
cameras are keyed by **bare view id** — `cam_param_min/IOI_01/{intrinsics.K_flat9,
extrinsics.worldToCamera12}`. But images copied into `target_dir/images` are renamed to
`<frame>_<run>_<view>` (e.g. `0000_runs_00000_IOI_01`), and the code queried
`cam_param_min/<full_stem>/…` → miss. The SMPL-param side (`out_param/<subj>/frame_.../
smpl_params/...`) matched fine; only the camera lookup was broken.

**Fix:** `load_gt_multi_from_image_dir()` now resolves the stem's trailing view id back to
an available `cam_param_min/<view>` key (exact match first, else `stem.endswith("_"+view)`).

### 2. Coordinate gauge / avg_scale — projection looks "spread across the frame"

The predicted SMPL mesh is placed in a **camera0-normalized gauge**. Two paths in
`gradio_demo` (search `has_gt_camera_for_mesh`):
- **GT-camera present:** vertices normalized to GT cam0 gauge, `avg_scale` derived from GT
  cameras (e.g. ~6.4) → mesh projects tightly onto people. Correct.
- **No GT camera** (plain image folder / `has_gt_camera_metadata` False): identity
  extrinsics + **`avg_scale=1.0`** (uncalibrated) → absolute scale unknown, so the body
  projects huge/fills the frame. This is inherent, **not a bug**; the projection gallery
  looks equally "spread" there.

Consequence: bug #1 also forced runs_* into the *failing* GT path (npz exists →
`has_gt_camera_metadata` True). Fixing #1 restores proper `avg_scale` and tight projection.

### 3. Two dataset formats — different GT loaders

- `MAMMA_eval_dance` (organized, GT in `out_data/<run>.npz`) → `load_gt_multi_from_image_dir`
  npz branch. Layout: `<root>/<split>/out_image/<run>/IOI_*.jpg`.
- `Mamma_mv_split` (raw, GT in `.data.pyd`) → `load_raw_mamma_gt_from_sequence`
  (`_looks_like_raw_mamma_sequence` branch, reads `cam_int`/`cam_ext` from the pyd; does
  NOT use `cam_param_min`, so unaffected by bug #1). Layout:
  `…/png/<seq>/<IOI_view>/<frame>.jpg` (+ `.data.pyd`, `.mask.jpg`).

## Attention visualization (added)

Cross-attention of the 20 person queries over image patches is captured and visualized.
- Standalone: `visualize_attention.py` (supports `--mv-root`/`--seq-dir`/`--frame` for
  Mamma_mv_split, and the eval `--dataset-root`/`--scene` layout). Exports the reusable
  helpers `CrossAttnCapture`, `select_attention`, `overlay_heatmap`, `make_grid`,
  `add_title_bar`.
- In the demo: `run_model` wraps the forward with `CrossAttnCapture` and writes per-person
  overlays + `attn_grid.npy` to `target_dir/attention/`. Two galleries are chained after
  the Reconstruct click:
  - `run_attention_gallery` — per-person heatmap over all views.
  - `run_attention_smpl_gallery` — attention heatmap + reprojected-mesh centroid + convex
    hull outline + attention-vs-mesh centroid pixel gap (a self-consistency metric).
    Reuses `project_world_points_to_cam` + stored gauge-space `smpl_vertices` +
    `smpl_visible_indices` (maps mesh block j ↔ query slot). The hull is suppressed when
    the mesh projects too large (uncalibrated gauge, per bug #2); centroid+gap always show.

Aggregator self-attention is NOT captured (fused `F.scaled_dot_product_attention`).

## Dense-landmark + per-person-mask heads (added)

Two auxiliary heads make each person query focus on ONE person (identity
disentangling in crowded/interacting scenes). Both **reuse the SMPL head's
person tokens** (`person_tokens`, `(B,P,C)`, now surfaced from
`SMPLMultiQueryTransRotHead`), so every head's slot `p` stays bound to the same
identity with NO extra matching. They plug into the existing Hungarian +
`has_smpl` machinery.

- `vggt/heads/smpl_dense_landmark_head.py` — `SMPLDenseLandmarkHead`. 512 vertex
  queries (`landmark_embed[i] + person_token[p]`) cross-attend **each view's own
  patch tokens** and predict **direct per-view 2D** `smpl_landmarks2d (B,S,P,512,2)`
  in `[-1,1]` + `smpl_landmarks_logvar (B,S,P,512)`. Same family as MAMMA's
  `MammaNetDecoder` but person-conditioned + multi-view (per-view decode). No
  camera/gauge dependency. Memory scales with `smpl_num_people*512` queries — see
  `DenseLandmarkHeadConfig.max_context_tokens` to cap the per-view context.
- `vggt/heads/person_mask_head.py` — `PersonMaskHead`. Dot-product of a projected
  person token with each view's patch tokens → `person_mask_logits (B,S,P,H,W)`
  at patch resolution (37×37 for 518/14). Cheap (~0.8M params).
- `vggt/heads/person_mask_head.py` — **`PersonMaskDPTHead`** (added later): pixel-level
  mask head. A DPT trunk (feature_only) runs ONCE over the aggregator's 4 intermediate
  layers → per-pixel embedding map at `H/person_mask_down_ratio` (259×259 for 518, ratio 2);
  person tokens project to the same dim and dot-product against it (Mask2Former-style)
  → `person_mask_logits (B,S,P,259,259)`. ~33M params, full model fwd+bwd peak ≈8GB
  (4 views, frozen aggregator). Select via `model.person_mask_head_type: dpt` +
  `person_mask_down_ratio` / `person_mask_embed_dim`; `"dot"` keeps the old head.
  Warm start: `checkpoint.init_mask_trunk_from_depth: True` copies the pretrained
  `depth_head.*` DPT trunk from model.pt into `person_mask_head.trunk.*` (56 tensors;
  the feature_only `output_conv1` differs in shape and is skipped).
- Enabled via config `model.enable_smpl_dense_landmark` / `enable_person_mask`.
  `smpl_num_people` (config knob) sets the query-slot count; keep it `>=` the
  dataset's `max_num_people` (config sets both to 5). Measured memory (frozen
  aggregator, 4 views, 16GB RTX 4070 Ti): base+SMPL 6.0GB, +mask +0.1GB,
  +dense(**5** people, full ctx) 9.5GB. At **20** people full ctx OOMs (12.4GB at
  ctx=2048); reduce `smpl_num_people` or cap `max_context_tokens`.
- These heads/losses only activate for the **raw Mamma_mv_split** path (which
  ships the GT). On organized datasets (MAMMA_syn) the batch has no
  `smpl_landmarks2d`/`person_mask`, so both losses are guarded to 0 and existing
  training is unaffected even with the heads enabled.

### GT (raw Mamma_mv_split only — free, no SMPL decode)
`training/data/landmark_mask_gt.py` derives targets from what the `.data.pyd`
already ships: `verts_512.pkl` is a `(512,10475)` partition-of-unity matrix `M`,
so landmark 2D = `M @ vertices2d`, visibility = `M @ vertex_visibility`; the
per-person mask is `*.mask.jpg` (pixel value == `person_idx+1`, occlusion-aware)
down-sampled to the patch grid. `SysSMPLMultiDataset` (raw path) emits
`smpl_landmarks2d (S,P,512,2)` (normalised), `smpl_landmarks2d_visibility (S,P,512)`,
`person_mask (S,P,pg,pg)` when `emit_landmarks` / `emit_person_mask` are set.
The mask GT grid is `H_final // person_mask_stride` (dataset knob): `None` keeps the
legacy patch grid (stride=patch_size → 37×37); `2` emits the pixel-level 259×259 GT
for the DPT mask head (must match `model.person_mask_down_ratio`; `compute_mask_loss`
bilinearly resamples the logits if they disagree). Config `mamma_mask_dpt.yaml` is the
landmark-OFF / DPT-mask-ON / mask-cost-matching ablation of `mamma_full.yaml`. The
mask rides through `process_one_image`'s `extra_maps` (nearest interp) so it gets
the identical crop/resize as the image; landmarks ride the `track` argument.
`base_dataset.process_one_image` gained an optional `extra_maps=None` param
(in/out dict, transformed in place) — backward compatible.
To train on the raw data, point the config's `SysSMPL_DIR` /
`SysSMPL_ANNOTATION_DIR` at `Mamma_mv_split/train` and set `emit_landmarks` /
`emit_person_mask: True` (a commented example is in `config/0621_mamma.yaml`).

### Loss (split into modules; `loss.py` is now a thin aggregator)
`loss.py` re-exports everything (backward compatible — `from training.loss import
...` still works) and keeps `MultitaskLoss`. Heavy code lives in:
- `loss_camera.py` — `compute_camera_loss`, `camera_loss_single`.
- `smpl_body.py` — SMPL/SMPL-X model (`_TorchSMPLX`), `_decode_smpl(x)_batch`,
  gender helpers, axis-angle↔rotmat, gauge/normalise/project, `compute_gt_mesh_*`.
  Leaf module (torch/numpy only).
- `smpl_matching.py` — `apply_hungarian_matching` (+ `_binary_cross_entropy_prob`).
  Imports `compute_gt_mesh_translate` from `smpl_body`. The cost matrix is
  vectorized (no per-pair Python loop) and built under `no_grad`. Besides
  pose/beta/trans/mesh_trans/presence it supports a **2D mask cost**
  (`hungarian_cost_mask_weight` in `loss.smpl`): pred `person_mask_logits` and GT
  `person_mask` are adaptive-avg-pooled to `hungarian_mask_cost_grid`² (default 32,
  resolution-independent) and compared per (pred,GT) pair with soft BCE via two
  matmuls. This pins slot↔person assignment when people are in contact (pose /
  mesh_translate costs tie there, image masks don't). Matched-pair diagnostics
  `hungarian_mask_cost` / `hungarian_presence_cost` are returned in the loss dict.
- `loss_smpl.py` — the loss functions: `compute_smpl_loss` (~790 lines, the bulk),
  `smpl_losses_plus_from_axis_angle`, `binary_focal_loss_with_logits`,
  `compute_smpl_3d_joint_loss`, **`compute_landmark_loss`** (direct-2D GNLL, gated
  by visibility + `has_smpl`). It re-exports smpl_body/smpl_matching names into its
  own namespace, so `loss.py`'s `from training.loss_smpl import ...` needs no change.
  Landmark + mask are computed INSIDE `compute_smpl_loss` so they share its
  Hungarian match + people-flatten; weights `weight_landmark`, `landmark_loss_type`,
  `weight_mask` live in the `loss.smpl` config block.
- `loss_mask.py` — `compute_mask_loss` (BCE on patch occupancy + soft-IoU metric).
Module deps (no cycles): `smpl_body` ← `smpl_matching` ← `loss_smpl` → `loss_mask`.
Backups: `loss_original_backup.py` (pre-any-split monolith) and
`loss_smpl_presplit_backup.py` (before the smpl_body/matching split).

### Debug scripts (decode-free, use `debug/mamma_raw_io.py`, not the slow
`mamma_debug_common`)
- `debug_04_landmark_mask_gt_projection.py` — overlays GT 512 landmarks + instance
  mask + patch-grid occupancy on a real 4-view frame (validates the GT).
- `debug_05_landmark_mask_loss_zero.py` — feeds GT back as prediction; asserts
  landmark GNLL == 0 and (soft-target) mask BCE reaches its binary-entropy floor
  (≪ the zero-prediction baseline of ln2≈0.693). Both scripts pass on
  `be_HsuS3iLSSWWZ_seq_000001` frame 0000.
- `debug_06_real_dataloader_check.py` — builds the **real** `SysSMPLMultiDataset`
  (raw path) and pulls one sample via `get_data`, then overlays the loader's
  OUTPUT (landmarks denormalised from [-1,1], 24 joints, per-person patch mask)
  onto each processed view — validates the full loader incl. `process_one_image`
  crop/resize + track/extra_maps transforms. Use `--max-sequences` / `--max-frames`
  to bound the build.

**Raw-build cost / knobs:** `_build_raw_mamma_sequences` cold-reads one pyd per
`(frame, view)`. A "scene" dir (e.g. `harmony4d_train_1_NC_200_00`) holds ~34
sequences under `png/`, each ~19 frames × 8 views → thousands of 2.2MB cold
reads on first run (warm cache is fast). `SysSMPLMultiDataset` now takes optional
`max_sequences` / `max_frames_per_sequence` (default None) to bound the build for
fast debug / quick-iteration startup.

Both new heads were smoke-tested in isolation (shapes + backward) and the full
`VGGT` forward+backward with both heads enabled was memory-probed on GPU; the
dataset→model→loss key names / `(B,S,P,...)` layouts were verified to align
(Hungarian gather + people-flatten cover the new keys). A single-process
end-to-end train step was NOT run here (loading ~150 large pyds is slow in this
env) — run one `debug_mode` / small `limit_train_batches` step on the real box to
confirm.

## Temporal inference: aggregator feature cache (added)

`VGGT.forward` folds the temporal axis into the batch axis before the aggregator
(`[B,T*V,3,H,W] -> [B*T,V,3,H,W]`, see `use_temporal_path_enabled` in
`vggt/models/vggt.py`), so global attention never crosses a timestep and frame
`t`'s tokens depend only on frame `t`'s views. A causal T=3 sliding window
therefore re-encoded every frame **three times**. `AggregatorTokenCache` +
`cached_temporal_forward` (both in
`inference/infer_temporal_smpl_mesh_hungarian_mp4.py`, imported by
`inference/infer_temporal_topk_mp4.py`) keep the previous `clip_length-1`
frames' final aggregator tokens and replay them, so each frame is encoded — and
JPEG-decoded — exactly once.

- On by default in both MP4 scripts; `--no-feature-cache` restores the original
  re-encode path. `run_ck30_all_sequences.sh` picks the cache up unchanged.
- All three scripts moved from the repo root into `inference/` (2026-09-07).
  They import repo-root siblings by plain name, so each now prepends
  `REPO_DIR = Path(__file__).resolve().parents[1]` to `sys.path` before those
  imports — the same bootstrap the other `inference/*.py` already used.
  `eval_checkpoint30_mask_refine_absolute.py` (repo root) gained
  `sys.path.insert(0, REPO/"inference")` for the same reason.
- **Verified exact**: with autocast disabled the cached and full-window paths
  agree bit-for-bit on `smpl_pose` / `smpl_beta` / `mesh_translate` /
  `smpl_presence_logits` (max abs diff 0.0). Under bf16 autocast they differ by
  ~1e-2 purely because the aggregator runs at batch 3 vs batch 1 (bf16 eps
  ~4e-3); `person_mask_logits` differs ~2e-2 in fp32 from batch-dependent cudnn
  conv algorithm selection, and only matters for `--translate-refine-mask`.
- Measured on a 4090, T=3/V=8: model forward **0.867 -> 0.284 s/frame (3.05x)**;
  peak activation memory also drops ~3x, so the cached path still fits when two
  jobs share one 24GB GPU and the non-cached path OOMs.
- `inference/eval_mamma_dance_temporal_mpjpe_vpe.py` already cached features; it gained a
  `FramePrefetcher` (`--prefetch-depth`, default 3) that decodes each frame's
  JPEGs and GT npz on worker threads. Frame preprocessing costs about as much
  wall time as the cached forward and uses the CPU rather than the GPU, so
  reading ahead takes it off the critical path: 0.62 -> 0.42 s/frame end to end,
  with MPJPE/VPE bit-identical to the sequential path.

## Running / headless testing

- Launch: `./run_demo_mamma_dance.sh` (or `python demo_gradio_smpl_multi.py --config
  0621_mamma --checkpoint <ckpt> --dataset-root <root> --dataset-split test`).
- **SMPL + landmark/mask demo:** `./run_demo_landmark_mask.sh` → `demo_gradio_landmark_mask.py`
  (env `CUDA_VISIBLE_DEVICES`/`CKPT`/`PORT`, default GPU 1 / checkpoint_300 / 7861). A
  self-contained Gradio app that loads a trained checkpoint, runs one raw-Mamma scene, and
  shows:
  - an interactive **3D scene** (`gr.Model3D` / `.glb`) with the predicted SMPL-X meshes
    (per-person colour) and GT meshes (translucent grey) in the SAME world frame;
  - per view: **predicted & GT SMPL reprojected** onto the image, **dense landmarks**,
    **patch masks**, and **person-query attention**;
  - a metric table: SMPL vertex reprojection L2, landmark L2 (px), mask IoU (Hungarian
    slot→person).
  Predicted SMPL is placed via the `mesh_translate` head in the cam0 gauge (`avg_scale`
  from `normalize_camera_extrinsics_points_and_3djoints_batch` on GT extrinsics), then the
  gauge is inverted back to world (`V=(V_g·scale−t0)@R0`) so the **GT world cameras**
  (`batch["extrinsics"]/["intrinsics"]`, OpenCV `_project_points_opencv`) reproject it —
  tight, apples-to-apples with GT. SMPL-X decode via `smpl_body._decode_smplx_batch`
  (10475 verts, `use_mamma`), faces from `smplx_models/neutral/model.pkl["f"]`. It does NOT
  import the heavy `demo_gradio_smpl_multi.py` (that one parse_args+build+`launch()` at
  import); the gauge math is reimplemented compactly. Unlike the main demo it launches
  under `main()`, so it's importable for headless tests (stub `gr.Blocks.launch`). Reuses
  `visualize_attention` helpers + `SysSMPLMultiDataset`. Static HTML variant (landmarks/
  mask/attention only): `debug/make_overfit_report.py` (base64 PNGs → `report.html`).
- `demo.launch()` runs at **module import time** (not under `if __name__=='__main__'`), so
  importing the module starts the server. To test functions headlessly, stub it before
  `runpy.run_path`: `gr.Blocks.launch = lambda self,*a,**k: None`, and prepend the repo +
  `training/` to `sys.path`. See scratchpad test scripts used during development.
- `gradio_demo` returns a fixed 9-tuple — do NOT change its signature; add outputs via new
  `.then(...)` steps and extra gallery components instead.
