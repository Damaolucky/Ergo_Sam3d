# Human Motion Geometry Preprocessing Pipeline

This repository preserves a verified Linux pipeline for:

1. mapping a human motion clip to RGB/depth frames,
2. extracting the action-aware keyframe RGB/depth sample for the shelf/object position,
3. building a scene point cloud,
4. generating a human mask and human-only point cloud,
5. running a modern HMR2-based human mesh recovery stage,
6. running a height-prior mesh-to-pointcloud alignment stage,
7. estimating the corresponding target height from the final hand position in the same depth frame.

The outputs are intended to support later mesh-depth alignment and quantitative geometry evaluation.

## Status

Verified stages:

- clip -> RGB/depth mapping
- RGB/depth sample extraction
- scene geometry preparation
- YOLO-based human mask generation
- HMR2 / 4D-Humans mesh recovery on the verified example clip
- height-prior mesh-to-pointcloud alignment on the verified example clip
- keyframe hand-anchored target-height estimation on the verified example clip

Current caveats:

- HMR2 still requires the official SMPL neutral model file
- mesh alignment is a partial scan registration, not a full SMPL pose-fitting optimization

## Data Assumptions

Default paths:

- session tarballs: `/mnt/lift_data/Data/<session>.tar.gz`
- clip videos: `/mnt/lift_data/Annotation/Clips/`
- clip metadata JSON: `/mnt/lift_data/Annotation/Annotation_json/`
- local working root: `~/hzhou`

Optional environment variable overrides:

- `ERGO_WORK_ROOT`
- `ERGO_DATA_ROOT`
- `ERGO_CLIPS_DIR`
- `ERGO_JSON_ROOT`
- `ERGO_YOLO_MODEL`

## Clip Naming and Keyframe Rule

Clip filenames follow the metadata pattern:

```text
<session>_<camera>_<height1>_<height1_strength>_<height2>_<height2_strength>_<weight>_<ratio>_<take>_<action>.mp4
```

For example:

```text
2024_05_03_15_sagittal_high_24_low_19_5_3_1_lift.mp4
```

The fields mean:

- `height1` / `height1_strength`: source shelf/object position
- `height2` / `height2_strength`: destination shelf/object position
- `action`: either `lift` or `put`

The production keyframe rule is:

- `*_lift.mp4`: use the first frame, because the object is still at the source position
- `*_put.mp4`: use the last frame, because the object has reached the destination position

So a full `low -> high` motion is represented by two clips: the `lift` clip uses `first_low_<strength>`, and the `put` clip uses `last_high_<strength>`.

## Repository Layout

```text
repo_root/
  README.md
  .gitignore
  requirements.txt
  environment.yml
  docs/
    workflow.md
    data_layout.md
    known_issues.md
    mesh_recovery.md
  examples/
    example_commands.md
    verified_results/
  scripts/
    bash/
      common.sh
      run_clip_mapping.sh
      run_extract_sample.sh
      run_prepare_geometry.sh
      run_human_mask.sh
      setup_hmr2.sh
      run_human_mesh_recovery.sh
      run_align_mesh.sh
      run_estimate_shelf_height.sh
      run_keyframe_pipeline.sh
    python/
      pipeline_utils.py
      map_clip_to_frames_from_tar.py
      extract_sample_from_mapping.py
      prepare_geometry_sample.py
      generate_human_mask.py
      recover_human_mesh.py
      align_mesh_to_pointcloud.py
      estimate_shelf_height.py
```

## Environment Setup

The original workflow assumes a conda environment named `hzhou`.

Create it with:

```bash
conda env create -f environment.yml
conda activate hzhou
```

Or install the Python packages with:

```bash
python -m pip install -r requirements.txt
```

Required tools and packages:

- `numpy`
- `matplotlib`
- `Pillow`
- `ultralytics`
- `huggingface_hub`
- `ffmpeg` on `PATH`

Notes:

- `ffmpeg` is required by `extract_sample_from_mapping.py`
- `yolov8n-seg.pt` should not be committed; Ultralytics can download it when needed
- the mesh recovery stage is designed around an external HMR2 / 4D-Humans install
- no `color_intrinsic` or color-depth extrinsic calibration file is currently available in the dataset layout, so RGB human masks are resized into depth space approximately rather than geometrically reprojected

## HMR2 Mesh Recovery Setup

The new mesh recovery stage uses [4D-Humans / HMR2](https://github.com/shubham-goel/4D-Humans) as an external dependency.

Recommended setup:

```bash
export ERGO_WORK_ROOT=~/hzhou
bash scripts/bash/setup_hmr2.sh
```

Important note:

- HMR2 automatically downloads its checkpoints
- HMR2 still requires the official SMPL neutral model file
- after downloading that file yourself, rerun:

```bash
ERGO_HMR2_SMPL_SOURCE=/path/to/SMPL_NEUTRAL.pkl bash scripts/bash/setup_hmr2.sh
```

## End-to-End Workflow

Verified example:

- session: `2024-05-03_15`
- clip: `2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4`

Run order:

```bash
export ERGO_WORK_ROOT=~/hzhou

SESSION="2024-05-03_15"
CLIP="2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4"

bash scripts/bash/run_keyframe_pipeline.sh "$SESSION" "$CLIP"

# Optional: if the subject's real height is known, use it directly for scale calibration
ERGO_TARGET_HUMAN_HEIGHT_M=1.72 \
ERGO_KNOWN_HUMAN_HEIGHT_M=1.72 \
bash scripts/bash/run_keyframe_pipeline.sh "$SESSION" "$CLIP"
```

Outputs are written under `${ERGO_WORK_ROOT:-~/hzhou}/outputs/`.

Important behavior:

- Step 1 creates `<clip>.mapping.json`
- Step 2 creates one action-aware keyframe sample by default, such as `<clip>__first_high_24.*` for a `lift` clip or `<clip>__last_high_24.*` for a `put` clip
- Step 3 moves that keyframe sample into its own folder under `outputs/<clip>__<role>_<position>/`
- Step 4 writes human mask and human point-cloud artifacts into that endpoint folder
- Step 5 writes mesh recovery artifacts into that endpoint folder
- Step 6 writes height-prior alignment artifacts into that endpoint folder
- Step 7 writes hand-anchored target-height estimates into that endpoint folder

## Step Outputs

`*.mapping.json`

- clip-level mapping from metadata time to RGB frame range and nearest depth frame
- includes inferred camera, timestamp normalization info, approximate FPS, and endpoint sample specs for `first` and `last`

`*.sample_manifest.json`

- manifest for one endpoint RGB/depth sample pair
- records the `sample_role` (`first` or `last`), the position label parsed from the clip metadata, and the output files for that endpoint

`geometry_stats.json`

- scene depth coverage statistics
- scene point cloud count and scene bounding box summary

`mask_stats.json`

- human mask coverage
- masked valid depth pixel count
- human point count and masked depth statistics

`mesh_recovery_stats.json`

- HMR2 mesh recovery metadata
- crop box, device, checkpoint, and mesh/joint output locations

`alignment_stats.json`

- yaw-only transform from recovered mesh space to human point-cloud space
- optional height calibration, lower-body anchoring, overlap metrics, and the mesh-guided alignment subset stats used for later cabinet-height estimation

`shelf_height_estimate.json`

- keyframe target height estimate in meters
- primary value is the final hand height in the selected keyframe
- floor reference used for `height = floor_y - target_y`
- source or destination position label parsed from the clip metadata, such as `high_24`
- ratio between the estimated target height and aligned human height

`shelf_height_preview.png`

- RGB overlay showing the detected hand anchor, its local depth patch, and the pixels used for the final estimate

`shelf_height_report.png`

- combined RGB overlay and local hand-patch height histogram
- easiest file to inspect when checking whether the hand-anchored target-height estimate is plausible

`shelf_height_summary.txt`

- short text readout of the final hand-anchored target height, uncertainty band, human-height ratio, and method

## Verified Example Notes

Verified mapping output for the example clip includes:

- `color_frame_range: [16970, 17028]`
- `first_color_frame.index: 16970`
- `last_color_frame.index: 17028`
- `mid_color_frame.index: 16999`
- `nearest_depth_frame.index: 8511`
- `approx_color_fps ~= 30`
- `approx_depth_fps ~= 15`

For this clip, the new endpoint sample folders are named:

- `2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4__first_high_24`

Verified geometry example:

- depth shape: `(720, 1280)`
- valid ratio: `0.8686`
- point cloud points: `800479`

Verified human mask example:

- `mask_shape: [720, 1280]`
- `mask_num_pixels: 45968`
- `mask_ratio: 0.04988`
- `human_point_count: 45045`
- `human_depth_mean_m: 3.9768`

Verified mesh recovery example:

- `num_vertices: 6890`
- `num_joints: 44`
- `device: cuda`

Verified height-prior alignment example:

- estimated human height: approximately `1.60 m`
- refined scale multiplier: approximately `0.99`
- alignment-subset-to-mesh mean distance: approximately `0.047 m`
- alignment-subset-to-mesh p95 distance: approximately `0.131 m`
- the alignment intentionally avoids scaling the mesh to match contaminated point-cloud depth thickness

Verified hand-anchored target-height example:

- the current estimator locks onto the operator's final hand position in the action-aware keyframe
- results are stored per clip in `shelf_height_estimate.json`, `shelf_height_preview.png`, and `shelf_height_report.png`
- unlike the older wide ROI method, this number is a hand-anchored proxy for the source/destination height, not a direct cabinet-surface reconstruction

## Known Limitations

- Timestamps in the session tarballs are absolute millisecond timestamps and must be converted to relative seconds by subtracting the first frame time.
- YOLO segmentation masks may be produced at a lower resolution than the depth frame and must be resized with nearest-neighbor before masking depth.
- No `color_intrinsic` or color-depth extrinsic calibration file is currently available, so RGB masks cannot yet be rigorously reprojected into the depth camera frame.
- Human point clouds can still contain background contamination.
- HMR2 mesh recovery still requires the official SMPL neutral model file even though the checkpoint download itself is automatic.
- The current mesh alignment stage is yaw-only and height-prior with multi-stage partial-Chamfer refinement; it is more stable than full 3D PCA, but it does not deform the SMPL pose/body shape.
- The current target-height stage measures the final hand height in the keyframe and uses it as a proxy for the source/destination shelf height. It is not yet an explicit cabinet-surface detector.

See [docs/workflow.md](docs/workflow.md), [docs/data_layout.md](docs/data_layout.md), [docs/known_issues.md](docs/known_issues.md), and [docs/mesh_recovery.md](docs/mesh_recovery.md) for more detail.

Curated production result snapshots are also stored under
`examples/verified_results/` for lightweight GitHub review.

## Next Steps

- clean the human point cloud before geometry estimation
- estimate cabinet geometry in the same depth frame and compare its top height against the aligned human height reference
- refine mesh-to-pointcloud alignment with explicit SMPL pose/shape fitting if needed
- evaluate orientation, scale, and position more robustly
