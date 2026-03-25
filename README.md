# Human Motion Geometry Preprocessing Pipeline

This repository preserves a verified Linux pipeline for:

1. mapping a human motion clip to RGB/depth frames,
2. extracting one RGB sample and one aligned depth sample,
3. building a scene point cloud,
4. generating a human mask and human-only point cloud,
5. running basic PCA-based human geometry analysis,
6. preparing a modern HMR2-based human mesh recovery stage,
7. preparing a height-prior mesh-to-pointcloud alignment stage.

The outputs are intended to support later mesh-depth alignment and quantitative geometry evaluation.

## Status

Verified stages:

- clip -> RGB/depth mapping
- RGB/depth sample extraction
- scene geometry preparation
- YOLO-based human mask generation
- PCA-based human geometry analysis
- HMR2 / 4D-Humans mesh recovery on the verified example clip
- height-prior mesh-to-pointcloud alignment on the verified example clip

Current caveats:

- HMR2 still requires the official SMPL neutral model file
- mesh alignment is still a baseline, not final registration

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
  scripts/
    bash/
      common.sh
      run_clip_mapping.sh
      run_extract_sample.sh
      run_prepare_geometry.sh
      run_human_mask.sh
      run_analyze_human_geometry.sh
      setup_hmr2.sh
      run_human_mesh_recovery.sh
      run_align_mesh.sh
    python/
      pipeline_utils.py
      map_clip_to_frames_from_tar.py
      extract_sample_from_mapping.py
      prepare_geometry_sample.py
      generate_human_mask.py
      analyze_human_geometry.py
      recover_human_mesh.py
      align_mesh_to_pointcloud.py
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

bash scripts/bash/run_clip_mapping.sh "$SESSION" "$CLIP"
bash scripts/bash/run_extract_sample.sh "$CLIP.mapping.json"
bash scripts/bash/run_prepare_geometry.sh "$CLIP.sample_manifest.json"
bash scripts/bash/run_human_mask.sh "$CLIP"
bash scripts/bash/run_analyze_human_geometry.sh "$CLIP"

# Optional next-stage setup
bash scripts/bash/setup_hmr2.sh

# New next-stage steps
bash scripts/bash/run_human_mesh_recovery.sh "$CLIP"
bash scripts/bash/run_align_mesh.sh "$CLIP"

# Optional: if the subject's real height is known, use it directly for scale calibration
bash scripts/bash/run_align_mesh.sh "$CLIP" --target-human-height-m 1.72
```

Outputs are written under `${ERGO_WORK_ROOT:-~/hzhou}/outputs/`.

Important behavior:

- Step 1 creates `<clip>.mapping.json`
- Step 2 creates `<clip>.sample_manifest.json` and sample files in `outputs/`
- Step 3 moves those sample files into `outputs/<clip>/`
- Steps 4 and 5 operate on `outputs/<clip>/`
- Step 6 writes mesh recovery artifacts into `outputs/<clip>/`
- Step 7 writes height-prior alignment artifacts into `outputs/<clip>/`

## Step Outputs

`*.mapping.json`

- clip-level mapping from metadata time to RGB frame range and nearest depth frame
- includes inferred camera, timestamp normalization info, and approximate FPS

`*.sample_manifest.json`

- manifest for the extracted RGB/depth sample pair
- points to the RGB image, depth arrays, intrinsics, and source mapping file

`geometry_stats.json`

- scene depth coverage statistics
- scene point cloud count and scene bounding box summary

`mask_stats.json`

- human mask coverage
- masked valid depth pixel count
- human point count and masked depth statistics

`human_geometry.json`

- PCA eigenvalues and axes for the human point cloud
- centroid, bounding box extent, and coarse yaw on the X-Z plane

`mesh_recovery_stats.json`

- HMR2 mesh recovery metadata
- crop box, device, checkpoint, and mesh/joint output locations

`alignment_stats.json`

- yaw-only transform from recovered mesh space to human point-cloud space
- optional height calibration, lower-body anchoring, and height-reference stats for later cabinet-height estimation

## Verified Example Notes

Verified mapping output for the example clip includes:

- `color_frame_range: [16970, 17028]`
- `mid_color_frame.index: 16999`
- `nearest_depth_frame.index: 8511`
- `approx_color_fps ~= 30`
- `approx_depth_fps ~= 15`

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

Verified human geometry example:

- `num_points: 45045`
- `centroid: [0.242, 0.047, 3.977]`
- `bbox_extent: [0.961, 2.506, 3.478]`
- `yaw_degrees: -88.50`

Verified mesh recovery example:

- `num_vertices: 6890`
- `num_joints: 44`
- `device: cuda`

Verified height-prior alignment example:

- aligned mesh extent: approximately `[1.027, 1.658, 0.722]` with native mesh height prior
- `mesh_native_height_m`: approximately `1.55`
- `observed_pointcloud_height_raw_m`: approximately `2.23`
- the new alignment intentionally avoids scaling the mesh to match contaminated point-cloud depth thickness

## Known Limitations

- Timestamps in the session tarballs are absolute millisecond timestamps and must be converted to relative seconds by subtracting the first frame time.
- YOLO segmentation masks may be produced at a lower resolution than the depth frame and must be resized with nearest-neighbor before masking depth.
- Human point clouds can still contain background contamination.
- The current yaw estimate is only a coarse PCA-based orientation, not a reliable human facing direction.
- HMR2 mesh recovery still requires the official SMPL neutral model file even though the checkpoint download itself is automatic.
- The current mesh alignment stage is yaw-only and height-prior; it is more stable than full 3D PCA, but it is still not a final registration method.

See [docs/workflow.md](docs/workflow.md), [docs/data_layout.md](docs/data_layout.md), [docs/known_issues.md](docs/known_issues.md), and [docs/mesh_recovery.md](docs/mesh_recovery.md) for more detail.

## Next Steps

- clean the human point cloud before geometry estimation
- estimate cabinet geometry in the same depth frame and compare its top height against the aligned human height reference
- refine mesh-to-pointcloud alignment beyond the current height-prior baseline
- evaluate orientation, scale, and position more robustly
