# Workflow

## Purpose

The pipeline converts one annotated lift clip into:

- a verified RGB/depth frame mapping,
- an action-aware keyframe RGB/depth sample for the shelf/object position,
- a scene point cloud,
- a human mask and human-only point cloud,
- a recovered human mesh and joints,
- a height-prior mesh-to-pointcloud alignment,
- a keyframe shelf/object height estimate.

## Verified Run Order

For the verified example:

- session: `2024-05-03_15`
- clip: `2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4`

Run:

```bash
export ERGO_WORK_ROOT=~/hzhou

SESSION="2024-05-03_15"
CLIP="2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4"

bash scripts/bash/run_keyframe_pipeline.sh "$SESSION" "$CLIP"

# Optional known-height calibration
ERGO_TARGET_HUMAN_HEIGHT_M=1.72 \
ERGO_KNOWN_HUMAN_HEIGHT_M=1.72 \
bash scripts/bash/run_keyframe_pipeline.sh "$SESSION" "$CLIP"
```

## Clip Action Rule

Clip filenames encode source and destination shelf positions:

```text
<session>_<camera>_<height1>_<height1_strength>_<height2>_<height2_strength>_<weight>_<ratio>_<take>_<action>.mp4
```

- `height1` / `height1_strength`: source position
- `height2` / `height2_strength`: destination position
- `*_lift.mp4`: process the first frame and label it with `height1`
- `*_put.mp4`: process the last frame and label it with `height2`

## Stage Details

### 1. Clip to frame/depth mapping

Entry points:

- `scripts/bash/run_clip_mapping.sh`
- `scripts/python/map_clip_to_frames_from_tar.py`

Behavior:

- reads the clip metadata JSON
- infers the source camera from `source_video`
- reads only the needed timestamp arrays from `/mnt/lift_data/Data/<session>.tar.gz`
- builds cache only for the current camera
- converts absolute timestamps into relative seconds
- infers the clip action from the filename suffix
- recommends `first` for `*_lift.mp4` and `last` for `*_put.mp4`
- outputs `<clip>.mapping.json`

Verified example output:

- `color_frame_range: [16970, 17028]`
- `first_color_frame.index: 16970`
- `last_color_frame.index: 17028`
- `mid_color_frame.index: 16999`
- `nearest_depth_frame.index: 8511`

### 2. RGB/depth sample extraction

Entry points:

- `scripts/bash/run_extract_sample.sh`
- `scripts/python/extract_sample_from_mapping.py`

Behavior:

- extracts the recommended keyframe RGB frame from the clip video with `ffmpeg`
- extracts the nearest keyframe depth frame from the tarball
- extracts `depth.scale.npy`
- extracts `depth.intrinsics.pkl`
- saves RGB, raw depth, metric depth, depth visualization, and one keyframe sample manifest

### 3. Geometry sample preparation

Entry points:

- `scripts/bash/run_prepare_geometry.sh`
- `scripts/python/prepare_geometry_sample.py`

Behavior:

- moves all sample artifacts into `~/hzhou/outputs/<clip_name>/`
- back-projects the full depth map into a scene point cloud
- writes `pointcloud.npy`, `pointcloud_preview.png`, and `geometry_stats.json`

Verified example output:

- depth shape: `(720, 1280)`
- valid ratio: `0.8686`
- point cloud points: `800479`

### 4. Human mask and human point cloud

Entry points:

- `scripts/bash/run_human_mask.sh`
- `scripts/python/generate_human_mask.py`

Behavior:

- runs `ultralytics` with `yolov8n-seg.pt`
- selects the largest `person` segmentation mask
- resizes the mask to the depth resolution if needed
- masks the depth map
- back-projects the masked depth map into `human_pointcloud.npy`

Verified example output:

- `mask_shape: [720, 1280]`
- `mask_num_pixels: 45968`
- `mask_ratio: 0.04988`
- `human_point_count: 45045`

### 5. Human mesh recovery

Entry points:

- `scripts/bash/setup_hmr2.sh`
- `scripts/bash/run_human_mesh_recovery.sh`
- `scripts/python/recover_human_mesh.py`

Behavior:

- clones and installs 4D-Humans / HMR2 as an external dependency
- uses the verified `human_mask.npy` to define the crop box
- runs HMR2 on the RGB sample
- saves mesh, joints, SMPL-style parameters, preview, and recovery stats

Current state:

- verified on the example clip after providing the official SMPL neutral model file
- server-compatible wrapper implemented and tested on GPU

Expected outputs:

- `human_mesh.obj`
- `pred_vertices_3d.npy`
- `pred_joints_3d.npy`
- `smpl_params.json`
- `mesh_preview.png`
- `mesh_recovery_stats.json`

### 6. Mesh-to-pointcloud height-prior alignment

Entry points:

- `scripts/bash/run_align_mesh.sh`
- `scripts/python/align_mesh_to_pointcloud.py`

Behavior:

- loads the recovered mesh and `human_pointcloud.npy`
- keeps the camera vertical axis fixed
- estimates only a yaw rotation in the X-Z plane
- uses the mesh's native human-height prior by default
- optionally accepts `--target-human-height-m` for explicit scale calibration
- uses torso-centered X/Z anchors and a lower-body Y anchor
- refines yaw/translation/scale with a multi-stage bounded partial-Chamfer objective inspired by SMPL-Fitting
- writes an aligned mesh and an overlay preview

Current state:

- verified on the example clip
- intended as the current partial scan-fitting method for cabinet-height reasoning

Expected outputs:

- `aligned_mesh.obj`
- `aligned_mesh_vertices.npy`
- `alignment_pointcloud_subset.npy`
- `alignment_pointcloud_subset_preview.png`
- `mesh_pointcloud_overlay_preview.png`
- `alignment_stats.json`

### 7. Shelf/object height estimation

Entry points:

- `scripts/bash/run_estimate_shelf_height.sh`
- `scripts/python/estimate_shelf_height.py`

Behavior:

- uses the prepared keyframe sample folder: first frame for `lift`, last frame for `put`
- uses the aligned human mesh feet as the floor reference when available
- searches the shelf-side depth ROI for the target level (`high`, `mid`, or `low`)
- computes height as `floor_y - target_y` because camera `Y` points down
- writes a JSON estimate, a text summary, an RGB overlay preview, and a histogram report for inspection

Expected outputs:

- `shelf_height_estimate.json`
- `shelf_height_preview.png`
- `shelf_height_report.png`
- `shelf_height_summary.txt`
