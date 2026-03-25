# Workflow

## Purpose

The pipeline converts one annotated lift clip into:

- a verified RGB/depth frame mapping,
- one RGB sample and one aligned depth sample,
- a scene point cloud,
- a human mask and human-only point cloud,
- a coarse PCA-based human geometry summary,
- a recovered human mesh and joints,
- a height-prior mesh-to-pointcloud alignment.

## Verified Run Order

For the verified example:

- session: `2024-05-03_15`
- clip: `2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4`

Run:

```bash
export ERGO_WORK_ROOT=~/hzhou

SESSION="2024-05-03_15"
CLIP="2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4"

bash scripts/bash/run_clip_mapping.sh "$SESSION" "$CLIP"
bash scripts/bash/run_extract_sample.sh "$CLIP.mapping.json"
bash scripts/bash/run_prepare_geometry.sh "$CLIP.sample_manifest.json"
bash scripts/bash/run_human_mask.sh "$CLIP"
bash scripts/bash/run_analyze_human_geometry.sh "$CLIP"

# New next-stage setup and execution
bash scripts/bash/setup_hmr2.sh
bash scripts/bash/run_human_mesh_recovery.sh "$CLIP"
bash scripts/bash/run_align_mesh.sh "$CLIP"
bash scripts/bash/run_align_mesh.sh "$CLIP" --target-human-height-m 1.72
```

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
- outputs `<clip>.mapping.json`

Verified example output:

- `color_frame_range: [16970, 17028]`
- `mid_color_frame.index: 16999`
- `nearest_depth_frame.index: 8511`

### 2. RGB/depth sample extraction

Entry points:

- `scripts/bash/run_extract_sample.sh`
- `scripts/python/extract_sample_from_mapping.py`

Behavior:

- extracts the midpoint RGB frame from the clip video with `ffmpeg`
- extracts the nearest depth frame from the tarball
- extracts `depth.scale.npy`
- extracts `depth.intrinsics.pkl`
- saves RGB, raw depth, metric depth, depth visualization, and sample manifest

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

### 5. Human geometry analysis

Entry points:

- `scripts/bash/run_analyze_human_geometry.sh`
- `scripts/python/analyze_human_geometry.py`

Behavior:

- runs PCA on `human_pointcloud.npy`
- writes `human_geometry.json`
- writes `human_pointcloud_pca_preview.png`

Verified example output:

- `centroid: [0.242, 0.047, 3.977]`
- `bbox_extent: [0.961, 2.506, 3.478]`
- `yaw_degrees: -88.50`

### 6. Human mesh recovery

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

### 7. Mesh-to-pointcloud height-prior alignment

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
- writes an aligned mesh and an overlay preview

Current state:

- verified on the example clip
- intended as a baseline for later refinement and cabinet-height reasoning

Expected outputs:

- `aligned_mesh.obj`
- `aligned_mesh_vertices.npy`
- `mesh_pointcloud_overlay_preview.png`
- `alignment_stats.json`
