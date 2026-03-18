# Workflow

## Purpose

The pipeline converts one annotated lift clip into:

- a verified RGB/depth frame mapping,
- one RGB sample and one aligned depth sample,
- a scene point cloud,
- a human mask and human-only point cloud,
- a coarse PCA-based human geometry summary.

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

## Optional SAM3D Trial

Entry points:

- `scripts/bash/setup_sam3d_body.sh`
- `scripts/bash/run_sam3d_trial.sh`

Current state:

- scaffold only
- blocked by model access approval for `facebook/sam-3d-body-dinov3`
- not yet verified end-to-end
