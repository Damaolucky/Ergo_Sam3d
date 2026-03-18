# Example Commands

Verified example:

- session: `2024-05-03_15`
- clip: `2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4`

```bash
export ERGO_WORK_ROOT=~/hzhou

SESSION="2024-05-03_15"
CLIP="2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4"

# Step 1: clip -> RGB/depth mapping
bash scripts/bash/run_clip_mapping.sh "$SESSION" "$CLIP"

# Step 2: extract one RGB frame and one depth frame
bash scripts/bash/run_extract_sample.sh "$CLIP.mapping.json"

# Step 3: move sample files into outputs/<clip>/ and prepare scene geometry
bash scripts/bash/run_prepare_geometry.sh "$CLIP.sample_manifest.json"

# Step 4: generate a human mask and masked human point cloud
bash scripts/bash/run_human_mask.sh "$CLIP"

# Step 5: run PCA-based geometry analysis
bash scripts/bash/run_analyze_human_geometry.sh "$CLIP"
```

Expected verified checkpoints from this example:

- mapping: `color_frame_range [16970, 17028]`
- geometry: `pointcloud points 800479`
- mask: `human_point_count 45045`
- human geometry: `yaw_degrees -88.50`
