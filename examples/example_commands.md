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

# Step 2: extract the final endpoint RGB+depth sample
bash scripts/bash/run_extract_sample.sh "$CLIP.mapping.json"

LAST_SAMPLE="${CLIP}__last_high_24"

# Step 3: move the final endpoint sample into outputs/<sample>/ and prepare scene geometry
bash scripts/bash/run_prepare_geometry.sh "${LAST_SAMPLE}.sample_manifest.json"

# Step 4: generate a human mask and masked human point cloud
bash scripts/bash/run_human_mask.sh "$LAST_SAMPLE"

# Step 5: run PCA-based geometry analysis
bash scripts/bash/run_analyze_human_geometry.sh "$LAST_SAMPLE"

# Step 6: setup HMR2 / 4D-Humans once
bash scripts/bash/setup_hmr2.sh

# Step 7: recover a human mesh with HMR2
bash scripts/bash/run_human_mesh_recovery.sh "$LAST_SAMPLE"

# Step 8: run the height-prior mesh-to-pointcloud alignment
bash scripts/bash/run_align_mesh.sh "$LAST_SAMPLE"

# Step 9: estimate the destination shelf/object height from the final frame
bash scripts/bash/run_estimate_shelf_height.sh "$LAST_SAMPLE"

# Optional: use a known subject height to set/calibrate the final scale explicitly
bash scripts/bash/run_align_mesh.sh "$LAST_SAMPLE" --target-human-height-m 1.72
bash scripts/bash/run_estimate_shelf_height.sh "$LAST_SAMPLE" --known-human-height-m 1.72
```

Expected verified checkpoints from this example:

- mapping: `color_frame_range [16970, 17028]`
- geometry: `pointcloud points 800479`
- mask: `human_point_count 45045`
- human geometry: `yaw_degrees -88.50`

Current next-stage status:

- HMR2 mesh recovery verified on the example clip once the official SMPL neutral model is available
- alignment stage verified as a height-prior, partial-Chamfer fitting method on the example clip
- shelf/object height estimation verified on the final-frame `last_high_24` example
