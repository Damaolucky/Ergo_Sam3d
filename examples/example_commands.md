# Example Commands

Verified example:

- session: `2024-05-03_15`
- clip: `2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4`

```bash
export ERGO_WORK_ROOT=~/hzhou

SESSION="2024-05-03_15"
CLIP="2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4"

# Run the full production pipeline.
bash scripts/bash/run_keyframe_pipeline.sh "$SESSION" "$CLIP"

# Optional: use a known subject height to set/calibrate the final scale.
ERGO_TARGET_HUMAN_HEIGHT_M=1.72 \
ERGO_KNOWN_HUMAN_HEIGHT_M=1.72 \
bash scripts/bash/run_keyframe_pipeline.sh "$SESSION" "$CLIP"
```

Expected verified checkpoints from this example:

- mapping: `color_frame_range [16970, 17028]`
- geometry: `pointcloud points 800479`
- mask: `human_point_count 45045`

Current next-stage status:

- HMR2 mesh recovery verified on the example clip once the official SMPL neutral model is available
- alignment stage verified as a height-prior, partial-Chamfer fitting method on the example clip
- target-height estimation runs on the action-aware keyframe sample and now uses the final hand position in that keyframe as the primary anchor
