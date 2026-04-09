# Verified Result Snapshots

This folder stores curated, lightweight snapshots of two production runs from
the remote server. These are not the full raw output folders. Large artifacts
such as `*.npy`, `*.ply`, and recovered meshes are intentionally omitted here so
that GitHub keeps only the files needed for review:

- key RGB/depth previews,
- human-mask and alignment previews,
- final hand-anchored target-height reports,
- summary JSON/TXT files.

The two clips form a matched `low_19 -> high_24` pair from the same take:

| Clip | Action | Keyframe rule | Sample folder | Final target label | Estimated target height |
| --- | --- | --- | --- | --- | --- |
| `2024_05_03_15_sagittal_low_19_high_24_4_3_1_lift.mp4` | `lift` | first frame | `...__first_low_19` | `low_19` | `0.3602 m` |
| `2024_05_03_15_sagittal_low_19_high_24_4_3_1_put.mp4` | `put` | last frame | `...__last_high_24` | `high_24` | `1.1177 m` |

These snapshots were regenerated from the remote production environment after
the hand-anchored target-height change landed. The current estimator measures
the operator's final hand height in the action-aware keyframe, rather than the
older wide shelf ROI.

Per-sample folders include:

- `*.sample_manifest.json`: keyframe choice and output manifest
- `*.mapping.json`: clip-to-frame/depth mapping
- `geometry_stats.json`: scene point-cloud summary
- `mask_stats.json`: human-mask / human-point-cloud summary
- `mesh_recovery_stats.json`: HMR2 runtime summary
- `alignment_stats.json`: mesh-to-pointcloud alignment metrics
- `shelf_height_estimate.json`: final hand-anchored target-height estimate
- `shelf_height_summary.txt`: short text summary
- preview images for RGB, depth, human mask, point cloud, mesh alignment, and height report
