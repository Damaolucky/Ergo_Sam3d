# Data Layout

## Expected Source Data

The pipeline assumes the following source layout on the Linux server:

```text
/mnt/lift_data/
  Data/
    <session>.tar.gz
  Annotation/
    Clips/
      <clip>.mp4
    Annotation_json/
      <session>.json
```

## Local Working Layout

By default, local outputs are written under:

```text
~/hzhou/
  outputs/
  cache/
```

## Main Output Files

The default keyframe sample role is action-aware:

- `*_lift.mp4` -> `<clip>__first_<height1>_<height1_strength>`
- `*_put.mp4` -> `<clip>__last_<height2>_<height2_strength>`

The generic placeholder below is `<role>_<position>`.

Before geometry preparation:

```text
~/hzhou/outputs/
  <clip>.mapping.json
  <clip>__<role>_<position>.rgb.png
  <clip>__<role>_<position>.depth_raw.npy
  <clip>__<role>_<position>.depth_meters.npy
  <clip>__<role>_<position>.depth_vis.png
  <clip>__<role>_<position>.intrinsics.pkl
  <clip>__<role>_<position>.sample_manifest.json
```

After geometry preparation:

```text
~/hzhou/outputs/<clip>__<role>_<position>/
  <clip>.mapping.json
  <clip>__<role>_<position>.rgb.png
  <clip>__<role>_<position>.depth_raw.npy
  <clip>__<role>_<position>.depth_meters.npy
  <clip>__<role>_<position>.depth_vis.png
  <clip>__<role>_<position>.intrinsics.pkl
  <clip>__<role>_<position>.sample_manifest.json
  pointcloud.npy
  pointcloud_rgb.ply
  pointcloud_preview.png
  geometry_stats.json
  human_mask.png
  human_mask.npy
  masked_depth_meters.npy
  human_pointcloud.npy
  human_pointcloud_rgb.ply
  human_pointcloud_preview.png
  mask_stats.json
  human_mesh.obj
  pred_vertices_3d.npy
  pred_joints_3d.npy
  smpl_params.json
  mesh_preview.png
  mesh_recovery_stats.json
  aligned_mesh.obj
  aligned_mesh_vertices.npy
  alignment_pointcloud_subset.npy
  alignment_pointcloud_subset_preview.png
  mesh_pointcloud_overlay_preview.png
  alignment_stats.json
  shelf_height_estimate.json
  shelf_height_preview.png
  shelf_height_report.png
  shelf_height_summary.txt
```

## Environment Variable Overrides

You can override the default layout with:

- `ERGO_WORK_ROOT`
- `ERGO_DATA_ROOT`
- `ERGO_CLIPS_DIR`
- `ERGO_JSON_ROOT`
- `ERGO_YOLO_MODEL`
