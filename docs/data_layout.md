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
  sam-3d-body/   # optional external dependency, not tracked by this repo
```

## Main Output Files

Before geometry preparation:

```text
~/hzhou/outputs/
  <clip>.mapping.json
  <clip>.rgb.png
  <clip>.depth_raw.npy
  <clip>.depth_meters.npy
  <clip>.depth_vis.png
  <clip>.intrinsics.pkl
  <clip>.sample_manifest.json
```

After geometry preparation:

```text
~/hzhou/outputs/<clip>/
  <clip>.mapping.json
  <clip>.rgb.png
  <clip>.depth_raw.npy
  <clip>.depth_meters.npy
  <clip>.depth_vis.png
  <clip>.intrinsics.pkl
  <clip>.sample_manifest.json
  pointcloud.npy
  pointcloud_preview.png
  geometry_stats.json
  human_mask.png
  human_mask.npy
  masked_depth_meters.npy
  human_pointcloud.npy
  human_pointcloud_preview.png
  mask_stats.json
  human_geometry.json
  human_pointcloud_pca_preview.png
  human_mesh.obj
  pred_vertices_3d.npy
  pred_joints_3d.npy
  smpl_params.json
  mesh_preview.png
  mesh_recovery_stats.json
  aligned_mesh.obj
  aligned_mesh_vertices.npy
  mesh_pointcloud_overlay_preview.png
  alignment_stats.json
```

## Environment Variable Overrides

You can override the default layout with:

- `ERGO_WORK_ROOT`
- `ERGO_DATA_ROOT`
- `ERGO_CLIPS_DIR`
- `ERGO_JSON_ROOT`
- `ERGO_YOLO_MODEL`
