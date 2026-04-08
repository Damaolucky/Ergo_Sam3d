# Known Issues

## Timestamp normalization is required

Session timestamps are stored as absolute timestamps, typically milliseconds since epoch. The verified workflow converts them to seconds and subtracts the first frame timestamp before comparing them to clip-relative times.

## YOLO mask resolution may not match depth resolution

`yolov8n-seg.pt` can return masks at a lower resolution such as `(384, 640)` while the depth image is `(720, 1280)`. The verified workflow resizes the binary mask to the depth resolution with nearest-neighbor interpolation before applying it.

## Human point clouds still contain background contamination

The current human mask is good enough to produce a first-pass human point cloud, but the result can still include background points. This is visible in the large `bbox_extent.z` values for the verified example.

## Yaw is only a coarse orientation cue

`yaw_degrees` in `human_geometry.json` comes from the first PCA axis on the X-Z plane. It should not be treated as the real human facing direction.

## HMR2 still requires the official SMPL neutral model

The selected mesh recovery stage uses 4D-Humans / HMR2. Its model checkpoints download automatically, but the official SMPL neutral model file is still required before mesh recovery can run end-to-end.

## Mesh alignment is still a partial registration

`align_mesh_to_pointcloud.py` uses a yaw-only, height-prior initialization followed by multi-stage partial-Chamfer refinement. This is more stable than the older full-3D PCA transform, but single-view depth still observes only the visible body surface while the mesh is a full body, so exact full-mesh overlap is not expected.

## Shelf/object height estimation is automatic ROI geometry

`estimate_shelf_height.py` estimates the target shelf/object height from the prepared keyframe depth ROI and uses the aligned human mesh feet as the floor reference. The prepared keyframe is `first` for `*_lift.mp4` and `last` for `*_put.mp4`. The result should be checked with `shelf_height_preview.png`, especially for clips where the shelf is occluded, the object is not yet at the expected source/destination position, or the target side is not the default right side.
