# Known Issues

## Timestamp normalization is required

Session timestamps are stored as absolute timestamps, typically milliseconds since epoch. The verified workflow converts them to seconds and subtracts the first frame timestamp before comparing them to clip-relative times.

## YOLO mask resolution may not match depth resolution

`yolov8n-seg.pt` can return masks at a lower resolution such as `(384, 640)` while the depth image is `(720, 1280)`. The verified workflow resizes the binary mask to the depth resolution with nearest-neighbor interpolation before applying it.

## Human point clouds still contain background contamination

The current human mask is good enough to produce a first-pass human point cloud, but the result can still include background points. This is visible in the large `bbox_extent.z` values for the verified example.

## Color-depth calibration files are not available

The current dataset layout does not provide a `color_intrinsic` file or an explicit color-depth extrinsic transform. Because of that, the pipeline cannot yet reproject RGB segmentation results into depth space rigorously. `generate_human_mask.py` therefore resizes the RGB-derived mask to the depth resolution, which is workable but approximate and can contribute to hand / body point-cloud leakage.

## HMR2 still requires the official SMPL neutral model

The selected mesh recovery stage uses 4D-Humans / HMR2. Its model checkpoints download automatically, but the official SMPL neutral model file is still required before mesh recovery can run end-to-end.

## Mesh alignment is still a partial registration

`align_mesh_to_pointcloud.py` uses a yaw-only, height-prior initialization followed by multi-stage partial-Chamfer refinement. This is more stable than the older full-3D PCA transform, but single-view depth still observes only the visible body surface while the mesh is a full body, so exact full-mesh overlap is not expected.

## Target-height estimation is hand-anchored, not cabinet-surface reconstruction

`estimate_shelf_height.py` now estimates the target height from the operator's final hand position in the prepared keyframe and uses the aligned human mesh feet as the floor reference. The prepared keyframe is `first` for `*_lift.mp4` and `last` for `*_put.mp4`. This is a better proxy for where the object is being handled, but it is still not a direct cabinet-surface reconstruction. The result should be checked with `shelf_height_preview.png`, especially when the hand is occluded, the mask leaks background, or the target side is not the default right side.
