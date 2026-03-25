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

## Mesh alignment is only a coarse initialization

`align_mesh_to_pointcloud.py` currently performs a PCA-based similarity transform. This is useful for scaffolding and initial debugging, but it is not a final registration method.

## SAM3D is blocked by model access approval

`setup_sam3d_body.sh` and `run_sam3d_trial.sh` are kept as scaffold scripts. The current blocker is checkpoint access for `facebook/sam-3d-body-dinov3`, not a local code bug.
