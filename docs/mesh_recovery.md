# Mesh Recovery Stage

## Selected Tool

This repository now uses **4D-Humans / HMR2** as the preferred next-stage human mesh recovery tool.

Why this choice:

- modern monocular human mesh baseline
- open-source and actively used
- checkpoint download is automatic
- outputs mesh vertices, joints, and SMPL-style parameters
- practical to integrate with the existing per-clip workflow

Primary source:

- official repo: `https://github.com/shubham-goel/4D-Humans`

## How this repo uses HMR2

Instead of relying on the official demo detector, this project uses the already verified `human_mask.npy` to compute a single-person crop box. That keeps the new stage consistent with the existing pipeline and avoids adding a separate detection dependency to the runtime path.

Entry points:

- `scripts/bash/setup_hmr2.sh`
- `scripts/bash/run_human_mesh_recovery.sh`
- `scripts/python/recover_human_mesh.py`

## Setup

Recommended server-side setup:

```bash
export ERGO_WORK_ROOT=~/hzhou
bash scripts/bash/setup_hmr2.sh
```

Optional environment variables:

- `ERGO_HMR2_ENV_NAME`
- `ERGO_HMR2_REPO`
- `ERGO_HMR2_SMPL_SOURCE`
- `ERGO_HMR2_PYTHON`
- `ERGO_HMR2_CHECKPOINT`
- `ERGO_OPENGL_PLATFORM`

## SMPL Requirement

HMR2 still requires the official **SMPL neutral model**.

Current expected filename:

- `SMPL_NEUTRAL.pkl`

Current expected target path after setup:

- `~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl`

This repo does **not** bundle that file and does **not** fetch unofficial mirrors.

## Outputs

`run_human_mesh_recovery.sh` writes the following into `~/hzhou/outputs/<clip_name>/`:

- `human_mesh.obj`
- `pred_vertices_3d.npy`
- `pred_joints_3d.npy`
- `smpl_params.json`
- `mesh_preview.png`
- `mesh_recovery_stats.json`

Notes:

- `pred_vertices_3d.npy` and `pred_joints_3d.npy` are saved in camera-space coordinates
- `smpl_params.json` stores rotation matrices, betas, crop camera, and full-frame camera translation

## Alignment Scaffold

Entry points:

- `scripts/bash/run_align_mesh.sh`
- `scripts/python/align_mesh_to_pointcloud.py`

Current behavior:

- loads `human_mesh.obj`
- loads `human_pointcloud.npy`
- preserves the camera vertical axis and solves only for yaw
- keeps the mesh's native human-height prior by default
- accepts `--target-human-height-m` when a known subject height is available
- saves `aligned_mesh.obj`
- saves `aligned_mesh_vertices.npy`
- saves `mesh_pointcloud_overlay_preview.png`
- saves `alignment_stats.json`

Current limitation:

- this is a baseline initialization only, not a final alignment method
