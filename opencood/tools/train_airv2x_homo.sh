#!/usr/bin/env bash
# Shared Stage-0 homogeneous bases. Train each once, then reuse.
set -euo pipefail

python opencood/tools/train.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_vehicle_camera.yaml \
  --tag homo_vehicle

python opencood/tools/train.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_rsu_camera.yaml \
  --tag homo_rsu

python opencood/tools/train.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_drone_camera.yaml \
  --tag homo_drone
