#!/usr/bin/env bash
# AirV2X STAMP adapter training from the three shared Stage-0 bases.
set -euo pipefail

VEH_DIR=${VEH_DIR:-opencood/logs/airv2x_homo_vehicle_camera/homo_vehicle}
RSU_DIR=${RSU_DIR:-opencood/logs/airv2x_homo_rsu_camera/homo_rsu}
DRONE_DIR=${DRONE_DIR:-opencood/logs/airv2x_homo_drone_camera/homo_drone}

python opencood/tools/train_airv2x_heter.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_stamp/airv2x_stamp_collab_camera.yaml \
  --vehicle_dir "${VEH_DIR}" \
  --rsu_dir "${RSU_DIR}" \
  --drone_dir "${DRONE_DIR}" \
  --tag stamp_collab
