#!/usr/bin/env bash
# AirV2X HEAL from the three shared Stage-0 bases.
# Set the checkpoint files and data paths before running this script.
set -euo pipefail

: "${VEH_CKPT:?Set VEH_CKPT to a homo vehicle .pth}"
: "${RSU_CKPT:?Set RSU_CKPT to a homo RSU .pth}"
: "${DRONE_CKPT:?Set DRONE_CKPT to a homo drone .pth}"
: "${P1_CKPT:?Set P1_CKPT to the original net_final_branch_v45.pth}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the AirV2X training split}"
: "${VAL_DATA:?Set VAL_DATA to an independent AirV2X validation split}"

python opencood/tools/train_airv2x_heter.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_heal/airv2x_HEAL_collab_camera.yaml \
  --vehicle_dir "${VEH_CKPT}" \
  --rsu_dir "${RSU_CKPT}" \
  --drone_dir "${DRONE_CKPT}" \
  --p1_checkpoint "${P1_CKPT}" \
  --root_dir "${TRAIN_DATA}" --validate_dir "${VAL_DATA}" \
  --tag heal_collab
