#!/usr/bin/env bash
# Run stage1 first; stage2 starts a separate run from its Stage-1 checkpoint.
set -euo pipefail

: "${VEH_CKPT:?Set VEH_CKPT to a homo vehicle .pth}"
: "${RSU_CKPT:?Set RSU_CKPT to a homo RSU .pth}"
: "${DRONE_CKPT:?Set DRONE_CKPT to a homo drone .pth}"
: "${P1_CKPT:?Set P1_CKPT to the original net_final_branch_v45.pth}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the AirV2X training split}"
: "${VAL_DATA:?Set VAL_DATA to an independent AirV2X validation split}"

stage=${1:-stage1}
common=(--vehicle_dir "${VEH_CKPT}" --rsu_dir "${RSU_CKPT}"
        --drone_dir "${DRONE_CKPT}" --p1_checkpoint "${P1_CKPT}"
        --root_dir "${TRAIN_DATA}" --validate_dir "${VAL_DATA}")
if [[ "${stage}" == stage1 ]]; then
  python opencood/tools/train_airv2x_heter.py \
    -y opencood/hypes_yaml/airv2x/camera/det/airv2x_negocollab/airv2x_negocollab_stage1_nego.yaml \
    "${common[@]}" --tag nego_stage1
elif [[ "${stage}" == stage2 ]]; then
  : "${NEGO_STAGE1_CKPT:?Set NEGO_STAGE1_CKPT to a stage1 .pth}"
  python opencood/tools/train_airv2x_heter.py \
    -y opencood/hypes_yaml/airv2x/camera/det/airv2x_negocollab/airv2x_negocollab_stage2_ft.yaml \
    "${common[@]}" --init_from "${NEGO_STAGE1_CKPT}" --tag nego_stage2
else
  echo "Usage: bash opencood/tools/train_negocollab.sh stage1|stage2" >&2
  exit 2
fi
