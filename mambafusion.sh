#!/bin/bash
source /home/suyi/conda_envs/airv2x_h200/bin/activate
# conda activate mambafusion
# export _FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1
# 使用 12.1 这套 toolkit
# export CUDA_HOME=/usr/local/cuda-12.1
# export CUDA_PATH=/usr/local/cuda-12.1
# export PATH=/usr/local/cuda-12.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
MODEL_DIR="/home/suyi/AirV2X-Perception_copy/AirV2X-Perception-Checkpoints/airv2x_intermediate_mambafusion"
LOG_DIR='/home/suyi/AirV2X-Perception_copy/opencood/logs/airv2x_intermediate_mambafusion/default_2026_03_09_09_28_12'
# MODEL_DIR="/home/suyi/AirV2X-Perception_old/AirV2X-Perception-Checkpoints/airv2x_intermediate_where2comm/release"
gpu_spec=${1:-0}
train=${2:-test}
pretrained=${3:-False}
ddp=${4:-False}
visible_gpus=${5:-${gpu_spec}}

if [ "$train" = "test" ]; then
    python opencood/tools/inference_multi_scenario.py \
        --model_dir "${LOG_DIR}" \
        --eval_best_epoch \
        --gpu_id ${gpu_spec} \
        --save_vis
elif [ "$train" = "train" ] && [ "$ddp" = "True" ]; then
    if [ "$pretrained" = "True" ]; then
        config_path="${LOG_DIR}/config.yaml"
        model_dir_args=(--model_dir "${LOG_DIR}")
    else
        config_path="${MODEL_DIR}/config.yaml"
        model_dir_args=()
    fi

    IFS=',' read -r -a gpu_array <<< "${visible_gpus}"
    nproc_per_node=${#gpu_array[@]}

    if [ "${nproc_per_node}" -lt 2 ]; then
        echo "DDP mode requires at least 2 GPUs in the visible GPU list."
        echo "Example: bash mambafusion.sh 0,1 train False True"
        exit 1
    fi

    export CUDA_VISIBLE_DEVICES="${visible_gpus}"
    torchrun --standalone --nproc_per_node="${nproc_per_node}" \
        opencood/tools/train.py \
        -y "${config_path}" \
        "${model_dir_args[@]}"
elif [ "$train" = "train" ]; then
    if [ "$pretrained" = "True" ]; then
        python opencood/tools/train.py \
            -y "${LOG_DIR}/config.yaml" \
            --gpu_id ${gpu_spec} \
            --model_dir "${LOG_DIR}"
    else
        python opencood/tools/train.py \
            -y "${MODEL_DIR}/config.yaml" \
            --gpu_id ${gpu_spec}
    fi
else
    echo "Unsupported mode: train=${train}, ddp=${ddp}"
    exit 1
fi
