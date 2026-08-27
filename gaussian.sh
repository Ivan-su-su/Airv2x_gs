#!/bin/bash
source /home/suyi/conda_envs/airv2x_h200/bin/activate
# conda activate mambafusion
# export _FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1
# 使用 12.1 这套 toolkit
# export CUDA_HOME=/usr/local/cuda-12.1
# export CUDA_PATH=/usr/local/cuda-12.1
# export PATH=/usr/local/cuda-12.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
MODEL_DIR="AirV2X-Perception-Checkpoints/airv2x_intermediate_gaussian"
# MODEL_DIR="/home/suyi/AirV2X-Perception_old/AirV2X-Perception-Checkpoints/airv2x_intermediate_where2comm/release"
gpu_id=${1:-1}
# train
train=${2:-train}
if [ "$train" = "test" ]; then
    python opencood/tools/inference_multi_scenario.py \
        --model_dir "${MODEL_DIR}" \
        --eval_best_epoch \
        --gpu_id ${gpu_id}
fi

if [ "$train" = "train" ]; then
    python opencood/tools/train.py \
        -y "${MODEL_DIR}/config.yaml" \
        --gpu_id ${gpu_id}
fi

# 双卡（多卡）分布式训练，推荐使用 torchrun 启动 DDP
# if [ "$train" = "train" ]; then
#     # 注意：nproc_per_node=2 代表用2块GPU，CUDA_VISIBLE_DEVICES 可指定实际使用哪两块；下方以0,1为例
#     CUDA_VISIBLE_DEVICES=0,1 torchrun \
#         --standalone --nproc_per_node=2 \
#         opencood/tools/train.py \
#         -y "${MODEL_DIR}/config.yaml"
# fi

: '
用法举例：
# 单卡训练
bash gaussian.sh 0 train False

# 双卡训练（推荐）
bash gaussian.sh 0,1 train True

# 测试
bash gaussian.sh 0 test

说明：
- gpu_id：默认0,1，可改为相应显卡编号
- train：train 或 test
- ddp：是否用分布式训练（True=多卡，False=单卡）

常见双卡报错：
如遇 NCCL 错误，优先检查 cuda/驱动、PyTorch、通信端口等。
'