#!/bin/bash

# 提取命令行参数
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 计算等待时间（4小时 = 14400秒）
WAIT_TIME=14400

# 启动训练脚本
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} &

# 获取训练脚本的进程ID
TRAINING_PID=$!

# 等待4小时
sleep $WAIT_TIME

# 终止训练脚本的进程
kill $TRAINING_PID

echo "训练已在4小时后停止。"
