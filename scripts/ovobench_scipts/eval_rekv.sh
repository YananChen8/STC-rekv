#!/bin/bash
export PYTHONPATH="/home/chenyanan-20260210/STC/STC:$PYTHONPATH"

cd model/online_bench_inference/ovobench
# ReKV 特定配置
MODEL_NAME="rekv"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
MODEL_PATH="${MODEL_PATH:-/mnt/users/chenyanan-20260210/models}"
ANNO_PATH="/mnt/users/chenyanan-20260210/ovo-bench/ovo_bench_new.json"
VIDEO_DIR="/mnt/users/chenyanan-20260210/ovo-bench/src_videos/src_videos"
CHUNKED_DIR="/mnt/public/video_datasets/OVO-Bench/videos/chunked_videos"
RESULT_DIR="results/${TIMESTAMP}/"
MODE="online"

# export CUDA_VISIBLE_DEVICES="4,5,6,7"
NUM_GPUS=2              #
TOTAL_PROCESSES=4

RETRIEVE_SIZE=64

# 任务列表
TASKS="EPM ASI HLD STU OJR ATR ACR OCR FPD REC SSR CRR"
# TASKS="HLD CRR"
# TASKS="CRR"


echo "=========================================="
echo "Starting ReKV Distributed Inference"
echo "Number of GPUs: $NUM_GPUS"
echo "Processes per GPU: $PROCESSES_PER_GPU"
echo "Total Processes: $TOTAL_PROCESSES"
echo "Retrieve Size: $RETRIEVE_SIZE"
echo "=========================================="

for TASK in $TASKS; do
    echo "=========================================="
    echo "Processing task: $TASK"
    echo "=========================================="
    
    # 使用总进程数运行
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$TOTAL_PROCESSES \
        inference_distributed.py \
        --model $MODEL_NAME \
        --model_path "$MODEL_PATH" \
        --anno_path $ANNO_PATH \
        --video_dir $VIDEO_DIR \
        --chunked_dir $CHUNKED_DIR \
        --result_dir $RESULT_DIR \
        --mode $MODE \
        --task $TASK \
        --retrieve_size $RETRIEVE_SIZE \
        --save_results True \
        --global_seed 42 \
        --tf32
done

echo "=========================================="
echo "ReKV inference completed!"
echo "Results: $RESULT_DIR/$MODEL_NAME/"
echo "=========================================="