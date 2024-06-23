#!/bin/bash

set -e
set -x

data_dir="data/coco/train"
output_dir="work_dirs/taskname"
cd /path/to/project

master_addr='localhost'
master_port=10001

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node=8 \
  --nnodes 1 --node_rank 0 --master_addr ${master_addr} --master_port ${master_port} \
  main_ibot.py \
  --arch vit_small \
  --output_dir ${output_dir} \
  --data_path ${data_dir} \
--teacher_temp 0.07 \
--warmup_teacher_temp_epochs 80 \
--norm_last_layer false \
--epochs 800 \
--lr 0.001 \
--batch_size_per_gpu 64 \
--shared_head true \
--out_dim 8192 \
--pred_shape rcc \
--local_crops_number 0 \
--global_crops_scale 0.14 1. \
--pred_ratio 0 0.4  \
--pred_ratio_var 0 0.2 \
--dc_ratio 0.3 \
--dc_sample_ratio 0.25 \
--dc

