#!/bin/bash

set -e
set -x

data_dir="/data1/coco/train"

export DETECTRON2_DATASETS=/data1/
cd /home/QiuCongpei/code/DcDp_on_ViTs/ICLR_24/iBOT_Copy

master_addr='localhost'
master_port=10001

path_base=/home/QiuCongpei/code/DcDp_on_ViTs/ICLR_24/iBOT_Copy/work_dirs/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for mr_max in 0.4
  do
  output_dir=${path_base}/mr_max_${mr_max}_200eps_lossr0.3_dcr0.25_bs32
  python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes 1 --node_rank 0 --master_addr ${master_addr} --master_port ${master_port} \
    main_ibot.py \
    --arch vit_small \
    --output_dir ${output_dir} \
    --data_path ${data_dir} \
  --teacher_temp 0.07 \
  --warmup_teacher_temp_epochs 40 \
  --norm_last_layer false \
  --epochs 200 \
  --lr 0.001 \
  --batch_size_per_gpu 32 \
  --shared_head true \
  --out_dim 8192 \
  --pred_shape rcc \
  --local_crops_number 0 \
  --global_crops_scale 0.14 1. \
  --pred_ratio 0 ${mr_max} \
  --pred_ratio_var 0 0.2 \
  --dc
done

sh /home/QiuCongpei/code/DcDp_on_ViTs/ICLR_24/iBOT_Copy/configs/Evaluation.sh
