#!/usr/bin/env bash

cd /path/to/project

task_name=your_work_name

# COCO Detection and Instance Segmentation
bash ./run.sh coco_det ${task_name} vit_small teacher 8   data.samples_per_gpu=4   lr_config.step=8,11   runner.max_epochs=12   optimizer.paramwise_cfg.layer_decay_rate=0.8

# ADE20K Semantic Segmentation
bash ./run.sh ade20k_seg ${task_name} vit_small teacher 4 \
  data.samples_per_gpu=4 \
  model.backbone.out_with_norm=true \
  optimizer.lr=3e-5