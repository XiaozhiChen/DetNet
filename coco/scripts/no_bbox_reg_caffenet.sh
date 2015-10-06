#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="coco/logs/no_bbox_reg_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py --gpu 3 \
  --solver coco/models/CaffeNet/no_bbox_reg/solver.prototxt \
  --iters 100000 \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb coco_2014_train \
  --cfg coco/cfgs/coco.yml

time python ./tools/test_net.py --gpu 3 \
  --def coco/models/CaffeNet/no_bbox_reg/test.prototxt \
  --net output/coco_baseline/coco_2014_train/caffenet_fast_rcnn_no_bbox_reg_iter_100000.caffemodel \
  --imdb coco_2014_val \
  --cfg coco/cfgs/coco.yml
