#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="coco/logs/coco_ctx_multiscale_vgg16_crowd.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 3 \
  --solver coco/models/VGG16/ctx/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --cfg coco/cfgs/coco_ctx_multiscale.yml \
  --imdb coco_2014_train \
  --iters 240000

time ./tools/test_net.py --gpu $1 \
  --def coco/models/VGG16/ctx/test.prototxt \
  --net output/coco_ctx_multiscale/coco_2014_train/vgg16_fast_rcnn_iter_240000.caffemodel \
  --cfg coco/cfgs/coco_multiscale.yml \
  --imdb coco_2014_val \
  --num_dets 100
