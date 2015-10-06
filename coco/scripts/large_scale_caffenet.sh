#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="coco/logs/large_scale_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py --gpu 3 \
  --solver coco/models/CaffeNet/solver.prototxt \
  --iters 160000 \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb coco_2014_train \
  --cfg coco/cfgs/coco_large_scale.yml \

time python ./tools/test_net.py --gpu 3 \
  --def coco/models/CaffeNet/test.prototxt \
  --net output/coco_large_scalec/coco_2014_train/caffenet_fast_rcnn_iter_160000.caffemodel \
  --imdb coco_2014_val \
  --cfg coco/cfgs/coco_large_scale.yml
