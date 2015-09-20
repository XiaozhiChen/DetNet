#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="kitti/logs/kitti_car$2_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver kitti/models/kitti_car/VGG16/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb kitti_car_train \
  --cfg kitti/cfgs/kitti_car$2.yml

time ./tools/test_net.py --gpu $1 \
  --def kitti/models/kitti_car/VGG16/test.prototxt \
  --net output/kitti_car$2/kitti_car_train/vgg16_fast_rcnn_iter_40000.caffemodel \
  --imdb kitti_car_val \
  --cfg kitti/cfgs/kitti_car$2.yml
