#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="kitti/logs/kitti_car$2_vgg_cnn_m_1024.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver kitti/models/kitti_car/VGG_CNN_M_1024/solver.prototxt \
  --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \
  --imdb kitti_car_train \
  --cfg kitti/cfgs/kitti_car$2.yml

time ./tools/test_net.py --gpu $1 \
  --def kitti/models/kitti_car/VGG_CNN_M_1024/test.prototxt \
  --net output/kitti_car$2/kitti_car_train/vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel \
  --imdb kitti_car_val \
  --cfg kitti/cfgs/kitti_car$2.yml
