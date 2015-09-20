#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="kitti/logs/test_kitti_ped_cyc$2_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver kitti/models/kitti_ped_cyc/VGG16/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb kitti_ped_cyc_trainval \
  --cfg kitti/cfgs/kitti_ped_cyc$2.yml

time ./tools/test_net.py --gpu $1 \
  --def kitti/models/kitti_ped_cyc/VGG16/test.prototxt \
  --net output/kitti_ped_cyc$2/kitti_ped_cyc_trainval/vgg16_fast_rcnn_iter_40000.caffemodel \
  --imdb kitti_ped_cyc_test \
  --cfg kitti/cfgs/kitti_ped_cyc$2.yml
