# train car model, VGG16, with orientation estimation and context
./kitti/scripts/kitti_car_vgg16.sh 0

# train car model, VGG16, with orientation estimation, no context
./kitti/scripts/kitti_car_vgg16_ort.sh 0

# train car model, VGG16, the original fast rcnn architechture
./kitti/scripts/kitti_car_vgg16_frcn.sh 0 _no_ort

# train ped/cyc model, VGG16, with orientation estimation and context
./kitti/scripts/kitti_ped_cyc_vgg16.sh 0

