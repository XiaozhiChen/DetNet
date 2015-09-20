# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from fast_rcnn.config import cfg
import datasets
import datasets.kitti
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
# import utils.cython_bbox
import cPickle
import subprocess
import time
from utils.cython_bbox import bbox_overlaps, bbox_ioa

class kitti(datasets.imdb):
    def __init__(self, image_set, subset, devkit_path=None):
        datasets.imdb.__init__(self, 'kitti_' + subset + '_' + image_set)
        self._image_set = image_set
        self._subset = subset
        # self._devkit_path = self._get_default_path() if devkit_path is None \
                            # else devkit_path
        self._data_path = self._get_default_data_path()
        if subset == 'car':
            self._classes = ('__background__', # always index 0
                             'car')
        elif subset == 'ped_cyc':
            self._classes = ('__background__', # always index 0
                             'pedestrian', 'cyclist')
        assert self.num_classes >= 2, 'ERROR: incorrect subset'

        # print self._devkit_path, self._data_path
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        # assert os.path.exists(self._devkit_path), \
                # 'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'image_2',
               index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(datasets.ROOT_DIR, 'data/kitti/object/ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

#     def _get_default_path(self):
        # """
        # Return the default path where KITTI is expected to be installed.
        # """
        # return os.path.join(datasets.ROOT_DIR, 'data', 'kitti/object/devkit/matlab')

    def _get_default_data_path(self):
        """
        Return the default data path where PASCAL VOC is expected to be installed.
        """
        if self._image_set in ['train', 'val', 'trainval']:
                return os.path.join(datasets.ROOT_DIR, 'data/kitti/object/training')
        elif self._image_set == 'test':
                return os.path.join(datasets.ROOT_DIR, 'data/kitti/object/testing')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def dontcare_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_dontcare_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} dontcare roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        dontcare_roidb = [self._load_dontcare_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(dontcare_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote dontcare roidb to {}'.format(cache_file)

        return dontcare_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_3DOP_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                '3DOP',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_boxes = sio.loadmat(filename)['boxes'].ravel()

        # box:[x1 y1 x2 y2]
        box_list = []
        for i in xrange(raw_boxes.shape[0]):
           box_list.append(raw_boxes[i][:, (1, 0, 3, 2)] - 1)

        # ignore "DontCare" regions, but it doesn't help
#        if self._image_set in {'val', 'test'}:
#            for i in xrange(raw_boxes.shape[0]):
#                box_list.append(raw_boxes[i][:, (1, 0, 3, 2)] - 1)
#        else:
#            # only use samples which have IoU < 0.15 and IoA < 0.1 with "DontCare" regions
#            dontcare_roidb = self.dontcare_roidb()
#            assert len(dontcare_roidb) == raw_boxes.shape[0]
#            for i in xrange(raw_boxes.shape[0]):
#                dontcare_boxes = dontcare_roidb[i]['boxes']
#                boxes = raw_boxes[i][:, (1, 0, 3, 2)] - 1
#                ious = bbox_overlaps(boxes.astype(np.float),
#                                         dontcare_boxes.astype(np.float))
#                ioas = bbox_ioa(boxes.astype(np.float),
#                                dontcare_boxes.astype(np.float))
#                iou_ioas = (ious < 0.15) & (ioas < 0.1)
#                I = iou_ioas.all(axis=1)
#                boxes = boxes[I,:]
#                box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'label_2', index + '.txt')
        # print 'Loading: {}'.format(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        num_objs = len(lines)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        alphas = np.zeros((num_objs), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = -1
        for line in lines:
            obj = line.strip().split(' ')
            try:
                cls = self._class_to_ind[obj[0].lower().strip()]
            except:
                continue
            # ignore objects with undetermined difficult level
            level = self._get_obj_level(obj)
            if level > 3:
                continue
            ix += 1
            # 0-based coordinates
            alpha = float(obj[3])
            x1 = float(obj[4])
            y1 = float(obj[5])
            x2 = float(obj[6])
            y2 = float(obj[7])
            alphas[ix] = alpha
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        alphas.resize(ix+1)
        boxes.resize(ix+1, 4)
        gt_classes.resize(ix+1)
        overlaps.resize(ix+1, self.num_classes)
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'alphas' :alphas,
                'flipped' : False}


    def _load_dontcare_annotation(self, index):
        """
        Load "DontCare" bounding boxes info from XML file in the KITTI
        format.
        """
        filename = os.path.join(self._data_path, 'label_2', index + '.txt')
        # print 'Loading: {}'.format(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        num_objs = len(lines)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)

        # Load object bounding boxes into a data frame.
        ix = -1
        for line in lines:
            obj = line.strip().split(' ')
            if obj[0].lower in self._classes and self._get_obj_level(obj) <= 3:
                continue
            if self._subset == 'car' and obj[0].lower not in {'dontcare', 'van'}:
                continue
            if self._subset == 'ped_cyc' and obj[0].lower not in {'dontcare', 'person_sitting'}:
                continue
            ix += 1
            # 0-based coordinates
            x1 = float(obj[4])
            y1 = float(obj[5])
            x2 = float(obj[6])
            y2 = float(obj[7])
            boxes[ix, :] = [x1, y1, x2, y2]

        boxes.resize(ix+1, 4)

        return {'boxes' : boxes}

    def _get_obj_level(self, obj):
        height = float(obj[7]) - float(obj[5]) + 1
        trucation = float(obj[1])
        occlusion = float(obj[2])
        if height >= 40 and trucation <= 0.15 and occlusion <= 0:
            return 1
        elif height >= 25 and trucation <= 0.3 and occlusion <= 1:
            return 2
        elif height >= 25 and trucation <= 0.5 and occlusion <= 2:
            return 3
        else:
            return 4

    def _write_kitti_results_file(self, all_boxes, all_alphas):
        use_salt = self.config['use_salt']
        comp_id = ''
        if use_salt:
            comp_id += '{}'.format(os.getpid())

        path = os.path.join(datasets.ROOT_DIR, 'kitti/results', 'kitti_' + self._subset + '_' + self._image_set + '_' + comp_id \
                                        + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'data')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(path, index + '.txt')
            with open(filename, 'wt') as f:
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    alphas = all_alphas[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the KITTI server expects 0-based indices
                    for k in xrange(dets.shape[0]):
                        alpha = alphas[k][0]
                        f.write('{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {:.3f}\n' \
                                   .format(cls.lower(), alpha, \
                                   dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, -1]))
        return path

    def _do_eval(self, path, output_dir='output'):
        cmd = os.path.join(datasets.ROOT_DIR, 'kitti/eval/cpp/evaluate_object {}'.format(os.path.dirname(path)))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, all_alphas, output_dir):
        path = self._write_kitti_results_file(all_boxes, all_alphas)
        if self._image_set != 'test':
            self._do_eval(path)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.kitti('train', 'car')
    res = d.roidb
    from IPython import embed; embed()
