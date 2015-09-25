# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from fast_rcnn.config import cfg
import datasets
import datasets.kitti
import os.path as osp
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

def _unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    return index, inv_index

def _validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()

class kitti(datasets.imdb):
    def __init__(self, image_set, subset, devkit_path=None):
        datasets.imdb.__init__(self, 'kitti_' + subset + '_' + image_set)
        self._image_set = image_set
        self._subset = subset  # car or ped_cyc
        self._data_path = self._get_default_data_path()
        if subset == 'car':
            self._classes = ('__background__', # always index 0
                             'car')
        elif subset == 'ped_cyc':
            self._classes = ('__background__', # always index 0
                             'pedestrian', 'cyclist')
        assert self.num_classes >= 2, 'ERROR: incorrect subset'

        print self._data_path
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self.set_proposal_method('selective_search')
        self.competition_mode(False)

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 5000}

        self._data_name =  ('test' if image_set == 'test' else 'trainval')
        self._gt_splits = ('train', 'val', 'trainval')
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
        return self._proposal_roidb('selective_search')

    def edge_boxes_roidb(self):
        return self._proposal_roidb('edge_boxes_AR')

    def mcg_roidb(self):
        return self._proposal_roidb('MCG')

    def _proposal_roidb(self, method):
        """
        Creates a roidb from pre-computed proposals of a particular methods.
        """
        top_k = self.config['top_k']
        cache_file = osp.join(self.cache_path, self.name +
                              '_{:s}_top{:d}'.format(method, top_k) +
                              '_roidb.pkl')

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{:s} {:s} roidb loaded from {:s}'.format(self.name, method,
                                                            cache_file)
            return roidb

        if self._image_set in self._gt_splits:
            gt_roidb = self.gt_roidb()
            method_roidb = self._load_proposals(method, gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, method_roidb)
        else:
            roidb = self._load_proposals(method, None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote {:s} roidb to {:s}'.format(method, cache_file)
        return roidb

    def _load_proposals(self, method, gt_roidb):
        """
        Load pre-computed proposals in the format provided by Jan Hosang:
        http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-
          computing/research/object-recognition-and-scene-understanding/how-
          good-are-detection-proposals-really/
        """
        box_list = []
        top_k = self.config['top_k']
        valid_methods = ['MCG', 'selective_search', 'edge_boxes_70',
                         '3DOP', '3DOP-SL', '3DOP-LS', '3DOP-LL', '3DOP-CA']
        assert method in valid_methods

        print 'Loading {} boxes'.format(method)
        for i, index in enumerate(self._image_index):
            if i % 1000 == 0:
                print '{:d} / {:d}'.format(i + 1, len(self._image_index))

            if method in ['3DOP', '3DOP-SL', '3DOP-LS', '3DOP-LL']:
                boxes = np.zeros((0,4), dtype=np.uint16)
                for ci, c in enumerate(self._classes):
                    if c == '__background__':
                        continue
                    box_file = osp.join(self.cache_path, '..', 'proposals',
                                        method, c, 'mat',
                                        self._get_box_file(index))
                    raw_data = sio.loadmat(box_file)['boxes']
                    this_boxes = (raw_data[:top_k, :] - 1).astype(np.uint16)
                    boxes = np.vstack((boxes, this_boxes))
            else:
                box_file = osp.join(self.cache_path, '..', 'proposals',
                                method, 'mat',
                                self._get_box_file(index))
                raw_data = sio.loadmat(box_file)['boxes']
                boxes = (raw_data[:top_k, :] - 1).astype(np.uint16)

            keep, inv_keep = _unique_boxes(boxes)
            boxes = boxes[keep, :]
            box_list.append(boxes)

#             im_ann = self._COCO.loadImgs(index)[0]
            # width = im_ann['width']
            # height = im_ann['height']
#             _validate_boxes(boxes, width=width, height=height)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _get_box_file(self, index):
        # first 14 chars / first 22 chars / all chars + .mat
        # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
        file_name = (index + '.mat')
        return osp.join(self._data_name, index[:4], file_name)

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
