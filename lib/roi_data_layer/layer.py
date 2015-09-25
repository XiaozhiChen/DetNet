# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {
            'data': 0,
            'rois': 1,
            'labels': 2}

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(1, 3, 100, 100)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[1].reshape(1, 5)

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[2].reshape(1)

        if cfg.TRAIN.BBOX_REG:
            self._name_to_top_map['bbox_targets'] = 3
            self._name_to_top_map['bbox_loss_weights'] = 4

            # bbox_targets blob: R bounding-box regression targets with 4
            # targets per class
            top[3].reshape(1, self._num_classes * 4)

            # bbox_loss_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top[4].reshape(1, self._num_classes * 4)

        if cfg.TRAIN.ORT_REG:
            self._name_to_top_map['ort_targets'] = 5
            self._name_to_top_map['ort_loss_weights'] = 6

            # orientation_targets blob: R bounding-box regression targets with 4
            # targets per class
            top[5].reshape(1, self._num_classes * 1)

            # orientation_loss_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top[6].reshape(1, self._num_classes * 1)


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)

class MultiRoIDataLayer(caffe.Layer):
    """Multi-ROI layer"""

    # zooming of rois
    def zoom(self, top_data, rho):
        assert rho > 0
        dx = (rho - 1) *  (top_data[:,3] - top_data[:,1]) / 2
        dy = (rho - 1) *  (top_data[:,4] - top_data[:,2]) / 2
        top_data[:,1] = top_data[:,1] - dx
        top_data[:,2] = top_data[:,2] - dy
        top_data[:,3] = top_data[:,3] + dx
        top_data[:,4] = top_data[:,4] + dy
        return top_data

    def context(self, top_data, rho):
        return self.zoom(top_data, rho)

    def central(self, top_data, rho):
        return self.zoom(top_data, rho)

    # crop left region
    def left(self, top_data, rho):
        assert rho > 0
        top_data[:,3] = rho * top_data[:,3] + (1 - rho) * top_data[:,1]
        return top_data

    # crop right region
    def right(self, top_data, rho):
        assert rho > 0
        top_data[:,1] = rho * top_data[:,1] + (1 - rho) * top_data[:,3]
        return top_data

    # crop upper region
    def up(self, top_data, rho):
        assert rho > 0
        top_data[:,4] = rho * top_data[:,4] + (1 - rho) * top_data[:,2]
        return top_data

    # crop bottom region
    def bottom(self, top_data, rho):
        assert rho > 0
        top_data[:,2] = rho * top_data[:,2] + (1 - rho) * top_data[:,4]
        return top_data

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        print self.param_str_
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        self._top_params = layer_params
        self._name_to_top_map = {}
        id = -1
        for key in layer_params:
            id += 1
            self._name_to_top_map.update({key:id})
            top[id].reshape(1, 5)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        for key in self._name_to_top_map:
            top_ind = self._name_to_top_map[key]
            # Reshape top blobs
            top[top_ind].reshape(*bottom[0].data.shape)
            # Copy data into net's input blobs
            top[top_ind].data[...] = bottom[0].data
            method = getattr(self, key)
            top[top_ind].data[...] = method(top[top_ind].data, self._top_params[key])

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


