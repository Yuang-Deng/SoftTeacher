# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import DETECTORS
from mmdet.models import TwoStageDetector
import torch


@DETECTORS.register_module()
class MMFasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 projector_dim=128,
                 init_cfg=None):
        super(MMFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.projector = torch.nn.Sequential(torch.nn.Linear(12544, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, projector_dim))
