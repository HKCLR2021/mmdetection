# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class BMaskRoIHead(StandardRoIHead):

    def __init__(self, bmask_roi_extractor, **kwargs):
        assert bmask_roi_extractor is not None
        
        super(BMaskRoIHead, self).__init__(**kwargs)
        self.bmask_roi_extractor = build_roi_extractor(bmask_roi_extractor)

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'], mask_results['boundary_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)

        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            b_mask_feats = self.bmask_roi_extractor(
                x[:self.bmask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                b_mask_feats = self.shared_head(b_mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]
            b_mask_feats = bbox_feats[pos_inds]

        mask_pred, boundary_pred = self.mask_head(mask_feats, b_mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats, boundary_pred=boundary_pred)
        return mask_results

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Obtain mask prediction without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            num_classes = self.mask_head.num_classes
            segm_results = [[[] for _ in range(num_classes)]
                            for _ in range(num_imgs)]
            boundary_results = [[[] for _ in range(num_classes)]
                            for _ in range(num_imgs)]
            mask_scores = [[[] for _ in range(num_classes)]
                           for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i]
                for i in range(num_imgs)
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            concat_det_labels = torch.cat(det_labels)
            
            mask_feats = mask_results['mask_feats']
            mask_pred = mask_results['mask_pred']
            boundary_pred = mask_results['boundary_pred']
            
            # split batch mask prediction back to each image
            num_bboxes_per_img = tuple(len(_bbox) for _bbox in _bboxes)
            mask_preds = mask_pred.split(num_bboxes_per_img, 0)
            boundary_preds = boundary_pred.split(num_bboxes_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            boundary_results = []
            mask_scores = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                    mask_scores.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    print(det_labels[i])
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    boundary_result = self.mask_head.get_seg_masks(
                        boundary_preds[i], _bboxes[i], torch.zeros(det_labels[i].shape).cuda().long(),
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
                    boundary_results.append(boundary_result)       
        return segm_results