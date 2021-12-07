import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector, losses
import torch.nn.functional as F
from torch.utils import data

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid


@DETECTORS.register_module()
class SoftTeacherBase(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None, memory_k=65536, ctr1_T=0.2, ctr2_T=0.2,
     ctr1_lam_sup=0.1, ctr1_lam_unsup=0.1, ctr2_lam_sup=0.1, ctr2_lam_unsup=0.1):
        super(SoftTeacherBase, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
        
        self.memory_k = memory_k
        self.ctr1_T = ctr1_T
        self.ctr2_T = ctr2_T
        self.ctr1_lam_sup = ctr1_lam_sup
        self.ctr1_lam_unsup = ctr1_lam_unsup
        self.ctr2_lam_sup = ctr2_lam_sup
        self.ctr2_lam_unsup = ctr2_lam_unsup
        self.projector_dim = model.projector_dim
        self.register_buffer("queue_vector", torch.randn(memory_k, model.projector_dim)) 
        self.queue_vector = F.normalize(self.queue_vector, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_ctr_loss = self.ctr_loss(data_groups["ctr_anchor_sup"], data_groups["ctr_dict_sup"])
            sup_ctr_loss['ctr1'] = sup_ctr_loss['ctr1'] * self.ctr1_lam_sup
            sup_ctr_loss['ctr2'] = sup_ctr_loss['ctr2'] * self.ctr2_lam_sup
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            sup_ctr_loss = {"sup_" + k: v for k, v in sup_ctr_loss.items()}
            loss.update(**sup_loss)
            loss.update(**sup_ctr_loss)
            
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"],
                    data_groups["ctr_anchor_unsup"], data_groups["ctr_dict_unsup"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data, anchor_data, ctr_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]

        anchornames = [meta["filename"] for meta in anchor_data["img_metas"]]
        ctrnames = [meta["filename"] for meta in ctr_data["img_metas"]]
        ctridx = [anchornames.index(name) for name in ctrnames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
            ctr_info = self.extract_teacher_info(
                ctr_data["img"][
                    torch.Tensor(ctridx).to(ctr_data["img"].device).long()
                ],
                [ctr_data["img_metas"][idx] for idx in ctridx],
                [ctr_data["proposals"][idx] for idx in ctridx]
                if ("proposals" in ctr_data)
                and (ctr_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)

        losses = dict()
        losses.update(self.compute_pseudo_label_loss(student_info, teacher_info))

        anchor_transform_matrix = [
            torch.from_numpy(meta["transform_matrix"]).float().to(anchor_data['img'].device)
            for meta in anchor_data['img_metas']
        ]

        ctr_box, ctr_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in ctr_info["det_bboxes"]],
            ctr_info["det_labels"],
            [bbox[:, 4] for bbox in ctr_info["det_bboxes"]],
            thr=self.train_cfg.cls_pseudo_threshold,
        )

        M = self._get_trans_mat(
            ctr_info["transform_matrix"], anchor_transform_matrix
        )
        anchor_box = self._transform_bbox(
            ctr_box,
            M,
            [meta["img_shape"] for meta in ctr_data["img_metas"]],
        )

        anchor_data['gt_bboxes'] = anchor_box
        anchor_data['gt_labels'] = ctr_labels
        ctr_data['gt_bboxes'] = ctr_box
        ctr_data['gt_labels'] = ctr_labels
        ctr_losses = self.ctr_loss(anchor_data=anchor_data, dict_data=ctr_data)
        ctr_losses['ctr1'] = ctr_losses['ctr1'] * self.ctr1_lam_unsup
        ctr_losses['ctr2'] = ctr_losses['ctr2'] * self.ctr2_lam_unsup
        losses.update(**ctr_losses)

        return losses

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        # loss.update(
        #     self.unsup_rcnn_cls_loss(
        #         student_info["backbone_feature"],
        #         student_info["img_metas"],
        #         proposals,
        #         pseudo_bboxes,
        #         pseudo_labels,
        #         teacher_info["transform_matrix"],
        #         student_info["transform_matrix"],
        #         teacher_info["img_metas"],
        #         teacher_info["backbone_feature"],
        #         student_info=student_info,
        #     )
        # )
        # loss.update(
        #     self.unsup_rcnn_reg_loss(
        #         student_info["backbone_feature"],
        #         student_info["img_metas"],
        #         proposals,
        #         pseudo_bboxes,
        #         pseudo_labels,
        #         student_info=student_info,
        #     )
        # )
        loss.update(
            self.unsup_rcnn_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
        )
        return loss

    @torch.no_grad()
    def concat_all_gather(self, features):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        device = features.device
        local_batch = torch.tensor(features.size(0)).to(device)
        batch_size_gather = [torch.ones((1)).to(device)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(batch_size_gather, local_batch.float(), async_op=False)
        
        batch_size_gather = [int(bs.item()) for bs in batch_size_gather]

        max_batch = max(batch_size_gather)
        size = (max_batch, features.size(1))
        temp_features = torch.zeros(max_batch - local_batch, features.size(1)).to(device)
        features = torch.cat([features, temp_features])

        # size = (int(tensors_gather[0].item()), features.size(1))
        # (int(tensors_gather[i].item()), features.size(1))
        features_gather = [torch.ones(size).to(device)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(features_gather, features, async_op=False)

        features_gather = [f[:bs, :] for bs, f in zip(batch_size_gather, features_gather)]

        features = torch.cat(features_gather, dim=0)

        return features

    @torch.no_grad()
    def _dequeue_and_enqueue(self, features):
        # gather keys before updating queue
        
        features = self.concat_all_gather(features)

        batch_size = features.size(0)

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size >= self.memory_k:
            redundant = ptr + batch_size - self.memory_k
            self.queue_vector[ptr:self.memory_k, :] = features.view(batch_size, -1)[:batch_size - redundant]
            self.queue_vector[:redundant, :] = features.view(batch_size, -1)[batch_size - redundant:]
        else:
            self.queue_vector[ptr:ptr + batch_size, :] = features.view(batch_size, -1)
        ptr = (ptr + batch_size) % self.memory_k  # move pointer

        self.queue_ptr[0] = ptr

    def ctr_loss(self, anchor_data, dict_data):
        losses = {}
        anchor_info, dict_info = self.extract_ctr_info(anchor_data, dict_data)

        # for ctr 1
        anchor_sample_res = anchor_info['sampling_results']
        dict_sample_res = dict_info['sampling_results']

        device = anchor_data['img'].device
        batch = anchor_sample_res[0].bboxes.size(0)

        pos_inds_anchor = torch.zeros([0]).to(device).long()
        pos_gt_map_anchor = torch.zeros([0]).to(device).long()
        pos_gt_map_ctr = torch.zeros([0]).to(device).long()
        pos_labels_anchor = torch.zeros([0]).to(device).long()
        pos_labels_ctr = torch.zeros([0]).to(device).long()
        for i, (res_anchor, res_ctr) in enumerate(zip(anchor_sample_res, dict_sample_res)):
            pos_inds_anchor = torch.cat([pos_inds_anchor, (torch.arange(0, res_anchor.pos_inds.size(0)).to(device).long() + (i * batch)).view(-1)])
            pos_gt_map_anchor = torch.cat([pos_gt_map_anchor, (res_anchor.pos_assigned_gt_inds + (i * batch)).view(-1)])
            pos_gt_map_ctr = torch.cat([pos_gt_map_ctr, (res_ctr.pos_assigned_gt_inds + (i * batch)).view(-1)])
            pos_labels_anchor = torch.cat([pos_labels_anchor, res_anchor.pos_gt_labels])
            pos_labels_ctr = torch.cat([pos_labels_ctr, res_ctr.pos_gt_labels])


        # student_proposal_rois = bbox2roi([res.pos_bboxes for res in anchor_sample_res])
        # student_proposals = self.student.roi_head.bbox_roi_extractor(anchor_info['backbone_feature'][:self.student.roi_head.bbox_roi_extractor.num_inputs], student_proposal_rois)
        # teacher_proposal_rois = bbox2roi([res.pos_bboxes for res in dict_sample_res])
        # teacher_proposals = self.teacher.roi_head.bbox_roi_extractor(dict_info['backbone_feature'][:self.teacher.roi_head.bbox_roi_extractor.num_inputs], teacher_proposal_rois)
        # if student_proposals.size(0) == 0 or teacher_proposals.size(0) == 0:
        #     losses['ctr1'] = 0
        # else:
        #     student_vec = self.student.projector(student_proposals.view(student_proposals.size(0), -1))
        #     student_vec = F.normalize(student_vec, dim=1)
        #     teacher_vec = self.teacher.projector(teacher_proposals.view(teacher_proposals.size(0), -1))
        #     teacher_vec = F.normalize(teacher_vec, dim=1)

        #     ctr1_logit = torch.zeros(0, self.dim).to(device)
        #     for i in range(pos_labels_anchor.size(0)):
        #         pos_inds = pos_gt_map_ctr == pos_gt_map_anchor[i]
        #         pos_logits = teacher_vec[pos_inds, :]
        #         if pos_logits.size(0) == 0:
        #             pos_inds = pos_gt_map_anchor == pos_gt_map_anchor[i]
        #             pos_logits = student_vec[pos_inds, :]
        #         rand_index = torch.randint(low=0, high=pos_proposal.size(0), size=(1,))
        #         ctr1_logit = torch.cat([ctr1_logit, pos_logits])

        anchor_proposal = bbox2roi([res.pos_bboxes for res in anchor_sample_res])
        dict_proposal = bbox2roi([res.pos_bboxes for res in dict_sample_res])
        ctr_proposal = torch.zeros([0, 5]).to(device)
        for gt_map in pos_gt_map_anchor:
            pos_inds = pos_gt_map_ctr == gt_map
            pos_proposal = dict_proposal[pos_inds]
            if pos_proposal.size(0) == 0:
                pos_inds = pos_gt_map_anchor == gt_map
                pos_proposal = anchor_proposal[pos_inds]
            rand_index = torch.randint(low=0, high=pos_proposal.size(0), size=(1,))
            ctr_proposal = torch.cat([ctr_proposal, pos_proposal[rand_index]], dim=0)

        losses = dict()
        ctr1_loss = self.ctr_loss_1(anchor_info['backbone_feature'], anchor_proposal, dict_info['backbone_feature'], ctr_proposal)
        losses.update(**ctr1_loss)
        ctr2_loss = self.ctr_loss_2(anchor_info['backbone_feature'], anchor_data['gt_bboxes'], anchor_data['gt_labels'])
        losses.update(**ctr2_loss)
        return losses
        

        

    def ctr_loss_1(self, student_feat, student_proposal, teacher_feat, teacher_proposal):
        losses = dict()
        device = student_feat[0].device
        student_proposal_rois = student_proposal
        student_proposals = self.student.roi_head.bbox_roi_extractor(student_feat[:self.student.roi_head.bbox_roi_extractor.num_inputs], student_proposal_rois)
        teacher_proposal_rois = teacher_proposal
        teacher_proposals = self.teacher.roi_head.bbox_roi_extractor(teacher_feat[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], teacher_proposal_rois)
        if student_proposals.size(0) == 0 or teacher_proposals.size(0) == 0:
            losses['ctr1'] = torch.zeros([1]).to(device)
            return losses 
        student_vec = self.student.projector(student_proposals.view(student_proposals.size(0), -1))
        student_vec = F.normalize(student_vec, dim=1)
        teacher_vec = self.teacher.projector(teacher_proposals.view(teacher_proposals.size(0), -1))
        teacher_vec = F.normalize(teacher_vec, dim=1)

        neg_logits = torch.einsum('nc,kc->nk', [student_vec, self.queue_vector.clone().detach()])
        pos_logits = torch.einsum('nc,nc->n', [student_vec, teacher_vec])
        logits = torch.cat([pos_logits[:, None], neg_logits], dim=1)
        logits /= self.ctr1_T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        losses['ctr1'] = F.cross_entropy(logits, labels)

        self._dequeue_and_enqueue(teacher_vec)

        return losses

    def ctr_loss_2(self, student_feat, pseudo_boxes, pseudo_labels):
        losses = dict()
        device = student_feat[0].device
        student_proposal_rois = bbox2roi([stup for stup in pseudo_boxes])
        student_proposals = self.student.roi_head.bbox_roi_extractor(student_feat[:self.student.roi_head.bbox_roi_extractor.num_inputs], student_proposal_rois)
        if student_proposals.size(0) == 0:
            losses['ctr2'] = torch.zeros([1]).to(device)
            return losses
        student_vec = self.student.projector(student_proposals.view(student_proposals.size(0), -1))
        student_vec = F.normalize(student_vec, dim=1)
        all_labels = torch.cat(pseudo_labels)

        teacher_vec = torch.zeros([0, self.projector_dim]).to(device)
        for label in all_labels:
            same_label_item = self.labeled_dataset.get_same_label_item(label)
            same_label_item = same_label_item[-1]
            # same_label_item = same_label_item[-1] if isinstance(same_label_item, (list)) else same_label_item
            while label not in same_label_item['gt_labels'].data.to(device):
                same_label_item = self.labeled_dataset.get_same_label_item(label)
            feat = self.teacher.extract_feat(same_label_item['img'].data.to(device)[None, :, :, :])
            teacher_proposal_rois = bbox2roi([same_label_item['gt_bboxes'].data[same_label_item['gt_labels'].data.to(device) == label].to(device)])
            rand_index = torch.randint(low=0, high=teacher_proposal_rois.size(0), size=(1,))
            teacher_proposal = self.teacher.roi_head.bbox_roi_extractor(feat[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], teacher_proposal_rois[rand_index])
            vec = self.teacher.projector(teacher_proposal.view(teacher_proposal.size(0), -1))
            teacher_vec = torch.cat([teacher_vec, vec], dim=0)
        teacher_vec = F.normalize(teacher_vec, dim=1)
        
        neg_logits = torch.einsum('nc,kc->nk', [student_vec, self.queue_vector.clone().detach()])
        pos_logits = torch.einsum('nc,nc->n', [student_vec, teacher_vec])
        logits = torch.cat([pos_logits[:, None], neg_logits], dim=1)
        logits /= self.ctr2_T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        losses = dict()
        losses['ctr2'] = F.cross_entropy(logits, labels)

        # self._dequeue_and_enqueue(teacher_vec)

        return losses

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        # M = self._get_trans_mat(student_transMat, teacher_transMat)
        # aligned_proposals = self._transform_bbox(
        #     selected_bboxes,
        #     M,
        #     [meta["img_shape"] for meta in teacher_img_metas],
        # )
        # TODO soft teacher?
        # with torch.no_grad():
        #     _, _scores = self.teacher.roi_head.simple_test_bboxes(
        #         teacher_feat,
        #         teacher_img_metas,
        #         aligned_proposals,
        #         None,
        #         rescale=False,
        #     )
        #     bg_score = torch.cat([_score[:, -1] for _score in _scores])
        #     assigned_label, _, _, _ = bbox_targets
        #     neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
        #     bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        # loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        # loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
        #     bbox_targets[1].size()[0], 1.0
        # )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 4].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}

    def unsup_rcnn_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        mode='student',
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        if mode == 'student':
            for i in range(num_imgs):
                assign_result = self.student.roi_head.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.student.roi_head.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                )
                sampling_results.append(sampling_result)
        else:
            for i in range(num_imgs):
                assign_result = self.teacher.roi_head.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.teacher.roi_head.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                )
                sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_ctr_info(self, strong_data, weak_data):
        strong_info = {}
        strong_info["img"] = strong_data['img']
        feat = self.student.extract_feat(strong_data['img'])
        strong_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            strong_info["rpn_out"] = list(rpn_out)
        strong_info["img_metas"] = strong_data['img_metas']
        # strong_info["proposals"] = strong_data['proposals']
        strong_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in strong_data['img_metas']
        ]
        proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
        proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=strong_data['img_metas'], cfg=proposal_cfg
            )

        strong_info['sampling_results'] = self.get_sampling_result(
            strong_data['img_metas'],
            proposal_list,
            strong_data['gt_bboxes'],
            strong_data['gt_labels'],
            mode='student'
        )
        strong_info["proposals"] = proposal_list

        weak_info = {}
        weak_info["img"] = weak_data['img']
        feat = self.teacher.extract_feat(weak_data['img'])
        weak_info["backbone_feature"] = feat
        if self.teacher.with_rpn:
            rpn_out = self.teacher.rpn_head(feat)
            weak_info["rpn_out"] = list(rpn_out)
        weak_info["img_metas"] = weak_data['img_metas']
        # weak_info["proposals"] = weak_data['proposals']
        weak_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in weak_data['img_metas']
        ]

        proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
        proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=weak_data['img_metas'], cfg=proposal_cfg
            )

        weak_info['sampling_results'] = self.get_sampling_result(
            weak_data['img_metas'],
            proposal_list,
            weak_data['gt_bboxes'],
            weak_data['gt_labels'],
            mode='teacher'
        )
        weak_info["proposals"] = proposal_list

        return strong_info, weak_info

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        if isinstance(self.train_cfg.contrastive_initial_score_thr, float):
            contrastive_thr = self.train_cfg.contrastive_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        # reg_unc = self.compute_uncertainty_with_aug(
        #     feat, img_metas, proposal_list, proposal_label_list
        # )
        # det_bboxes = [
        #     torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        # ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]
        #teacher 产生的box又过了一次roi head
        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
