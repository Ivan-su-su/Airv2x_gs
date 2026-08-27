# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Yuheng Wu <yuhengwu@kaist.ac.kr>, Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)

        return loss

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None
    ):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert (
                weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            )
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLossMultiClass(nn.Module):
    def __init__(self, args):
        super(PointPillarLossMultiClass, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args["cls_weight"]
        self.reg_coe = args["reg"]
        self.obj_weight = args.get("obj_weight", 1.0)
        self.iou_weight = args.get("iou_weight", 0.0)  # 0.0 means disabled
        self.recall_weight = args.get("recall_weight", 0.0)  # 0.0 means disabled, for encouraging more detections
        self.flow_weight = args["flow_weight"] if "flow_weight" in args else 1.0
        self.loss_dict = {}
        self.use_dir = False
        # self.obj_gamma = args["obj_gamma"]
        # self.obj_alpha = args["obj_alpha"]
        # self.neg_weight = 0.2

        self.cls_num = args["num_class"]

    def forward(self, output_dict, target_dict, prefix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict

        cls_label -> one_hot label
        """
        rm = output_dict["rm{}".format(prefix)]  # [B, #anchor*7, 50, 176]
        psm = output_dict["psm{}".format(prefix)]  # [B, #anchor*#class, 50, 176]
        obj = output_dict["obj{}".format(prefix)]  # [B, #anchor, 50, 176]
        targets = target_dict["targets"]
        
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()  # N, C, H, W -> N, H, W, C
        obj_preds = obj.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A)
        box_cls_labels = target_dict["pos_equal_one"]  # [B, 50, 176, 2]
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        # added
        pos_mask = target_dict["pos_equal_one"]  # (B, H, W, A)
        neg_mask = target_dict["neg_equal_one"]  # (B, H, W, A)

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_labels = target_dict["class_ids"]  # [B, H, W, A]
        cls_targets = cls_labels
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape),
            self.cls_num,
            dtype=cls_preds.dtype,
            device=cls_targets.device,
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_labels = one_hot_targets.view(
            cls_targets.shape[0], cls_targets.shape[1], cls_targets.shape[2], -1
        )
        
        cls_loss_src = self.cls_loss_func(
            cls_preds, cls_labels, weights=cls_weights
        )  # [N, M]
        
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(
            box_preds_sin, reg_targets_sin, weights=reg_weights
        )

        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        # objectness loss using Focal Loss (reuse obj_loss_func)
        obj_preds_expanded = obj_preds.unsqueeze(-1)  # [B, H, W, A] -> [B, H, W, A, 1]
        pos_mask_expanded = pos_mask.unsqueeze(-1).float()  # [B, H, W, A] -> [B, H, W, A, 1]
        obj_loss_src = self.obj_loss_func(
            obj_preds_expanded,  # input: logits
            pos_mask_expanded,   # target: 0/1
            torch.ones_like(pos_mask_expanded)  # weights: all ones
        )
        # Use mean (same as original implementation)
        obj_loss = obj_loss_src.mean()
        obj_loss_weighted = obj_loss * self.obj_weight

        total_loss = reg_loss + conf_loss + obj_loss_weighted

        # Recall loss (optional, only if recall_weight > 0)
        # Use BCE loss on positive samples ONLY to encourage higher confidence predictions
        # This helps address the issue of too few detections by penalizing low confidence on GT positives
        recall_loss_weighted = 0.0
        if self.recall_weight > 0 and positives.sum() > 0:
            # Flatten obj_preds and pos_mask
            obj_preds_flat = obj_preds.view(psm.shape[0], -1)  # [B, H*W*A]
            pos_mask_flat = pos_mask.view(psm.shape[0], -1)  # [B, H*W*A]
            
            # Extract ONLY positive samples for loss calculation
            pos_indices = pos_mask_flat > 0  # [B, H*W*A]
            if pos_indices.sum() > 0:
                # Get positive samples only
                pos_obj_preds = obj_preds_flat[pos_indices]  # [N_pos]
                pos_obj_sigmoid = torch.sigmoid(pos_obj_preds)  # [N_pos]
                
                # Use simple BCE loss (not Focal Loss) for positive samples
                # Target is 1.0 for all positive samples
                recall_loss = -torch.log(pos_obj_sigmoid + 1e-6).sum()
                recall_loss_weighted = recall_loss * self.recall_weight
                total_loss = total_loss + recall_loss_weighted

        # IoU loss (optional, only if iou_weight > 0)
        iou_loss_weighted = 0.0
        if self.iou_weight > 0 and positives.sum() > 0:
            # Get anchor_box from target_dict or output_dict
            anchor_box = target_dict.get("anchor_box", None)
            if anchor_box is None:
                anchor_box = output_dict.get("anchor_box", None)
            if anchor_box is not None:
                # Use positives mask (already flattened to [B, H*W*A])
                pred_boxes = self._decode_delta_to_boxes(rm, anchor_box, positives)
                gt_boxes = self._decode_delta_to_boxes(targets, anchor_box, positives)
                
                if pred_boxes.shape[0] > 0 and gt_boxes.shape[0] > 0 and pred_boxes.shape[0] == gt_boxes.shape[0]:
                    # Calculate IoU - use mean (same as MambaFusion: sum then divide by num_pos)
                    from opencood.utils.iou3d_nms import iou3d_nms_utils
                    iou = iou3d_nms_utils.paired_boxes_iou3d_gpu(pred_boxes.float(), gt_boxes.float())
                    num_pos = pred_boxes.shape[0]
                    iou_loss = (1.0 - iou).sum() / max(num_pos, 1)
                    iou_loss_weighted = iou_loss * self.iou_weight
                    total_loss = total_loss + iou_loss_weighted

        loss_dict_update = {
                "total_loss{}".format(prefix): total_loss.item(),
                "reg_loss{}".format(prefix): reg_loss.item(),
                "conf_loss{}".format(prefix): conf_loss.item(),
            "obj_loss{}".format(prefix): obj_loss_weighted.item(),
            }
        # Always record recall_loss and iou_loss if weights are set, even if 0
        if self.recall_weight > 0:
            if isinstance(recall_loss_weighted, torch.Tensor):
                loss_dict_update["recall_loss{}".format(prefix)] = recall_loss_weighted.item()
            else:
                loss_dict_update["recall_loss{}".format(prefix)] = float(recall_loss_weighted)
        if self.iou_weight > 0:
            if isinstance(iou_loss_weighted, torch.Tensor):
                loss_dict_update["iou_loss{}".format(prefix)] = iou_loss_weighted.item()
            else:
                loss_dict_update["iou_loss{}".format(prefix)] = float(iou_loss_weighted)
        self.loss_dict.update(loss_dict_update)

        return total_loss

    def cls_loss_func(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        B, H, W, AC = input.shape
        C = self.cls_num
        A = AC // C

        input = input.view(B, H, W, A, C)        # [B, H, W, A, C]
        target = target.view(B, H, W, A, C)      # [B, H, W, A, C]
        weights = weights.view(B, H, W, A, 1)    # [B, H, W, A, 1]

        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss

        # Apply weights per anchor
        weighted_loss = loss * weights  # shape [B, H, W, A, C]

        return weighted_loss.sum() / B
    
    def obj_loss_func(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or (
            weights.shape.__len__() == 1 and target.shape.__len__() == 2
        ):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0)
            - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def _decode_delta_to_boxes(self, deltas, anchor_box, pos_mask):
        """
        Decode delta (relative to anchor) to absolute boxes.
        
        Args:
            deltas: [B, H*W*A, 7] - delta values
            anchor_box: [H, W, A, 7] or [H*W*A, 7] - anchor boxes
            pos_mask: [B, H*W*A] - positive mask (bool)
            
        Returns:
            boxes: [N_pos, 7] - decoded boxes for positive samples
        """
        B, N, _ = deltas.shape
        device = deltas.device
        
        # Reshape anchor_box if needed
        if anchor_box.dim() == 4:  # [H, W, A, 7]
            anchor_box = anchor_box.view(-1, 7)  # [H*W*A, 7]
        elif anchor_box.dim() == 3:  # [H, W, A] -> should not happen
            anchor_box = anchor_box.view(-1, 7)
        
        anchor_box = anchor_box.to(device).float()
        
        # Get positive samples (flatten batch dimension)
        pos_indices_flat = pos_mask.view(-1)  # [B*H*W*A]
        if pos_indices_flat.sum() == 0:
            return torch.empty((0, 7), device=device)
        
        # Flatten deltas and get positive samples
        deltas_flat = deltas.view(-1, 7)  # [B*H*W*A, 7]
        pos_deltas = deltas_flat[pos_indices_flat]  # [N_pos, 7]
        pos_anchors = anchor_box[pos_indices_flat]  # [N_pos, 7]
        
        # Decode boxes (same logic as delta_to_boxes3d)
        boxes = torch.zeros_like(pos_deltas)
        
        # Calculate anchor diagonal for x, y normalization
        anchors_d = torch.sqrt(pos_anchors[:, 4] ** 2 + pos_anchors[:, 5] ** 2)  # [N_pos]
        
        # Decode x, y (normalized by anchor diagonal)
        boxes[:, 0] = pos_deltas[:, 0] * anchors_d + pos_anchors[:, 0]
        boxes[:, 1] = pos_deltas[:, 1] * anchors_d + pos_anchors[:, 1]
        
        # Decode z (normalized by anchor height)
        boxes[:, 2] = pos_deltas[:, 2] * pos_anchors[:, 3] + pos_anchors[:, 2]
        
        # Decode h, w, l (exp scale)
        boxes[:, 3] = torch.exp(pos_deltas[:, 3]) * pos_anchors[:, 3]  # h
        boxes[:, 4] = torch.exp(pos_deltas[:, 4]) * pos_anchors[:, 4]  # w
        boxes[:, 5] = torch.exp(pos_deltas[:, 5]) * pos_anchors[:, 5]  # l
        
        # Decode yaw (additive)
        boxes[:, 6] = pos_deltas[:, 6] + pos_anchors[:, 6]
        
        return boxes

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(
            boxes2[..., dim : dim + 1]
        )
        rad_tg_encoding = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(
            boxes2[..., dim : dim + 1]
        )

        boxes1 = torch.cat(
            [boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1 :]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1 :]], dim=-1
        )
        return boxes1, boxes2

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)
        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = [v for k, v in self.loss_dict.items() if "total_loss" in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0]

        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(
            epoch, batch_id + 1, batch_len, total_loss
        )
        for k, v in self.loss_dict.items():
            print_msg += "{}: {:.2f} | ".format(
                k.replace("_loss", "").replace("_single", ""), v
            )

        if not writer is None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v, epoch * batch_len + batch_id)
                
        return print_msg

    def _forward(self, output_dict, target_dict, prefix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict

        cls_label -> one_hot label
        """
        rm = output_dict["rm{}".format(prefix)]  # [B, 14, 50, 176]
        psm = output_dict["psm{}".format(prefix)]  # [B, 2, 50, 176]
        targets = target_dict["targets"]

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()  # N, C, H, W -> N, H, W, C

        box_cls_labels = target_dict["pos_equal_one"]  # [B, 50, 176, 2]
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape),
            2,
            dtype=cls_preds.dtype,
            device=cls_targets.device,
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:] # here remove the BG class

        cls_loss_src = self.cls_loss_func(
            cls_preds, one_hot_targets, weights=cls_weights
        )  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(
            box_preds_sin, reg_targets_sin, weights=reg_weights
        )

        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        total_loss = conf_loss
        # total_loss = reg_loss + conf_loss

        # print('psm: ', psm.shape, cls_preds.shape)
        # print('rm: ', rm.shape, box_preds_sin.shape)

        self.loss_dict.update(
            {
                "total_loss{}".format(prefix): total_loss,
                #'reg_loss{}'.format(prefix): reg_loss,
                "conf_loss{}".format(prefix): conf_loss,
            }
        )

        return total_loss

    def _logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = [v.item() for k, v in self.loss_dict.items() if "total_loss" in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0]
        # reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict["conf_loss"]

        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(
            epoch, batch_id + 1, batch_len, total_loss
        )
        for k, v in self.loss_dict.items():
            print_msg += "{}: {:.2f} | ".format(
                k.replace("_loss", "").replace("_single", ""), v.item()
            )

        # print_msg = ("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
        #             " || Loc Loss: %.4f" % (
        #                 epoch, batch_id + 1, batch_len,
        #                 total_loss.item(), conf_loss.item(), reg_loss.item()))

        if self.use_dir:
            dir_loss = self.loss_dict["dir_loss"]
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        # print(print_msg)

        if not writer is None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v.item(), epoch * batch_len + batch_id)
            # writer.add_scalar('Regression_loss', reg_loss.item(),
            #                 epoch*batch_len + batch_id)
            # writer.add_scalar('Confidence_loss', conf_loss.item(),
            #                 epoch*batch_len + batch_id)

            if self.use_dir:
                writer.add_scalar(
                    "dir_loss", dir_loss.item(), epoch * batch_len + batch_id
                )
        
        return print_msg