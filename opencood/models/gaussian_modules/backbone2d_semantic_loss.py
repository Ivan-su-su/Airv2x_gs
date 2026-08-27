# -*- coding: utf-8 -*-
"""
Semantic Segmentation Loss for Backbone2D
用于预训练 Backbone2D 语义分类头的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone2DSemanticLoss(nn.Module):
    """
    语义分类损失函数类（用于Backbone2D）
    
    用于预训练 Backbone2D 的 image_backbone 和 detection_head
    使用 CrossEntropyLoss 进行多类别语义分割
    """
    
    def __init__(self, args):
        """
        初始化损失函数
        
        Args:
            args (dict): 配置参数字典，包含：
                - num_classes (int): 类别数量（包含背景类）
                - loss_weight (float): 损失权重，默认1.0
                - class_weights (list, optional): 类别权重列表，用于处理类别不平衡
                - ignore_index (int, optional): 忽略的类别索引，默认-1
        """
        super(Backbone2DSemanticLoss, self).__init__()
        
        self.num_classes = args.get('num_classes', 7)
        self.loss_weight = args.get('loss_weight', 1.0)
        self.ignore_index = args.get('ignore_index', -1)
        
        # 类别权重（可选，用于处理类别不平衡）
        class_weights = args.get('class_weights', None)
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            else:
                raise ValueError("class_weights must be a list or None")
        else:
            class_weights = None
        
        # 创建 CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        self.loss_dict = {}
        
        print(f"[Backbone2DSemanticLoss] 初始化完成:")
        print(f"  - Num Classes: {self.num_classes}")
        print(f"  - Loss Weight: {self.loss_weight}")
        print(f"  - Ignore Index: {self.ignore_index}")
        if class_weights is not None:
            print(f"  - Class Weights: {class_weights}")
    
    def forward(self, output_dict, target_dict, prefix=""):
        """
        计算语义分类损失
        
        Args:
            output_dict (dict): 模型输出字典，应包含：
                - 'semantic_logits': [B*N, M, H_feat, W_feat] logits
            target_dict (dict): 目标字典，应包含：
                - 'semantic_targets': [B, N, H_feat, W_feat] long tensor，类别标签
            prefix (str): 前缀，用于多任务场景
        
        Returns:
            total_loss (torch.Tensor): 总损失值
        """
        # 获取预测logits
        if 'semantic_logits' in output_dict:
            semantic_logits = output_dict['semantic_logits']
        else:
            raise KeyError("output_dict must contain 'semantic_logits'")
        
        # 获取目标标签
        if 'semantic_targets' in target_dict:
            semantic_targets = target_dict['semantic_targets']
        else:
            raise KeyError("target_dict must contain 'semantic_targets'")
        
        # 确保数据类型正确
        # semantic_logits: [B*N, M, H_feat, W_feat]
        # semantic_targets: [B, N, H_feat, W_feat]
        if semantic_logits.dim() == 4:
            B_N, M, H_feat, W_feat = semantic_logits.shape
            # 重塑为 [B*N*H_feat*W_feat, M] 和 [B*N*H_feat*W_feat]
            semantic_logits = semantic_logits.permute(0, 2, 3, 1).contiguous()  # [B*N, H_feat, W_feat, M]
            semantic_logits = semantic_logits.view(-1, M)  # [B*N*H_feat*W_feat, M]
        else:
            raise ValueError(
                f"semantic_logits should be 4D tensor [B*N, M, H_feat, W_feat], "
                f"got shape {semantic_logits.shape}"
            )
        
        if semantic_targets.dim() == 4:
            B, N, H_feat_t, W_feat_t = semantic_targets.shape
            # 确保尺寸匹配
            if H_feat_t != H_feat or W_feat_t != W_feat:
                raise ValueError(
                    f"semantic_targets spatial size ({H_feat_t}, {W_feat_t}) "
                    f"does not match semantic_logits spatial size ({H_feat}, {W_feat})"
                )
            semantic_targets = semantic_targets.view(-1)  # [B*N*H_feat*W_feat]
            semantic_targets = semantic_targets.long()
        else:
            raise ValueError(
                f"semantic_targets should be 4D tensor [B, N, H_feat, W_feat], "
                f"got shape {semantic_targets.shape}"
            )
        
        # 确保在同一设备上
        if semantic_targets.device != semantic_logits.device:
            semantic_targets = semantic_targets.to(semantic_logits.device)
        
        # 检查形状匹配
        if semantic_logits.shape[0] != semantic_targets.shape[0]:
            raise ValueError(
                f"semantic_logits and semantic_targets must have same number of pixels, "
                f"got {semantic_logits.shape[0]} vs {semantic_targets.shape[0]}"
            )
        
        # 将无效类别索引设为 ignore_index，避免 CUDA 断言 t >= 0 && t < n_classes 失败
        # 有效类别为 [0, num_classes-1]，超出范围或负值均视为无效
        invalid_mask = (semantic_targets < 0) | (semantic_targets >= self.num_classes)
        if invalid_mask.any():
            num_invalid = invalid_mask.sum().item()
            if num_invalid > 0:
                print(
                    f"[Warning] semantic_targets 含 {num_invalid} 个无效类别索引 "
                    f"(有效范围 0~{self.num_classes - 1}, 实际 max={semantic_targets.max().item()}), 已设为 ignore_index={self.ignore_index}"
                )
            semantic_targets = semantic_targets.clone()
            semantic_targets[invalid_mask] = self.ignore_index
        
        # 计算损失
        # 注意：必须使用 logits（未经过 softmax），而不是 probs（已 softmax）
        # 因为 CrossEntropyLoss 内部会先做 log_softmax，如果传入已 softmax 的值会导致数值不稳定
        semantic_loss = self.criterion(semantic_logits, semantic_targets)
        total_loss = semantic_loss * self.loss_weight
        
        # 更新损失字典
        self.loss_dict.update({
            f'total_loss{prefix}': total_loss.item(),
            f'semantic_loss{prefix}': semantic_loss.item(),
        })
        
        return total_loss
    
    def logging(self, epoch, batch_id, batch_len, writer=None, pbar=None):
        """
        打印和记录损失信息
        
        Args:
            epoch (int): 当前epoch
            batch_id (int): 当前batch索引
            batch_len (int): 总batch数量
            writer (SummaryWriter, optional): TensorBoard writer
            pbar (tqdm, optional): 进度条对象
        """
        total_loss = self.loss_dict.get('total_loss', 0.0)
        semantic_loss = self.loss_dict.get('semantic_loss', 0.0)
        
        msg = (
            f"[epoch {epoch}][{batch_id + 1}/{batch_len}], "
            f"|| Loss: {total_loss:.4f} || "
            f"Semantic Loss: {semantic_loss:.4f}"
        )
        
        if pbar is not None:
            pbar.set_description(msg)
            return msg
        else:
            print(msg)
            return msg
        
        # 记录到TensorBoard
        if writer is not None:
            global_step = epoch * batch_len + batch_id
            writer.add_scalar('Semantic_Loss', semantic_loss, global_step)
            writer.add_scalar('Total_Loss', total_loss, global_step)

