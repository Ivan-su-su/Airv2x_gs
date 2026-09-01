# -*- coding: utf-8 -*-
"""
Semantic Segmentation Loss for Backbone3D
用于预训练 Backbone3D 语义分类头的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone3DSemanticLoss(nn.Module):
    """
    语义分类损失函数类
    
    用于预训练 Backbone3D 的 encoder 和 semantic_head
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
        super(Backbone3DSemanticLoss, self).__init__()
        
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
        
        print(f"[Backbone3DSemanticLoss] 初始化完成:")
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
                - 'semantic_logits' 或 'semantic_logits_dense': [N_voxel, num_classes] logits
            target_dict (dict): 目标字典，应包含：
                - 'semantic_targets': [N_voxel] long tensor，类别标签
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
        if semantic_logits.dim() == 2:
            # [N_voxel, num_classes]
            pass
        else:
            raise ValueError(
                f"semantic_logits should be 2D tensor [N_voxel, num_classes], "
                f"got shape {semantic_logits.shape}"
            )
        
        if semantic_targets.dim() == 1:
            # [N_voxel]
            semantic_targets = semantic_targets.long()
        else:
            raise ValueError(
                f"semantic_targets should be 1D tensor [N_voxel], "
                f"got shape {semantic_targets.shape}"
            )
        
        # 确保在同一设备上
        if semantic_targets.device != semantic_logits.device:
            semantic_targets = semantic_targets.to(semantic_logits.device)
        
        # 检查形状匹配
        if semantic_logits.shape[0] != semantic_targets.shape[0]:
            raise ValueError(
                f"semantic_logits and semantic_targets must have same first dimension, "
                f"got {semantic_logits.shape[0]} vs {semantic_targets.shape[0]}"
            )
        
        # 检查类别范围
        if semantic_targets.max() >= self.num_classes:
            print(
                f"[Warning] semantic_targets contains class index >= num_classes "
                f"({semantic_targets.max()} >= {self.num_classes}), "
                f"will be clamped or ignored"
            )
        
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

