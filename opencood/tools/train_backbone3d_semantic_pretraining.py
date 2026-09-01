# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib
# 
# 预训练脚本：用于预训练 Backbone3D 的语义分类部分
# 训练整个流程：VFE → Encoder → Semantic Head

import argparse
import os
import sys
from pathlib import Path

import torch, resource
torch.multiprocessing.set_sharing_strategy('file_system')
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
from torch.cuda import amp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils, train_utils

def train_parser():
    """
    Configure command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Backbone3D Semantic Pretraining")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                      help="Path to training configuration yaml file")
    parser.add_argument("--model_dir", default="",
                      help="Path to continue training from a checkpoint")
    parser.add_argument("--dist_url", default="env://",
                      help="URL used to set up distributed training")
    parser.add_argument("--rank", default=0, type=int,
                      help="Node rank for distributed training")
    parser.add_argument("--tag", default="semantic_pretrain",
                      help="Tag for the training session")
    parser.add_argument("--worker", default=8, type=int,
                      help="Number of workers for data loading")
    parser.add_argument("--amp", action="store_true",
                      help="Enable automatic mixed precision training")
    parser.add_argument("--gpu_id", default=0, type=int,
                      help="GPU ID to use")
    return parser.parse_args()

def setup_dataloader(dataset, hypes, opt, is_train=True):
    """
    Set up data loader for training or validation.
    
    Args:
        dataset: Dataset instance
        hypes (dict): Configuration parameters
        opt: Command line arguments
        is_train (bool): Whether this is for training or validation
        
    Returns:
        DataLoader: Configured data loader
    """
    if opt.distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
        if is_train:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, hypes["train_params"]["batch_size"], drop_last=True)
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=opt.worker,
                collate_fn=dataset.collate_batch_train,
                timeout=5200,
                pin_memory=True,
                prefetch_factor=1
            )
        else:
            loader = DataLoader(
                dataset,
                sampler=sampler,
                num_workers=opt.worker,
                collate_fn=dataset.collate_batch_train,
                timeout=5200,
                pin_memory=True,
                prefetch_factor=1
            )
    else:
        loader = DataLoader(
            dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=opt.worker,
            collate_fn=dataset.collate_batch_train,
            shuffle=is_train,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4
        )
    return loader

def validate_model(model, val_loader, criterion, epoch, device, hypes, scaler=None):
    """
    验证模型
    
    Args:
        model: 模型实例
        val_loader: 验证数据加载器
        criterion: 损失函数
        epoch: 当前epoch
        device: 设备
        hypes: 配置字典
        scaler: 混合精度scaler
    
    Returns:
        total_loss: 总损失
        n_sample: 样本数量
    """
    model.eval()
    total_loss, n_sample = 0.0, 0
    with torch.no_grad():
        for _, batch_data in tqdm(enumerate(val_loader),
                                  total=len(val_loader),
                                  desc="Validation", leave=False):
            if batch_data is None:
                continue
            
            batch_data = train_utils.to_device(batch_data, device)
            
            # 获取available_agent列表
            available_agent = hypes.get("model", {}).get("args", {}).get("available_agent", ["vehicle"])
            
            with amp.autocast(enabled=scaler is not None):
                # 前向传播
                batch_dict = model(batch_data["ego"], available_agent=available_agent)
                
                # 计算损失
                # 从batch_dict中提取semantic_logits和semantic_targets
                output_dict = {}
                target_dict = {}
                
                for agent in available_agent:
                    if agent in batch_dict:
                        # 收集semantic_logits
                        if 'semantic_logits' in batch_dict[agent]:
                            if 'semantic_logits' not in output_dict:
                                output_dict['semantic_logits'] = batch_dict[agent]['semantic_logits']
                            else:
                                # 如果有多个agent，合并logits
                                output_dict['semantic_logits'] = torch.cat([
                                    output_dict['semantic_logits'],
                                    batch_dict[agent]['semantic_logits']
                                ], dim=0)
                        
                        # 收集semantic_targets
                        if 'semantic_targets' in batch_dict[agent]:
                            if 'semantic_targets' not in target_dict:
                                target_dict['semantic_targets'] = batch_dict[agent]['semantic_targets']
                            else:
                                target_dict['semantic_targets'] = torch.cat([
                                    target_dict['semantic_targets'],
                                    batch_dict[agent]['semantic_targets']
                                ], dim=0)
                
                # 检查是否有有效的logits和targets
                if 'semantic_logits' not in output_dict or 'semantic_targets' not in target_dict:
                    print("[Warning] Missing semantic_logits or semantic_targets, skipping batch")
                    continue
                
                # 确保logits和targets的数量匹配
                if output_dict['semantic_logits'].shape[0] != target_dict['semantic_targets'].shape[0]:
                    print(f"[Warning] Mismatch: logits shape {output_dict['semantic_logits'].shape[0]}, "
                          f"targets shape {target_dict['semantic_targets'].shape[0]}, skipping batch")
                    continue
                
                loss = criterion(output_dict, target_dict)

            bsz = 1 if isinstance(loss, torch.Tensor) else len(loss)
            total_loss += loss.item() * bsz   
            n_sample += bsz

    return total_loss, n_sample

def main():
    """Main training function for semantic pretraining."""
    # Setup
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    hypes["tag"] = opt.tag
    
    # Build datasets
    print("Building datasets...")
    # 需要启用visualize模式以包含origin_lidar和label_dict等字段
    visualize_mode = hypes.get("visualize", True)  # 预训练需要标签，默认启用
   
    train_dataset = build_dataset(hypes, visualize=visualize_mode, train=True)
    val_dataset = build_dataset(hypes, visualize=visualize_mode, train=False)
    
    # Create dataloaders
    train_loader = setup_dataloader(train_dataset, hypes, opt, is_train=True)
    val_loader = setup_dataloader(val_dataset, hypes, opt, is_train=False)
    
    # Create model - 使用预训练版本的模型
    print("Creating pretraining model...")
    # 修改模型创建逻辑，使用预训练版本
    backbone_config = hypes["model"]["args"]["BACKBONE_3D"]
    
    # 导入预训练模型
    from opencood.models.gaussian_modules.backbone3d_semantic_pretraining import Gaussian3DBackboneForPretraining
    
    model = Gaussian3DBackboneForPretraining(model_cfg=backbone_config)
    
    total_params = sum(p.nelement() for p in model.parameters())
    trainable_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 打印模型结构信息
    print("\n[Model Structure for Pretraining]")
    print("  训练整个流程（所有模块都会参与反向传播）：")
    print("    1. VFE (DynamicVoxelVFE): 点云 → 体素特征")
    print("    2. Encoder: 稀疏卷积编码器 (SparseConv3d)")
    print("    3. Semantic Head: 语义分类头 (输出 num_classes 维 logits)")
    print("  ")
    print("  损失计算：使用 semantic_logits_dense (logits)，不是 semantic_probs (probs)")
    print("  原因：CrossEntropyLoss 内部会做 log_softmax，传入已 softmax 的值会导致数值不稳定\n")
    
    # Setup device and distributed training
    if torch.cuda.is_available():
        gpu_id = int(opt.gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")
    
    model.to(device)

    if opt.distributed:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        device = torch.device(f"cuda:{opt.gpu}")
        torch.cuda.set_device(opt.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            find_unused_parameters=True
        )
    
    # Setup training components
    # 创建语义分类损失函数
    loss_config = hypes.get("loss", {}).get("args", {})
    
    from opencood.models.gaussian_modules.backbone3d_semantic_loss import Backbone3DSemanticLoss
    criterion = Backbone3DSemanticLoss(loss_config)
    
    optimizer = train_utils.setup_optimizer(hypes, model)
    
    # Initialize mixed precision training if enabled
    scaler = amp.GradScaler() if opt.amp and torch.cuda.is_available() else None
    if scaler:
        print("Using mixed precision training")
    
    # Load checkpoint if continuing training
    if opt.model_dir:
        init_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, init_epoch=init_epoch, n_iter_per_epoch=len(train_loader))
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        print(f"Results will be saved to: {saved_path}")
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, n_iter_per_epoch=len(train_loader))
    
    # Training loop
    writer = SummaryWriter(saved_path)
    print("Starting semantic pretraining...")
    epochs = hypes["train_params"]["epoches"]
    
    
    for epoch in range(init_epoch, max(epochs, init_epoch)):
        # Print current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}, Current learning rate: {current_lr:.6f}")
        
        # Training epoch
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch_data in pbar:
            if batch_data is None:
                continue
            # 检查可用的agents
            available_agents = []
            for agent in ['vehicle', 'rsu', 'drone']:
                batch_dict = batch_data["ego"]
                if agent == 'vehicle' and 'origin_lidar' in batch_dict:
                    # 检查vehicle的origin_lidar是否有效
                    origin_lidar = batch_dict['origin_lidar']
                    if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                        available_agents.append(agent)
                elif agent != 'vehicle' and f'origin_lidar_{agent}' in batch_dict:
                    # 检查RSU/Drone的origin_lidar是否有效
                    origin_lidar = batch_dict[f'origin_lidar_{agent}']
                    if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                        available_agents.append(agent)
            
            # Forward pass
            model.zero_grad()
            optimizer.zero_grad()
            
            batch_data = train_utils.to_device(batch_data, device)
            
            with amp.autocast(enabled=scaler is not None):
                # 前向传播：VFE → Encoder → Semantic Head
                batch_dict = model(batch_data["ego"], available_agent=available_agents)
                
                # 准备损失计算的输入
                output_dict = {}
                target_dict = {}
                
                # 从batch_dict中提取所有agent的semantic_logits和semantic_targets
                for agent in available_agents:
                    if agent in batch_dict:
                        # 收集semantic_logits
                        if 'semantic_logits' in batch_dict[agent]:
                            if 'semantic_logits' not in output_dict:
                                output_dict['semantic_logits'] = batch_dict[agent]['semantic_logits']
                            else:
                                # 合并所有agent的logits
                                output_dict['semantic_logits'] = torch.cat([
                                    output_dict['semantic_logits'],
                                    batch_dict[agent]['semantic_logits']
                                ], dim=0)
                        
                        # 收集semantic_targets
                        if 'semantic_targets' in batch_dict[agent]:
                            if 'semantic_targets' not in target_dict:
                                target_dict['semantic_targets'] = batch_dict[agent]['semantic_targets']
                            else:
                                # 合并所有agent的targets
                                target_dict['semantic_targets'] = torch.cat([
                                    target_dict['semantic_targets'],
                                    batch_dict[agent]['semantic_targets']
                                ], dim=0)
                
                # 检查是否有有效的logits和targets
                if 'semantic_logits' not in output_dict or 'semantic_targets' not in target_dict:
                    print("[Warning] Missing semantic_logits or semantic_targets, skipping batch")
                    continue
                
                # 确保logits和targets的数量匹配
                if output_dict['semantic_logits'].shape[0] != target_dict['semantic_targets'].shape[0]:
                    print(f"[Warning] Mismatch: logits shape {output_dict['semantic_logits'].shape[0]}, "
                          f"targets shape {target_dict['semantic_targets'].shape[0]}, skipping batch")
                    continue
                
                # 计算损失（使用logits，不是probs）
                loss = criterion(output_dict, target_dict)
            
            # Backward pass with mixed precision support
            if scaler is not None:
                scaler.scale(loss).backward()
                # 梯度裁剪的作用是防止梯度爆炸，尤其是在深层网络或长序列训练时，过大的梯度可能导致模型参数瞬间异常、训练不收敛或者出现NaN。
                # 通过clip_grad_norm_，可以将每个参数的梯度L2范数限制在max_norm范围内，从而提升训练的稳定性与收敛性。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            
            # Update progress bar
            try:
                print_msg = criterion.logging(epoch, i, len(train_loader), writer, pbar)
            except:
                print_msg = criterion.logging(epoch, i, len(train_loader), writer)
            if print_msg:
                pbar.set_description(print_msg)
            
            # Log training loss
            if opt.rank == 0:
                with open(os.path.join(saved_path, "train_loss.txt"), "a+") as f:
                    f.write(f"Epoch[{epoch}], iter[{i}/{len(train_loader)}], loss[{loss.item():.4f}]\n")
            
            # 清理显存
            torch.cuda.empty_cache()
                    
        if opt.distributed:
            torch.cuda.synchronize()             
            torch.distributed.barrier()  
        
        # Save checkpoint
        if opt.rank == 0 and epoch % hypes["train_params"]["save_freq"] == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            
            checkpoint_path = os.path.join(saved_path, f"net_epoch{epoch + 1}.pth")
            torch.save(save_dict, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
        
        # TODO: 暂时不用validation
        # Validation
        need_val = (epoch % hypes["train_params"]["eval_freq"] == 0)

        if need_val:
            if opt.distributed and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            local_sum, local_cnt = validate_model(model, val_loader,
                                                criterion, epoch,
                                                device, hypes, scaler)

            if opt.distributed:
                stats = torch.tensor([local_sum, local_cnt],
                                    dtype=torch.float32, device=device)
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                global_sum, global_cnt = stats.tolist()
            else:
                global_sum, global_cnt = local_sum, local_cnt

            if opt.rank == 0:
                val_loss = global_sum / max(global_cnt, 1)
                print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
                writer.add_scalar("Validate_Loss", val_loss, epoch)
                with open(os.path.join(saved_path, "validation_loss.txt"), "a+") as f:
                    f.write(f"Epoch[{epoch}], loss[{val_loss:.4f}]\n")

        scheduler.step(epoch)
                
        if opt.distributed:
            torch.distributed.barrier(device_ids=[opt.gpu])
        
    print(f"\nPretraining finished. Checkpoints saved to {saved_path}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

