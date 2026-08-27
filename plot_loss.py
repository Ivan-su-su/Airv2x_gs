#!/usr/bin/env python3
"""
解析 train_loss.txt 并可视化训练过程中的 loss 变化。
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 支持中文显示（若系统有中文字体）
def setup_chinese_font():
    try:
        for f in font_manager.fontManager.ttflist:
            if 'SimHei' in f.name or 'DejaVu' in f.name or 'WenQuanYi' in f.name:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']
                break
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

def parse_train_loss(log_path):
    """解析 train_loss.txt，返回 steps 和 losses。"""
    pattern = re.compile(r'Epoch\[(\d+)\], iter\[\d+/\d+\], loss\[([\d.]+)\]')
    steps = []
    losses = []
    epochs = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                epoch, loss = int(m.group(1)), float(m.group(2))
                epochs.append(epoch)
                losses.append(loss)
                steps.append(len(steps))  # 全局 step 从 0 开始
    return np.array(steps), np.array(losses), np.array(epochs)

def parse_validation_loss(log_path):
    """解析 validation_loss.txt，返回 epochs 和 losses。"""
    pattern = re.compile(r'Epoch\[(\d+)\], loss\[([\d.]+)\]')
    epochs_list = []
    losses_list = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                epochs_list.append(int(m.group(1)))
                losses_list.append(float(m.group(2)))
    return np.array(epochs_list), np.array(losses_list)

def smooth_curve(x, window=100):
    """简单移动平均平滑。"""
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')

def main():
    log_dir = '/home/suyi/AirV2X-Perception_copy/opencood/logs/airv2x_intermediate_mambafusion/default_2026_02_06_10_38_45'
    log_path = os.path.join(log_dir, 'validation_loss.txt')
    if not os.path.isfile(log_path):
        print(f'File not found: {log_path}')
        return

    setup_chinese_font()
    steps, losses, epochs = parse_train_loss(log_path)
    print(f'Parsed {len(steps)} records, Epoch 0-{epochs.max()}, loss range [{losses.min():.4f}, {losses.max():.4f}]')

    # 平滑曲线：与 steps 对齐（平滑后长度会变短，取居中对齐）
    window = min(200, max(50, len(losses) // 500))
    smoothed = smooth_curve(losses, window)
    start = (window - 1) // 2
    steps_smooth = steps[start : start + len(smoothed)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # 上图：按 iteration 的 loss（原始 + 平滑）
    ax1 = axes[0]
    ax1.plot(steps, losses, alpha=0.25, linewidth=0.5, color='steelblue', label='Raw loss')
    ax1.plot(steps_smooth, smoothed, color='darkblue', linewidth=1.2, label=f'Smoothed (window={window})')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (by iteration)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(steps))
    # 限制 y 轴范围，排除初期极大值以便看清主要收敛趋势（取 95 分位数）
    y_max = np.percentile(losses, 95)
    ax1.set_ylim(0, y_max * 1.05)

    # 下图：每个 epoch 的 train 平均 loss，以及 validation loss（若有）
    unique_epochs = np.unique(epochs)
    epoch_mean_loss = [losses[epochs == e].mean() for e in unique_epochs]
    ax2 = axes[1]
    ax2.plot(unique_epochs, epoch_mean_loss, color='coral', linewidth=1.5, marker='o', markersize=3, label='Train (mean)')
    y_max_ax2 = max(epoch_mean_loss)

    val_path = os.path.join(log_dir, 'validation_loss.txt')
    if os.path.isfile(val_path):
        val_epochs, val_losses = parse_validation_loss(val_path)
        ax2.plot(val_epochs, val_losses, color='green', linewidth=1.5, marker='s', markersize=4, label='Validation')
        y_max_ax2 = max(y_max_ax2, val_losses.max())
        print(f'Validation: {len(val_epochs)} records, Epochs {val_epochs.min()}-{val_epochs.max()}, loss range [{val_losses.min():.4f}, {val_losses.max():.4f}]')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Train & Validation Loss per Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, y_max_ax2 * 1.05)

    plt.tight_layout()
    out_path = os.path.join(log_dir, 'train_loss_curve.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

    # 单独保存一张 validation loss 图
    if os.path.isfile(val_path):
        val_epochs, val_losses = parse_validation_loss(val_path)
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(val_epochs, val_losses, color='green', linewidth=2, marker='s', markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, val_losses.max() * 1.05)
        plt.tight_layout()
        val_out = os.path.join(log_dir, 'validation_loss_curve.png')
        plt.savefig(val_out, dpi=150, bbox_inches='tight')
        print(f'Saved: {val_out}')
        plt.close()

if __name__ == '__main__':
    main()
