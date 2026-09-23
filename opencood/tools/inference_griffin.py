# -*- coding: utf-8 -*-
"""Evaluate Griffin full Gaussian detector with OpenCOOD multiclass IoU AP."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils_airv2x as eval_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--eval_epoch", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    class Opt:
        model_dir = args.model_dir
        config_file = "config.yaml"

    hypes = yaml_utils.load_yaml(None, Opt())
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = train_utils.create_model(hypes).to(device)
    epoch_id, model = train_utils.load_model(
        args.model_dir,
        model,
        epoch=args.eval_epoch,
        start_from_best=False,
        device=device,
    )
    model.eval()

    result_stat = defaultdict(dict)
    iou_thresholds = (0.3, 0.5, 0.7)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Griffin eval epoch={epoch_id}"):
            batch = train_utils.to_device(batch, device)
            output = {"ego": model(batch["ego"])}
            pred_box, pred_score, pred_labels, gt_box, gt_labels, _ = (
                dataset.post_process(batch, output)
            )

            if pred_box is None:
                pred_box = gt_box.new_empty((0, *gt_box.shape[1:]))
                pred_score = gt_box.new_empty((0,))
                pred_labels = torch.empty(
                    (0,), dtype=torch.long, device=gt_box.device
                )

            for iou in iou_thresholds:
                eval_utils.calculate_multiclass_tp_fp(
                    pred_box,
                    pred_score,
                    pred_labels,
                    gt_box,
                    gt_labels,
                    iou_thresh=iou,
                    result_stat=result_stat,
                )

    lines = []
    for iou in iou_thresholds:
        ap_per_class, mean_ap = eval_utils.compute_multiclass_ap_map(
            result_stat,
            iou_thresh=iou,
            global_sort_detections=True,
        )
        lines.append(
            f"epoch={epoch_id} IoU={iou:.1f} "
            f"AP={ap_per_class} mAP={mean_ap:.4f}"
        )

    out_path = os.path.join(args.model_dir, f"griffin_eval_epoch{epoch_id}.txt")
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
