# -*- coding: utf-8 -*-
"""Train AirV2X HEAL / STAMP / NegoCollab from shared Stage-0 bases.

Stage-0 homogeneous bases themselves should be trained with `train.py`.
This script loads the three bases with explicit semantic mapping and
never overwrites shared modules by sequential full-checkpoint loads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.cuda import amp
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils, train_utils
from opencood.tools.train import (
    _get_model_state_dict_for_save,
    is_main_process,
    resume_training_from_checkpoint,
    setup_dataloader,
)
from opencood.utils.airv2x_base_checkpoint import (
    load_local_branch,
    load_raw_checkpoint,
    load_stage0_bases_for_heal_or_stamp,
    resolve_checkpoint_path,
)
from opencood.utils.airv2x_freeze import (
    assert_lss_frozen,
    assert_lss_not_in_optimizer,
    freeze_lss_encoders,
    lss_parameter_checksum,
    print_trainable_parameter_groups,
)


def train_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirV2X heter baseline training")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True)
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--fusion_method", "-f", default="intermediate")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--worker", default=0, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gpu_id", default=0)
    parser.add_argument("--vehicle_dir", default=None, type=str)
    parser.add_argument("--vehicle_epoch", type=int, default=None)
    parser.add_argument("--rsu_dir", default=None, type=str)
    parser.add_argument("--rsu_epoch", type=int, default=None)
    parser.add_argument("--drone_dir", default=None, type=str)
    parser.add_argument("--drone_epoch", type=int, default=None)
    parser.add_argument("--skip_base_load", action="store_true")
    return parser.parse_args()


def _core_method(hypes: dict) -> str:
    return str(hypes.get("model", {}).get("core_method", "")).lower()


def load_shared_stage0(model, opt, hypes: dict) -> None:
    """Load the three Stage-0 bases with explicit semantics."""
    if opt.skip_base_load:
        print("[ckpt] skip_base_load=True, using randomly initialized extra modules")
        return
    if not (opt.vehicle_dir and opt.rsu_dir and opt.drone_dir):
        raise ValueError(
            "HEAL / STAMP / NegoCollab training requires --vehicle_dir, "
            "--rsu_dir and --drone_dir Stage-0 checkpoints"
        )
    method = _core_method(hypes)
    if "negocollab" in method:
        mapping = {
            "vehicle": (opt.vehicle_dir, opt.vehicle_epoch),
            "rsu": (opt.rsu_dir, opt.rsu_epoch),
            "drone": (opt.drone_dir, opt.drone_epoch),
        }
        for agent_type, (ckpt_dir, epoch) in mapping.items():
            path = resolve_checkpoint_path(ckpt_dir, epoch)
            state = load_raw_checkpoint(path)
            load_local_branch(
                model.local_branches[agent_type],
                state,
                agent_type,
                source=path,
            )
        print("[ckpt] NegoCollab: three independent local bases loaded")
    else:
        load_stage0_bases_for_heal_or_stamp(
            model,
            vehicle_ckpt=opt.vehicle_dir,
            rsu_ckpt=opt.rsu_dir,
            drone_ckpt=opt.drone_dir,
            vehicle_epoch=opt.vehicle_epoch,
            rsu_epoch=opt.rsu_epoch,
            drone_epoch=opt.drone_epoch,
        )
    freeze_lss_encoders(model)
    assert_lss_frozen(model)
    print_trainable_parameter_groups(model, title="after Stage-0 load")


def main() -> None:
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    hypes["tag"] = opt.tag
    main_process = is_main_process(opt)

    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    train_loader = setup_dataloader(train_dataset, hypes, opt, is_train=True)
    val_loader = setup_dataloader(val_dataset, hypes, opt, is_train=False)

    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        gpu_id = int(opt.gpu) if getattr(opt, "distributed", False) else int(opt.gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")
    model.to(device)
    load_shared_stage0(model, opt, hypes)

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True
        )

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    raw_model = model.module if hasattr(model, "module") else model
    assert_lss_not_in_optimizer(raw_model, optimizer)
    lss_before = lss_parameter_checksum(raw_model)

    scaler = amp.GradScaler() if opt.amp and torch.cuda.is_available() else None
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, n_iter_per_epoch=len(train_loader)
    )

    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, _ = resume_training_from_checkpoint(
            saved_path=saved_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes) if main_process else None
        if opt.distributed:
            holder = [saved_path]
            torch.distributed.broadcast_object_list(holder, src=0)
            saved_path = holder[0]
    assert saved_path is not None
    writer = SummaryWriter(saved_path) if main_process else None

    epochs = hypes["train_params"]["epoches"]
    for epoch in range(init_epoch, epochs):
        if opt.distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=not main_process)
        for i, batch_data in pbar:
            if batch_data is None:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch
            with amp.autocast(enabled=scaler is not None):
                output_dict = model(batch_data["ego"])
                loss = criterion(output_dict, batch_data["ego"]["label_dict"])
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if i == 0:
                from opencood.utils.airv2x_freeze import assert_no_lss_grad

                assert_no_lss_grad(raw_model)
                lss_after = lss_parameter_checksum(raw_model)
                if abs(lss_after - lss_before) > 1e-6:
                    raise AssertionError(
                        f"LSS checksum changed after optimizer step: "
                        f"{lss_before} → {lss_after}"
                    )
            if main_process:
                try:
                    criterion.logging(epoch, i, len(train_loader), writer, pbar)
                except TypeError:
                    criterion.logging(epoch, i, len(train_loader), writer)
        scheduler.step(epoch)
        if main_process and epoch % hypes["train_params"]["save_freq"] == 0:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": _get_model_state_dict_for_save(model),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(saved_path, f"net_epoch{epoch + 1}.pth"))
    print(f"Training finished. Checkpoints: {saved_path}")


if __name__ == "__main__":
    main()
