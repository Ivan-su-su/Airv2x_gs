# -*- coding: utf-8 -*-
"""TEMPORARY TEST→TRAIN mix. Delete this file and its call sites later.

Used by vehicle/rsu branch finetune:
  * ``apply_test_mix_to_hypes`` fills ``extra_train_scenes`` (first keep_frac
    of every TEST scene, stride 1). Dataset loading stays in basedataset.
  * Mix batches skip heatmap (TEST has no SAM3 ``*_seg.bin``); depth only.
  * ``dump_mix_manifest`` writes timestamps to exclude from later TEST eval.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch


def apply_test_mix_to_hypes(hypes: Dict[str, Any]) -> None:
    """Replace ``extra_train_scenes`` with first ``test_mix_keep_frac`` of TEST."""
    finetune = hypes.get("finetune") or {}
    frac = float(finetune.get("test_mix_keep_frac", 0.2))
    test_dir = str(hypes["test_dir"])
    scenes = sorted(
        os.path.join(test_dir, name)
        for name in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, name)) and not name.startswith(".")
    )
    if not scenes:
        raise FileNotFoundError(f"no TEST scenes under {test_dir}")
    hypes["extra_train_scenes"] = [
        {"path": path, "keep_frac": frac, "stride": 1} for path in scenes
    ]
    print(
        f"[test-mix] extra_train_scenes n={len(scenes)} keep_frac={frac} "
        f"stride=1 (heatmap skipped on these batches)"
    )


def batch_is_test_mix(ego: Dict[str, Any]) -> bool:
    """True iff this collated batch came from the TEST mix."""
    meta = ego.get("metadata_path_list") or ego.get("metadata_path") or ""
    if isinstance(meta, (list, tuple)):
        meta = meta[0] if meta else ""
    return "/test/" in str(meta).replace("\\", "/")


def background_heatmap_targets(
    predictions: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """All-background ids matching heatmap logits (no SAM3, no heatmap grad)."""
    targets: Dict[str, torch.Tensor] = {}
    for agent_type, pred in predictions.items():
        logits = pred["heatmap_logits"]
        targets[agent_type] = torch.zeros(
            int(logits.shape[0]),
            int(logits.shape[-2]),
            int(logits.shape[-1]),
            dtype=torch.long,
            device=logits.device,
        )
    return targets


def mix_records(dataset: Any) -> List[Dict[str, str]]:
    """TEST timestamps currently in the train index."""
    records: List[Dict[str, str]] = []
    database = getattr(dataset, "scenario_database", None)
    if not database:
        return records
    for scene_dict in database.values():
        first = next(iter(scene_dict.values()))
        scene_rows: List[Dict[str, str]] = []
        for timestamp, rec in first.items():
            if not isinstance(rec, dict) or "metadata_path" not in rec:
                continue
            meta = str(rec["metadata_path"]).replace("\\", "/")
            if "/test/" not in meta:
                scene_rows = []
                break
            parts = meta.split("/")
            scene = parts[parts.index("test") + 1] if "test" in parts else ""
            scene_rows.append(
                {
                    "scene": scene,
                    "timestamp": str(timestamp),
                    "metadata_path": rec["metadata_path"],
                }
            )
        records.extend(scene_rows)
    return records


def dump_mix_manifest(dataset: Any, saved_path: str) -> Path:
    """Write ``extra_train_mix.json`` under the run dir for held-out TEST eval."""
    records = mix_records(dataset)
    out = Path(saved_path) / "extra_train_mix.json"
    payload = {
        "n": len(records),
        "note": "Exclude these timestamps when scoring TEST (they were in TRAIN).",
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[test-mix] wrote {out} n={len(records)}")
    return out
