# -*- coding: utf-8 -*-
"""Raw Griffin dataset adapter for Gaussian-0822 P1 adaptation.

This loader intentionally reads ``griffin-release`` directly instead of the
NuScenes conversion because P1 adaptation needs two raw-only signals:

* per-view instance segmentation masks for objectness heatmap supervision;
* vehicle ``lidar_top`` PLY/BIN point clouds for sparse camera-depth supervision.

The spatial contract matches the AirV2X baseline branch:

* Vehicle: four RGB cameras, output stride 8 -> 45x80 targets.
* Drone: bottom RGB only, output stride 4 -> 90x160 heatmap target.

There is no drone depth target here. Griffin drone bottom uses analytic ideal
ray/ground projection downstream.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from opencood.utils.camera_utils import imagenet_normalize_display_rgb


VEHICLE_CAMS: Tuple[str, ...] = ("front", "back", "left", "right")
DRONE_CAMS: Tuple[str, ...] = ("bottom",)

# Griffin is generated from CARLA instance-segmentation cameras. In the raw
# BGRA PNG read by OpenCV, channel 2 is CARLA's semantic-tag R channel.
# Griffin's own converter keeps these detection classes:
#   12 pedestrian; 14 car; 15 truck; 16 bus; 18 motorcycle; 19 bicycle.
DEFAULT_TARGET_SEMANTIC_TAGS: Tuple[int, ...] = (12, 14, 15, 16, 18, 19)


def _read_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _frame_id(value: Any) -> str:
    value = str(value)
    return value.zfill(6) if value.isdigit() else value


def _load_calib(side_root: Path, sensor: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = _read_json(side_root / "calib" / f"{sensor}.json")
    extrinsic = np.asarray(data["extrinsic"], dtype=np.float64)
    intrinsic = data.get("intrinsic")
    intrinsic_np = None if intrinsic is None else np.asarray(intrinsic, dtype=np.float64)
    if extrinsic.shape != (4, 4):
        raise ValueError(f"bad extrinsic for {sensor}: {extrinsic.shape}")
    if intrinsic_np is not None and intrinsic_np.shape != (3, 3):
        raise ValueError(f"bad intrinsic for {sensor}: {intrinsic_np.shape}")
    return extrinsic, intrinsic_np


def _load_scene_infos(vehicle_root: Path) -> List[Dict[str, Any]]:
    path = vehicle_root / "scene_infos.json"
    data = _read_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected scene_infos format: {path}")
    return data


def _scene_frames(scene: Mapping[str, Any]) -> List[str]:
    info = scene.get("info") or {}
    frames = info.get("frames") or []
    return [_frame_id(x) for x in frames]


def _scene_name(scene: Mapping[str, Any], index: int) -> str:
    raw = scene.get("name")
    if raw is None:
        return f"scene-{index:04d}"
    return str(raw)


def _match_split_scene(raw_name: str, split_name: str) -> bool:
    """Match raw ``scene_infos`` names to Griffin converter split names.

    Converted NuScenes names are ``scene-XXXX-{raw_name}``, while some split
    files use the raw suffix directly. Supporting both keeps this adapter
    independent from converter internals.
    """
    return split_name == raw_name or split_name.endswith("-" + raw_name)


def resolve_split_frames(
    vehicle_root: Path,
    split: str,
    split_json: Optional[Path],
) -> List[str]:
    scenes = _load_scene_infos(vehicle_root)
    split = str(split)
    if split not in ("train", "val"):
        raise ValueError(f"split must be train/val, got {split}")

    selected: List[Mapping[str, Any]] = []
    if split_json is not None and split_json.exists():
        split_data = _read_json(split_json)
        # Official Griffin split JSON uses:
        # {"batch_split": {"train": [...], "val": [...]}}
        # Also keep compatibility with a flat train/val JSON.
        split_block = split_data.get("batch_split", split_data)
        if not isinstance(split_block, Mapping):
            raise ValueError(f"unexpected split json structure: {split_json}")
        names = [str(x) for x in split_block.get(split, [])]
        if not names:
            raise ValueError(f"{split_json} has no non-empty '{split}' list")
        unmatched = list(names)
        for i, scene in enumerate(scenes):
            raw_name = _scene_name(scene, i)
            matches = [name for name in names if _match_split_scene(raw_name, name)]
            if matches:
                selected.append(scene)
                for name in matches:
                    if name in unmatched:
                        unmatched.remove(name)
        if unmatched:
            examples = [_scene_name(scene, i) for i, scene in enumerate(scenes[:8])]
            raise ValueError(
                f"Could not match {len(unmatched)} Griffin split scene names. "
                f"First unmatched={unmatched[:5]}, raw examples={examples}."
            )
    else:
        # Safe deterministic fallback only. Prefer the official split JSON.
        cut = max(1, int(round(0.8 * len(scenes))))
        selected = scenes[:cut] if split == "train" else scenes[cut:]
        print(
            "[GriffinP1Dataset] WARNING: official split_json missing; using "
            f"deterministic scene 80/20 fallback for {split}."
        )

    frames: List[str] = []
    for scene in selected:
        frames.extend(_scene_frames(scene))
    return frames


def photometric_distortion_bgr(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Dependency-light replica of Griffin's PhotoMetricDistortionMultiViewImage.

    The official transform applies each operation with p=0.5 and chooses
    contrast either before or after HSV jitter. Input/output are BGR float32
    in approximately [0,255].
    """
    out = img.astype(np.float32, copy=True)

    if rng.randint(2):
        out += rng.uniform(-32.0, 32.0)

    mode = int(rng.randint(2))
    if mode == 1 and rng.randint(2):
        out *= rng.uniform(0.5, 1.5)

    # OpenCV float HSV expects RGB/BGR values in [0,1] for stable S/V ranges.
    tmp = np.clip(out / 255.0, 0.0, 1.0)
    hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
    if rng.randint(2):
        hsv[..., 1] *= rng.uniform(0.5, 1.5)
    if rng.randint(2):
        hsv[..., 0] += rng.uniform(-18.0, 18.0)
        hsv[..., 0] = np.mod(hsv[..., 0], 360.0)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255.0

    if mode == 0 and rng.randint(2):
        out *= rng.uniform(0.5, 1.5)

    if rng.randint(2):
        out = out[..., rng.permutation(3)]

    return np.clip(out, 0.0, 255.0).astype(np.float32)


def _bgr_to_normalized_tensor(
    bgr: np.ndarray,
    final_hw: Tuple[int, int],
    image_scale: float,
) -> torch.Tensor:
    """Match Griffin P1 preprocessing exactly: scale first, then final resize."""
    scale = float(image_scale)
    if scale <= 0.0:
        raise ValueError(f"image_scale must be > 0, got {scale}")
    work = bgr
    if abs(scale - 1.0) > 1.0e-8:
        src_h, src_w = int(bgr.shape[0]), int(bgr.shape[1])
        scaled_w = max(1, int(round(src_w * scale)))
        scaled_h = max(1, int(round(src_h * scale)))
        work = cv2.resize(
            bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
        )
    final_h, final_w = int(final_hw[0]), int(final_hw[1])
    resized = cv2.resize(
        work, (final_w, final_h), interpolation=cv2.INTER_LINEAR
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return imagenet_normalize_display_rgb(tensor)


def _rgb_to_normalized_tensor(
    bgr: np.ndarray, final_hw: Tuple[int, int]
) -> torch.Tensor:
    """Backward-compatible one-step resize used by older debug scripts."""
    return _bgr_to_normalized_tensor(bgr, final_hw, image_scale=1.0)


def _mask_to_foreground(
    mask: np.ndarray,
    mode: str = "carla_semantic",
    target_semantic_tags: Sequence[int] = DEFAULT_TARGET_SEMANTIC_TAGS,
) -> np.ndarray:
    """Decode Griffin/CARLA instance segmentation into binary objectness.

    CARLA raw instance-segmentation encodes:
      R = semantic tag
      G/B = unique instance id bytes
    OpenCV reads PNG as BGR/BGRA, therefore semantic R is ``mask[..., 2]``.

    ``nonzero`` is retained only for debugging/compatibility; it is wrong for
    Griffin objectness because roads/buildings/etc. also have non-zero IDs.
    """
    mode = str(mode)
    if mode == "carla_semantic":
        if mask.ndim != 3 or mask.shape[2] < 3:
            raise ValueError(
                f"carla_semantic expects BGR/BGRA instance PNG, got {mask.shape}"
            )
        semantic = mask[..., 2]
        tags = np.asarray([int(x) for x in target_semantic_tags], dtype=semantic.dtype)
        if tags.size == 0:
            raise ValueError("target_semantic_tags must be non-empty")
        return np.isin(semantic, tags)

    if mode == "nonzero":
        if mask.ndim == 2:
            return mask != 0
        if mask.ndim == 3:
            channels = mask[..., :3] if mask.shape[2] == 4 else mask
            return np.any(channels != 0, axis=2)
        raise ValueError(f"unexpected instance mask shape {mask.shape}")

    raise ValueError(
        f"Unsupported mask_foreground_mode={mode}; "
        "use carla_semantic (recommended) or nonzero"
    )


def _block_any(binary: np.ndarray, stride: int) -> np.ndarray:
    if binary.ndim != 2:
        raise ValueError(f"binary mask must be HxW, got {binary.shape}")
    h, w = binary.shape
    stride = int(stride)
    if h % stride != 0 or w % stride != 0:
        raise ValueError(f"{h}x{w} not divisible by stride={stride}")
    return binary.reshape(h // stride, stride, w // stride, stride).any(axis=(1, 3))


_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _resolve_lidar_path(vehicle_root: Path, frame: str) -> Path:
    """Return the raw Griffin vehicle lidar file for ``frame``.

    The released 50-scene Griffin data stores ``lidar_top`` as ``.ply``.
    ``.bin`` is kept as a compatibility fallback for converted/local variants.
    """
    lidar_dir = vehicle_root / "lidar" / "lidar_top"
    for suffix in (".ply", ".bin"):
        path = lidar_dir / f"{frame}{suffix}"
        if path.exists():
            return path
    # Return the canonical release path so FileNotFoundError is informative.
    return lidar_dir / f"{frame}.ply"


def _load_ply_xyz(path: Path) -> np.ndarray:
    """Read x/y/z from an ASCII or binary PLY without external dependencies."""
    with path.open("rb") as handle:
        first = handle.readline().strip()
        if first != b"ply":
            raise ValueError(f"{path}: missing PLY magic header")

        fmt = None
        vertex_count = None
        current_element = None
        vertex_props: List[Tuple[str, str]] = []

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"{path}: truncated PLY header")
            line = raw_line.decode("ascii", errors="strict").strip()
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) >= 2 and parts[1] == "list":
                    raise ValueError(
                        f"{path}: list-valued vertex PLY properties are unsupported"
                    )
                if len(parts) != 3:
                    raise ValueError(f"{path}: malformed PLY property line: {line}")
                prop_type, prop_name = parts[1], parts[2]
                if prop_type not in _PLY_DTYPES:
                    raise ValueError(
                        f"{path}: unsupported PLY scalar type {prop_type}"
                    )
                vertex_props.append((prop_name, prop_type))
            elif parts[0] == "end_header":
                break

        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
        if vertex_count is None:
            raise ValueError(f"{path}: PLY has no vertex element")
        prop_names = [name for name, _ in vertex_props]
        for axis in ("x", "y", "z"):
            if axis not in prop_names:
                raise ValueError(
                    f"{path}: PLY vertex properties missing '{axis}': {prop_names}"
                )
        xyz_cols = [prop_names.index(axis) for axis in ("x", "y", "z")]

        if vertex_count == 0:
            return np.zeros((0, 3), dtype=np.float64)

        if fmt == "ascii":
            # max_rows prevents any later face/edge elements from being parsed.
            values = np.loadtxt(
                handle,
                dtype=np.float64,
                max_rows=int(vertex_count),
            )
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.shape[0] != int(vertex_count):
                raise ValueError(
                    f"{path}: expected {vertex_count} vertices, got {values.shape[0]}"
                )
            if values.shape[1] < len(vertex_props):
                raise ValueError(
                    f"{path}: expected {len(vertex_props)} vertex properties, "
                    f"got {values.shape[1]}"
                )
            return values[:, xyz_cols].astype(np.float64, copy=False)

        byte_order = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype(
            [
                (name, byte_order + _PLY_DTYPES[prop_type])
                for name, prop_type in vertex_props
            ]
        )
        vertices = np.fromfile(handle, dtype=dtype, count=int(vertex_count))
        if vertices.shape[0] != int(vertex_count):
            raise ValueError(
                f"{path}: expected {vertex_count} binary vertices, "
                f"got {vertices.shape[0]}"
            )
        return np.column_stack(
            [vertices["x"], vertices["y"], vertices["z"]]
        ).astype(np.float64, copy=False)


def _load_lidar_xyz(path: Path, num_features: int = 4) -> np.ndarray:
    """Load Griffin vehicle lidar XYZ from release PLY or legacy float32 BIN."""
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return _load_ply_xyz(path)
    if suffix == ".bin":
        raw = np.fromfile(str(path), dtype=np.float32)
        num_features = int(num_features)
        if num_features < 3 or raw.size % num_features != 0:
            raise ValueError(
                f"{path}: {raw.size} float32 values cannot reshape to "
                f"Nx{num_features}. Adjust lidar_num_features."
            )
        pts = raw.reshape(-1, num_features)
        return pts[:, :3].astype(np.float64, copy=False)
    raise ValueError(f"Unsupported Griffin lidar file: {path}")


def project_lidar_to_camera_cells(
    xyz_lidar: np.ndarray,
    t_lidar_to_ego: np.ndarray,
    t_cam_to_ego: np.ndarray,
    k_orig: np.ndarray,
    original_hw: Tuple[int, int],
    final_hw: Tuple[int, int],
    stride: int,
    foreground_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project object LiDAR points to camera and keep nearest optical-z per P1 cell."""
    orig_h, orig_w = int(original_hw[0]), int(original_hw[1])
    final_h, final_w = int(final_hw[0]), int(final_hw[1])
    stride = int(stride)
    out_h, out_w = final_h // stride, final_w // stride
    if foreground_mask.shape != (orig_h, orig_w):
        raise ValueError(
            f"foreground_mask={foreground_mask.shape} vs image={(orig_h, orig_w)}"
        )

    xyz_h = np.concatenate(
        [xyz_lidar, np.ones((xyz_lidar.shape[0], 1), dtype=np.float64)], axis=1
    )
    t_lidar_to_cam = np.linalg.inv(t_cam_to_ego) @ t_lidar_to_ego
    cam = (t_lidar_to_cam @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    valid = np.isfinite(z) & (z > 1.0e-3)
    cam = cam[valid]
    z = z[valid]

    if cam.shape[0] == 0:
        depth = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return depth, np.zeros((out_h, out_w), dtype=np.bool_)

    uvw = (k_orig @ cam.T).T
    u = uvw[:, 0] / z
    v = uvw[:, 1] / z
    in_img = (u >= 0.0) & (u < orig_w) & (v >= 0.0) & (v < orig_h)
    u = u[in_img]
    v = v[in_img]
    z = z[in_img]

    # Only points projected onto Griffin object pixels supervise depth.
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    on_object = foreground_mask[vi, ui]
    u = u[on_object] * (float(final_w) / float(orig_w))
    v = v[on_object] * (float(final_h) / float(orig_h))
    z = z[on_object]

    j = np.floor(u / stride).astype(np.int64)
    i = np.floor(v / stride).astype(np.int64)
    in_grid = (i >= 0) & (i < out_h) & (j >= 0) & (j < out_w)
    i, j, z = i[in_grid], j[in_grid], z[in_grid]

    depth = np.full((out_h, out_w), np.inf, dtype=np.float32)
    if z.size:
        flat_idx = i * out_w + j
        np.minimum.at(depth.reshape(-1), flat_idx, z.astype(np.float32))
    valid_cells = np.isfinite(depth)
    depth[~valid_cells] = np.nan
    return depth, valid_cells


class GriffinP1Dataset(Dataset):
    """Two-agent Griffin P1 dataset: vehicle 4-view + drone bottom-only."""

    def __init__(self, cfg: Mapping[str, Any], split: str) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.split = str(split)
        self.root = Path(self.cfg["root"]).expanduser().resolve()
        self.vehicle_root = self.root / "vehicle-side"
        self.drone_root = self.root / "drone-side"
        self.final_hw = tuple(int(x) for x in self.cfg.get("final_dim", [360, 640]))
        self.image_scale = float(self.cfg.get("image_scale", 0.5))
        self.vehicle_stride = int(self.cfg.get("vehicle_stride", 8))
        self.drone_stride = int(self.cfg.get("drone_stride", 4))
        self.lidar_num_features = int(self.cfg.get("lidar_num_features", 4))
        self.mask_mode = str(
            self.cfg.get("mask_foreground_mode", "carla_semantic")
        )
        self.target_semantic_tags = tuple(
            int(x)
            for x in self.cfg.get(
                "target_semantic_tags", DEFAULT_TARGET_SEMANTIC_TAGS
            )
        )
        self.max_mask_fg_ratio = float(self.cfg.get("max_mask_foreground_ratio", 0.98))
        self.photometric = bool(self.cfg.get("photometric_distortion", True)) and self.split == "train"
        self.seed = int(self.cfg.get("seed", 20260918))

        split_json_value = self.cfg.get("split_json")
        split_json = Path(split_json_value).expanduser().resolve() if split_json_value else None
        raw_frames = resolve_split_frames(self.vehicle_root, self.split, split_json)
        self.frames = [f for f in raw_frames if self._frame_complete(f)]
        missing = len(raw_frames) - len(self.frames)
        if not self.frames:
            raise RuntimeError(f"No usable Griffin frames for split={self.split}")
        if missing:
            print(f"[GriffinP1Dataset] dropped {missing} incomplete frames from {self.split}")

        self.vehicle_calib: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for cam in VEHICLE_CAMS:
            ext, intr = _load_calib(self.vehicle_root, cam)
            assert intr is not None
            self.vehicle_calib[cam] = (ext, intr)
        self.t_lidar_to_ego, _ = _load_calib(self.vehicle_root, "lidar_top")

        self.drone_calib: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for cam in DRONE_CAMS:
            ext, intr = _load_calib(self.drone_root, cam)
            assert intr is not None
            self.drone_calib[cam] = (ext, intr)

        print(
            f"[GriffinP1Dataset] split={self.split} frames={len(self.frames)} "
            f"final={self.final_hw} image_scale={self.image_scale} "
            f"veh_stride={self.vehicle_stride} drone_stride={self.drone_stride} "
            f"photometric={self.photometric}"
        )

    def _frame_complete(self, frame: str) -> bool:
        required = [
            _resolve_lidar_path(self.vehicle_root, frame),
            self.drone_root / "camera" / "bottom" / f"{frame}.png",
            self.drone_root / "camera" / "instance_bottom" / f"{frame}.png",
        ]
        for cam in VEHICLE_CAMS:
            required.append(self.vehicle_root / "camera" / cam / f"{frame}.png")
            required.append(self.vehicle_root / "camera" / f"instance_{cam}" / f"{frame}.png")
        return all(path.exists() for path in required)

    def __len__(self) -> int:
        return len(self.frames)

    def _rng(self, index: int, view_offset: int) -> np.random.RandomState:
        if self.photometric:
            # Each worker has its own numpy RNG state; draw a fresh seed so
            # Griffin photometric distortion changes across epochs/views.
            return np.random.RandomState(np.random.randint(0, 2**31 - 1))
        return np.random.RandomState(self.seed + index * 31 + view_offset * 1009)

    def _load_view(
        self,
        side_root: Path,
        cam: str,
        frame: str,
        stride: int,
        rng: np.random.RandomState,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], np.ndarray, float]:
        rgb_path = side_root / "camera" / cam / f"{frame}.png"
        mask_path = side_root / "camera" / f"instance_{cam}" / f"{frame}.png"

        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(rgb_path)
        original_hw = (int(bgr.shape[0]), int(bgr.shape[1]))
        if self.photometric:
            bgr = photometric_distortion_bgr(bgr.astype(np.float32), rng)
        img_t = _bgr_to_normalized_tensor(
            bgr, self.final_hw, self.image_scale
        )

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        fg = _mask_to_foreground(
            mask,
            self.mask_mode,
            target_semantic_tags=self.target_semantic_tags,
        )
        fg_ratio_raw = float(fg.mean())
        if fg_ratio_raw > self.max_mask_fg_ratio:
            raise ValueError(
                f"Suspicious Griffin instance mask {mask_path}: foreground ratio "
                f"{fg_ratio_raw:.3f} > {self.max_mask_fg_ratio:.3f}. Do NOT train "
                "until mask encoding is checked with check_griffin_p1_data.py."
            )
        final_h, final_w = self.final_hw
        fg_rs = cv2.resize(
            fg.astype(np.uint8), (final_w, final_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        target = _block_any(fg_rs, stride).astype(np.int64)
        return img_t, torch.from_numpy(target).long(), original_hw, fg, fg_ratio_raw

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame = self.frames[index]
        xyz_lidar = _load_lidar_xyz(
            _resolve_lidar_path(self.vehicle_root, frame),
            self.lidar_num_features,
        )

        vehicle_imgs: List[torch.Tensor] = []
        vehicle_hm: List[torch.Tensor] = []
        vehicle_depth: List[torch.Tensor] = []
        vehicle_depth_valid: List[torch.Tensor] = []
        vehicle_mask_ratio: List[float] = []

        for view_idx, cam in enumerate(VEHICLE_CAMS):
            img_t, hm_t, original_hw, fg_mask, fg_ratio = self._load_view(
                self.vehicle_root,
                cam,
                frame,
                self.vehicle_stride,
                self._rng(index, view_idx),
            )
            t_cam_to_ego, k_orig = self.vehicle_calib[cam]
            depth, valid = project_lidar_to_camera_cells(
                xyz_lidar,
                self.t_lidar_to_ego,
                t_cam_to_ego,
                k_orig,
                original_hw,
                self.final_hw,
                self.vehicle_stride,
                fg_mask,
            )
            vehicle_imgs.append(img_t)
            vehicle_hm.append(hm_t)
            vehicle_depth.append(torch.from_numpy(depth).float())
            vehicle_depth_valid.append(torch.from_numpy(valid).bool())
            vehicle_mask_ratio.append(fg_ratio)

        drone_imgs: List[torch.Tensor] = []
        drone_hm: List[torch.Tensor] = []
        drone_mask_ratio: List[float] = []
        for view_idx, cam in enumerate(DRONE_CAMS):
            img_t, hm_t, _original_hw, _fg_mask, fg_ratio = self._load_view(
                self.drone_root,
                cam,
                frame,
                self.drone_stride,
                self._rng(index, 100 + view_idx),
            )
            drone_imgs.append(img_t)
            drone_hm.append(hm_t)
            drone_mask_ratio.append(fg_ratio)

        return {
            "frame_id": frame,
            "vehicle": {
                "imgs": torch.stack(vehicle_imgs, dim=0),
                "heatmap_target": torch.stack(vehicle_hm, dim=0),
                "camera_z_gt": torch.stack(vehicle_depth, dim=0),
                "depth_valid": torch.stack(vehicle_depth_valid, dim=0),
                "mask_fg_ratio": torch.tensor(vehicle_mask_ratio, dtype=torch.float32),
            },
            "drone": {
                "imgs": torch.stack(drone_imgs, dim=0),
                "heatmap_target": torch.stack(drone_hm, dim=0),
                "mask_fg_ratio": torch.tensor(drone_mask_ratio, dtype=torch.float32),
            },
        }
