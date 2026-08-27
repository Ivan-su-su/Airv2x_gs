import torch
import torch.nn.functional as F
from backbone2d_semantic import OptimizedLSSBasedTPVGeneratorV2


def test_tpv_builders_consistency(device=None, verbose=True, atol=1e-5):
    """
    对比 _build_tpv_from_lss (v1) 和 _build_tpv_from_lss_v2 (v2) 的数值一致性。

    - 只测 TPV 投影本身：给两个函数同样的 image_feat / depth_prob / world_coords，
      看三张平面 xy/xz/yz 的输出是否在数值上几乎一致。
    - 不涉及 _compute_world_coords，世界坐标用随机数即可，
      只要两个函数看到的是同一组 world_coords，它们的行为就应该一样。
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========= 1. 构建一个小配置的模型实例 =========
    model_cfg = {
        "TPV_FEATURES": 64,
        "TPV_SIZE": [20, 40, 8],        # 小一点的 tpv_size，方便测试
        "PC_RANGE": [-10.0, -10.0, -3.0, 10.0, 10.0, 3.0],
        "VOXEL_SIZE": [1.0, 1.0, 0.5],
        "NUM_CLASSES": 4,
        "EMPTY_CLASS_INDEX": 0,
        "DEPTH_BINS": 16,              # 测试用深度bin也可以小一点
        "DBOUND": [2.0, 30.0, 2.0],
        "TOP_K_DEPTHS": 8,
        "GAUSSIAN_THRESHOLD": 0.1,
        "GAUSSIAN_SCALE_RANGE": [0.01, 3.2],
        "IMAGE_FEATURES": 32,          # 测试用 feature 维度也降一点
    }

    model = OptimizedLSSBasedTPVGeneratorV2(model_cfg).to(device)
    model.eval()

    # ========= 2. 随机造一批输入（只给 TPV 用到的量） =========
    B, N = 2, 3
    C = model.image_channels         # 32
    D = model.depth_bins             # 16
    H, W = 6, 8                      # 随便来个小 feature map

    # image_feat: [B, N, C, H, W]
    image_feat = torch.randn(B, N, C, H, W, device=device)

    # depth_prob: [B, N, D, H, W]  用 softmax 保证每条 ray 上是一个分布
    depth_logits = torch.randn(B, N, D, H, W, device=device)
    depth_prob = F.softmax(depth_logits, dim=2)

    # world_coords: [B, N, D, H, W, 3]
    # 这里直接随机生成，范围控制在 pc_range 附近即可
    pc_min = torch.tensor(model.pc_range[:3], device=device)
    pc_max = torch.tensor(model.pc_range[3:], device=device)
    world_coords = torch.empty(B, N, D, H, W, 3, device=device)
    world_coords[..., 0] = torch.rand(B, N, D, H, W, device=device) * (pc_max[0] - pc_min[0]) + pc_min[0]  # x
    world_coords[..., 1] = torch.rand(B, N, D, H, W, device=device) * (pc_max[1] - pc_min[1]) + pc_min[1]  # y
    world_coords[..., 2] = torch.rand(B, N, D, H, W, device=device) * (pc_max[2] - pc_min[2]) + pc_min[2]  # z

    # ========= 3. 分别调用 v1 和 v2 =========
    with torch.no_grad():
        tpv_v1 = model._build_tpv_from_lss(image_feat, depth_prob, world_coords)
        tpv_v2 = model._build_tpv_from_lss_v2(image_feat, depth_prob, world_coords)

    # ========= 4. 对比三张平面 xy/xz/yz =========
    all_ok = True
    for plane in ["xy", "xz", "yz"]:
        diff = (tpv_v1[plane] - tpv_v2[plane]).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        if verbose:
            print(f"[TPV TEST] Plane {plane}: max_err={max_err:.6e}, mean_err={mean_err:.6e}")
        if max_err > atol:
            all_ok = False
            print(f"[TPV TEST][WARN] Plane {plane} mismatch exceeds tolerance {atol}!")

    if all_ok:
        print(f"[TPV TEST] SUCCESS: v1 和 v2 在容差 {atol} 内一致 ✅")
    else:
        print(f"[TPV TEST] FAIL: 至少一个平面超过容差 {atol} ❌")

    return all_ok


if __name__ == "__main__":
    # 直接跑文件时执行一次测试
    test_tpv_builders_consistency()
