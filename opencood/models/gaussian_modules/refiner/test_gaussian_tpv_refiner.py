import torch
import torch.nn.functional as F
from gaussian_refiner_query import GaussianTPVRefiner,GaussianDecoder,GaussianAggregator,TPVFeatureFlattener,SparseGaussianSelfAttention,GaussianTPVCrossAttention

# ======== 一些辅助函数 ========

def sample_random_quat(N: int, device):
    """采样随机单位四元数 [N,4]"""
    q = torch.randn(N, 4, device=device)
    q = F.normalize(q, p=2, dim=-1)
    return q

def make_random_gaussians(
    N: int,
    feature_dim: int,
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0],
    semantic_dim: int = 4,
    device: str = "cpu",
):
    """构造一个假的 gaussian dict"""
    pc_range = torch.tensor(pc_range, device=device, dtype=torch.float32)
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range.tolist()

    mu = torch.empty(N, 3, device=device)
    mu[:, 0].uniform_(x_min, x_max)
    mu[:, 1].uniform_(y_min, y_max)
    mu[:, 2].uniform_(z_min, z_max)

    scale = torch.rand(N, 3, device=device) * 2.0 + 0.5  # [0.5, 2.5] 之类
    rotation = sample_random_quat(N, device)
    features = torch.randn(N, feature_dim, device=device)
    semantic = torch.randn(N, semantic_dim, device=device)

    return {
        "mu": mu,
        "scale": scale,
        "rotation": rotation,
        "features": features,
        "semantic": semantic,
    }

def print_gauss_stats(name, gdict):
    print(f"\n[{name}]")
    for k, v in gdict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, mean={v.float().mean().item():.4f}, std={v.float().std().item():.4f}")

# ======== 单模块测试 ========

def test_tpv_flattener(device="cpu"):
    print("==== Test TPVFeatureFlattener ====")
    B = 1
    C = 64
    Hxy, Wxy = 8, 8
    Hxz, Wxz = 6, 10
    Hyz, Wyz = 5, 7

    tpv_xy = torch.randn(B, C, Hxy, Wxy, device=device)
    tpv_xz = torch.randn(B, C, Hxz, Wxz, device=device)
    tpv_yz = torch.randn(B, C, Hyz, Wyz, device=device)

    flattener = TPVFeatureFlattener().to(device)
    value, spatial_shapes, level_start_index = flattener(tpv_xy, tpv_xz, tpv_yz)

    print("value:", value.shape)  # [B, sum(HW), C]
    print("spatial_shapes:", spatial_shapes, spatial_shapes.shape)
    print("level_start_index:", level_start_index, level_start_index.shape)


def test_aggregator(device="cpu"):
    print("\n==== Test GaussianAggregator ====")
    feature_dim = 128
    embed_dims = 256
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]

    N_img, N_lidar = 16, 24
    img_gaussians = make_random_gaussians(N_img, feature_dim, pc_range, device=device)
    lidar_gaussians = make_random_gaussians(N_lidar, feature_dim, pc_range, device=device)

    # query 版本
    img_query_gaussians = make_random_gaussians(N_img, feature_dim, pc_range, device=device)
    lidar_query_gaussians = make_random_gaussians(N_lidar, feature_dim, pc_range, device=device)

    aggregator = GaussianAggregator(
        img_feature_dim=feature_dim,
        lidar_feature_dim=feature_dim,
        output_dim=embed_dims,
        pc_range=pc_range,
        num_learnable_pts=2,  # 随便给个 >0 的看看 learnable corner 部分
    ).to(device)

    # 假设 TPV 各平面 shape
    tpv_spatial_shapes = torch.tensor([[8, 8], [6, 10], [5, 7]], dtype=torch.long, device=device)

    merged, merged_query = aggregator(
        img_gaussians=img_gaussians,
        lidar_gaussians=lidar_gaussians,
        img_query_gaussians=img_query_gaussians,
        lidar_query_gaussians=lidar_query_gaussians,
        tpv_spatial_shapes=tpv_spatial_shapes,
    )

    print_gauss_stats("merged", merged)
    print("  ref_xy:", merged["ref_xy"].shape)
    print("  ref_xz:", merged["ref_xz"].shape)
    print("  ref_yz:", merged["ref_yz"].shape)
    print("  corners_3d:", merged["corners_3d"].shape)

    print_gauss_stats("merged_query", merged_query)


def test_sparse_self_attention(device="cpu"):
    print("\n==== Test SparseGaussianSelfAttention ====")
    embed_dims = 256
    num_heads = 8
    N = 40
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]

    means = torch.empty(N, 3, device=device)
    means[:, 0].uniform_(pc_range[0], pc_range[3])
    means[:, 1].uniform_(pc_range[1], pc_range[4])
    means[:, 2].uniform_(pc_range[2], pc_range[5])

    features = torch.randn(N, embed_dims, device=device)

    # random voxel coords
    voxel_coords = torch.randint(0, 50, (N, 3), device=device)

    # query gaussians (和 normal 不同的一组位置)
    Nq = 30
    query_means = means[:Nq] + torch.randn(Nq, 3, device=device) * 0.5
    query_features = torch.randn(Nq, embed_dims, device=device)

    self_attn = SparseGaussianSelfAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        k_neighbors=8,
        dropout=0.1,
        max_distance=10.0,
        use_voxel_knn=True,  # 这里测试 voxel + cross-faiss
    ).to(device)

    out_g, out_q = self_attn(
        features=features,
        means=means,
        voxel_coords=voxel_coords,
        query_features=query_features,
        query_means=query_means,
        query_voxel_coords=None,
    )

    print("gauss_out:", out_g.shape)
    print("query_out:", out_q.shape)


def test_cross_attention(device="cpu"):
    print("\n==== Test GaussianTPVCrossAttention ====")
    embed_dims = 256
    num_heads = 8
    num_levels = 3
    num_points = 5
    num_anchors = 7  # 假设角点+learnable点

    B = 1
    Nq = 20
    C = embed_dims

    # query
    query = torch.randn(B, Nq, C, device=device)

    # value (TPV flatten 后)
    Hxy, Wxy = 8, 8
    Hxz, Wxz = 6, 10
    Hyz, Wyz = 5, 7
    sum_hw = Hxy * Wxy + Hxz * Wxz + Hyz * Wyz
    value = torch.randn(B, sum_hw, C, device=device)

    spatial_shapes = torch.tensor([[Hxy, Wxy], [Hxz, Wxz], [Hyz, Wyz]], dtype=torch.long, device=device)
    level_start_index = torch.tensor([0,
                                      Hxy * Wxy,
                                      Hxy * Wxy + Hxz * Wxz], dtype=torch.long, device=device)

    # 每个 level 的 reference_points: [B, Nq, A_l, 2]
    ref_xy = torch.rand(B, Nq, num_anchors, 2, device=device)
    ref_xz = torch.rand(B, Nq, num_anchors, 2, device=device)
    ref_yz = torch.rand(B, Nq, num_anchors, 2, device=device)
    ref_list = [ref_xy, ref_xz, ref_yz]

    cross_attn = GaussianTPVCrossAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        fuse="concat",        # 或 "sum"
        num_anchors=num_anchors,
    ).to(device)

    out = cross_attn(
        query=query,
        value=value,
        reference_points_list=ref_list,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    print("cross_attn output:", out.shape)


def test_decoder(device="cpu"):
    print("\n==== Test GaussianDecoder ====")
    embed_dims = 256
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]
    N = 32

    gauss = make_random_gaussians(N, embed_dims, pc_range, device=device)
    query_gauss = make_random_gaussians(N, embed_dims, pc_range, device=device)

    decoder = GaussianDecoder(
        embed_dims=embed_dims,
        pc_range=pc_range,
        scale_range=[0.01, 3.2],
        unit_xyz=[4.0, 4.0, 2.0],
    ).to(device)

    updated_g, updated_q = decoder(gauss, query_gauss)
    print_gauss_stats("updated_normal", updated_g)
    print_gauss_stats("updated_query", updated_q)


# ======== 整体 Refiner 测试 ========

def test_full_refiner(device="cpu"):
    print("\n==== Test Full GaussianTPVRefiner Forward & Backward ====")
    torch.manual_seed(0)

    feature_dim = 128
    tpv_feature_dim = 64
    embed_dims = 256
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]

    # 构造 TPV 特征
    B = 1
    Hxy, Wxy = 8, 8
    Hxz, Wxz = 6, 10
    Hyz, Wyz = 5, 7

    tpv_features = {
        "xy": torch.randn(B, tpv_feature_dim, Hxy, Wxy, device=device),
        "xz": torch.randn(B, tpv_feature_dim, Hxz, Wxz, device=device),
        "yz": torch.randn(B, tpv_feature_dim, Hyz, Wyz, device=device),
    }

    # 构造 img / lidar 高斯 + query 高斯
    N_img, N_lidar = 20, 30
    img_gaussians = make_random_gaussians(N_img, feature_dim, pc_range, device=device)
    lidar_gaussians = make_random_gaussians(N_lidar, feature_dim, pc_range, device=device)

    img_query_gaussians = make_random_gaussians(N_img, feature_dim, pc_range, device=device)
    lidar_query_gaussians = make_random_gaussians(N_lidar, feature_dim, pc_range, device=device)

    # 构造 refiner
    refiner = GaussianTPVRefiner(
        feature_dim=feature_dim,
        tpv_feature_dim=tpv_feature_dim,
        embed_dims=embed_dims,
        num_heads=8,
        num_layers=1,
        k_neighbors=8,
        pc_range=pc_range,
        num_points=5,
        num_learnable_pts=2,
        max_distance=10.0,
        scale_range=[0.01, 3.2],
        unit_xyz=[4.0, 4.0, 2.0],
        dropout=0.1,
        use_voxel_knn=True,
        voxel_size=[0.4, 0.4, 0.125],
    ).to(device)

    # 前向
    updated_g, updated_q = refiner(
        tpv_features=tpv_features,
        img_gaussians=img_gaussians,
        lidar_gaussians=lidar_gaussians,
        img_query_gaussians=img_query_gaussians,
        lidar_query_gaussians=lidar_query_gaussians,
    )

    print_gauss_stats("updated_gaussians", updated_g)
    print_gauss_stats("updated_query_gaussians", updated_q)

    # 简单做个 backward，检查梯度是否能正常回传
    loss = updated_g["features"].pow(2).mean() + updated_q["features"].pow(2).mean()
    loss.backward()

    # 打印一两个参数的 grad 情况
    some_param = None
    for name, p in refiner.named_parameters():
        if p.requires_grad and p.grad is not None:
            some_param = (name, p)
            break

    if some_param is not None:
        name, p = some_param
        print(f"\nGradient check: param '{name}' grad mean = {p.grad.float().mean().item():.6f}")
    else:
        print("No parameter has grad, something is wrong.")
    name, p = some_param
    g = p.grad.float()
    print(f"param '{name}' grad mean = {g.mean().item():.6e}")
    print(f"param '{name}' grad abs mean = {g.abs().mean().item():.6e}")
    print(f"param '{name}' grad max = {g.abs().max().item():.6e}")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    test_tpv_flattener(device)
    test_aggregator(device)
    test_sparse_self_attention(device)
    test_cross_attention(device)
    test_decoder(device)
    test_full_refiner(device)

    print("\nAll tests finished.")
