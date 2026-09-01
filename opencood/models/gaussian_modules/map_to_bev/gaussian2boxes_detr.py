"""
Gaussian-to-DETR Detection Module

Goal: Convert normal Gaussian sets and query Gaussian sets into DETR-style detection outputs
using voxel-based neighbor association and kernel cross-attention.

Process:
1. Apply ContinuousGaussianVFE to both normal and query Gaussians
2. Build Query→Normal neighbor relationships based on voxel hash
3. Embed Gaussians (geometry + features) into unified d_model space
4. Apply Gaussian kernel cross-attention (with geometric bias)
5. Predict class logits and box parameters via DETR head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
from torch_scatter import scatter_add, scatter_mean, scatter_max

class ContinuousGaussianVFE(nn.Module):
    """
    不降采样的连续体素聚合（MeanVFE 思想 + 连续中心）
    - 将高斯按离散体素 (b,z,y,x) 分组
    - 对每组做：mu/scale/feature 的 mean 或 sum；rotation 做四元数稳健池化
    - 代表高斯的中心 mu 为“连续均值”，非栅格中心
    - 同时返回整数 voxel_coords 以供后续 BEVScatter 使用
    """
    def __init__(
        self,
        feature_dim: int,
        voxel_size: tuple,                     # (vx, vy, vz)
        grid_size: tuple,                      # (H, W, D)  注意：x→H, y→W, z→D 的一致性
        pc_range: tuple,                       # (xmin, ymin, zmin, xmax, ymax, zmax)
        aggregation: str = "mean",             # "mean" | "sum"
        quat_pooling: str = "antipodal",       # "antipodal" | "markley"
        return_abs_height: bool = True
    ):
        super().__init__()
        assert aggregation in ("mean", "sum")
        self.feature_dim = feature_dim
        self.H, self.W, self.D = grid_size
        self.aggregation = aggregation
        self.quat_pooling = quat_pooling
        self.return_abs_height = return_abs_height

        # buffers
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer("pc_min", torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer("pc_max", torch.tensor(pc_range[3:], dtype=torch.float32))

        # 线性编码用的尺度
        self.scale_xyz = self.H * self.W * self.D
        self.scale_yz  = self.W * self.D
        self.scale_z   = self.D

    @torch.no_grad()
    def _discrete_voxel_indices(self, mu: torch.Tensor, bidx: torch.Tensor):
        """
        连续坐标 → 整数体素索引 (b,z,y,x) 以及有效掩码
        mu: [N,3] in (x,y,z) world
        """
        idx_xyz = torch.floor((mu - self.pc_min) / self.voxel_size).long()  # [N,3] (x,y,z) -> (ix,iy,iz)
        valid = (
            (idx_xyz[:, 0] >= 0) & (idx_xyz[:, 0] < self.H) &
            (idx_xyz[:, 1] >= 0) & (idx_xyz[:, 1] < self.W) &
            (idx_xyz[:, 2] >= 0) & (idx_xyz[:, 2] < self.D)
        )
        voxel_bzyx = torch.stack(
            (bidx,
             idx_xyz[:, 2],   # z
             idx_xyz[:, 1],   # y
             idx_xyz[:, 0]),  # x
            dim=1
        )  # [N,4]
        return voxel_bzyx, valid

    def _group_code(self, voxel_bzyx: torch.Tensor):
        """(b,z,y,x) → 唯一线性编码"""
        b, z, y, x = voxel_bzyx.unbind(1)
        code = b.long() * self.scale_xyz + z.long() * self.scale_yz + y.long() * self.scale_z + x.long()
        return code  # [N]

    def _quat_pool(self, q: torch.Tensor, inv: torch.Tensor):
        """
        组内四元数池化
        q:   [N,4]
        inv: [N]  每个元素属于第 inv[i] 组
        """
        if self.quat_pooling == "antipodal":
            # 以每组第一个为参考，反极性对齐后 group-mean
            unq = torch.unique(inv)
            # 取各组第一个索引
            ref_idx = torch.zeros_like(unq)
            for i, gid in enumerate(unq):
                ref_idx[i] = torch.nonzero(inv == gid, as_tuple=False)[0, 0]
            ref = q[ref_idx[inv]]                          # [N,4]
            dot = (q * ref).sum(-1, keepdim=True)
            aligned = q * torch.sign(dot + 1e-8)           # 反极性
            pooled = scatter_mean(aligned, inv, dim=0)
            return F.normalize(pooled, p=2, dim=1)
        else:
            # 简化版 Markley：直接均值后归一化（需要更稳健可改为特征向量法）
            pooled = scatter_mean(q, inv, dim=0)
            return F.normalize(pooled, p=2, dim=1)

    def forward(self, gaussians: dict) -> dict:
        """
        Input:
            gaussians = {
              'mu': [N,3], 'scale':[N,3], 'rotation':[N,4], 'features':[N,C],
              'batch_idx': [N] (可选，缺省全0)
            }
        Output:
            {
              'mu': [M,3]            # 连续中心（组内 μ 的均值）
              'scale': [M,3]
              'rotation': [M,4]
              'features': [M,C]
              'voxel_coords': [M,4]  # (b,z,y,x) 整数体素索引，供 BEV 使用
              'height_map': [M]      # 可选：连续高度（mu_z）
            }
        """
        mu   = gaussians["mu"]
        sc   = gaussians["scale"]
        rot  = gaussians["rotation"]
        feat = gaussians["features"]
        semantic = gaussians["semantic"]
        bidx = gaussians.get("batch_idx", torch.zeros(mu.size(0), device=mu.device, dtype=torch.long))

        # 1) 连续 → 离散体素索引
        voxel_bzyx, valid = self._discrete_voxel_indices(mu, bidx)
        if not valid.any():
            empty = dict(
                mu=mu.new_zeros((0, 3)), scale=sc.new_zeros((0, 3)),
                rotation=rot.new_zeros((0, 4)), features=feat.new_zeros((0, self.feature_dim)),
            )
            if self.return_abs_height: empty["height_map"] = mu.new_zeros((0,))
            empty_voxel_coords = voxel_bzyx.new_zeros((0, 4))
            return empty, empty_voxel_coords

        mu, sc, rot, feat, semantic, voxel_bzyx = mu[valid], sc[valid], rot[valid], feat[valid], semantic[valid], voxel_bzyx[valid]

        # 2) 分组（同一 voxel 的点归为一组）
        code = self._group_code(voxel_bzyx)                        # [N]
        unq_code, inv = torch.unique(code, return_inverse=True)    # inv: N→M
        M = unq_code.numel()

        # 3) 连续中心与参数聚合
        if self.aggregation == "mean":
            mu_out   = scatter_mean(mu,   inv, dim=0)        # [M,3]
            sc_out   = scatter_mean(sc,   inv, dim=0)        # [M,3]
            feat_out = scatter_mean(feat, inv, dim=0)        # [M,C]
            semantic_out = scatter_mean(semantic, inv, dim=0)        # [M,2]
        else:
            mu_out   = scatter_add(mu,   inv, dim=0) / scatter_add(torch.ones_like(mu[:, :1]), inv, dim=0)
            sc_out   = scatter_add(sc,   inv, dim=0) / scatter_add(torch.ones_like(sc[:, :1]), inv, dim=0)
            feat_out = scatter_add(feat, inv, dim=0)  
            semantic_out = scatter_add(semantic, inv, dim=0) / scatter_add(torch.ones_like(semantic[:, :1]), inv, dim=0)
        rot_out = self._quat_pool(rot, inv)                                  # [M,4]

        # 4) 还原整数体素坐标 (b,z,y,x) 供 BEV 使用
        unq_code = unq_code.long()
        vb = unq_code // self.scale_xyz
        rem = unq_code %  self.scale_xyz
        vz  = rem // self.scale_yz
        rem = rem %  self.scale_yz
        vy  = rem // self.scale_z
        vx  = rem %  self.scale_z
        voxel_coords = torch.stack([vb, vz, vy, vx], dim=1)                  # [M,4]
        
        gaussians = dict(
            mu=mu_out, scale=sc_out, rotation=rot_out, features=feat_out, semantic=semantic_out,
        )                 # 连续高度（可用于额外监督/可视化）

        return gaussians,voxel_coords

class GaussianEmbedding(nn.Module):
    """
    高斯嵌入模块：将高斯的几何参数和特征参数映射到统一的 d_model 维度

    功能：
    - 几何编码：mu (3) + log(scale) (3) + rotation (4) → [10] → d_model
    - 属性编码：features (C_feat) + semantic (C_sem) → d_model
    - 最终嵌入 = 几何嵌入 + 属性嵌入（残差式相加）
    """
    def __init__(self, feature_dim: int, semantic_dim: int, d_model: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.d_model = d_model

        # 几何编码：mu(3) + log_scale(3) + rotation(4) = 10 维
        self.geom_fc = nn.Linear(10, d_model, bias=True)

        # 属性编码：features + semantic
        attr_dim = feature_dim + semantic_dim
        self.attr_fc = nn.Linear(attr_dim, d_model, bias=True)

    def forward(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Input:
            gaussians: dict, 包含 keys:
                'mu': [N, 3]           # 高斯中心
                'scale': [N, 3]        # 高斯尺度
                'rotation': [N, 4]     # 四元数旋转
                'features': [N, feature_dim]      # 特征向量
                'semantic': [N, semantic_dim]     # 语义特征
        Return:
            emb: [N, d_model]          # 统一嵌入向量
        """
        mu = gaussians['mu']                    # [N, 3]
        scale = gaussians['scale']              # [N, 3]
        rotation = gaussians['rotation']        # [N, 4]
        features = gaussians['features']        # [N, feature_dim]
        semantic = gaussians['semantic']        # [N, semantic_dim]

        # 几何向量：mu + log(scale + eps) + rotation
        log_scale = torch.log(scale + 1e-3)     # [N, 3]
        geom = torch.cat([mu, log_scale, rotation], dim=-1)  # [N, 10]

        # 属性向量：features + semantic
        attr = torch.cat([features, semantic], dim=-1)       # [N, feature_dim + semantic_dim]

        # 分别映射到 d_model 并相加
        geom_emb = self.geom_fc(geom)           # [N, d_model]
        attr_emb = self.attr_fc(attr)           # [N, d_model]
        emb = geom_emb + attr_emb               # [N, d_model]

        return emb


class QueryToNormalVoxelNeighborBuilder(nn.Module):
    """
    基于 voxel hash 的 Query→Normal 邻居构建器

    功能：
    - 对 normal pooled gaussians 的 voxel_coords 建立整数编码索引
    - 对每个 query pooled gaussian，基于其 voxel_coords 在局部窗口内查找 normal 邻居
    - 在候选 normal 中，根据 Mahalanobis 距离 d2 选出 top-K 邻居

    输出：
        neighbor_info = {
            'neighbor_indices': [Q_p, K] (long),    # normal 的索引，无效为 -1
            'neighbor_d2':      [Q_p, K] (float),   # Mahalanobis 距离平方
            'neighbor_masks':   [Q_p, K] (bool)     # 有效邻居掩码
        }
    """
    def __init__(
        self,
        grid_size: Tuple[int, int, int],   # (H, W, D)
        max_neighbors: int = 64,
        window_size: int = 1               # 查询窗口半径，如 1 表示 [-1,0,1]
    ):
        super().__init__()
        self.H, self.W, self.D = grid_size
        self.max_neighbors = max_neighbors
        self.window_size = window_size

        # 用于线性编码的尺度
        self.scale_xyz = self.H * self.W * self.D
        self.scale_yz = self.W * self.D
        self.scale_z = self.D

    def _voxel_code(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        """
        将体素坐标 (b,z,y,x) 编码为唯一整数
        voxel_coords: [M, 4]
        Return: [M] 整数编码
        """
        b, z, y, x = voxel_coords.unbind(dim=1)
        code = b.long() * self.scale_xyz + z.long() * self.scale_yz + y.long() * self.scale_z + x.long()
        return code

    @staticmethod
    def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """
        四元数转旋转矩阵
        q: [N,4] (w,x,y,z)
        Return:
            R: [N,3,3]
        """
        q = F.normalize(q, p=2, dim=-1)
        w, x, y, z = q.unbind(-1)
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
            2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
            2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
        ], dim=-1).reshape(-1, 3, 3)
        return R

    @torch.no_grad()
    def forward(
        self,
        pooled_normal: Dict[str, torch.Tensor],
        voxel_coords_normal: torch.Tensor,   # [N_p, 4]
        pooled_query: Dict[str, torch.Tensor],
        voxel_coords_query: torch.Tensor     # [Q_p, 4]
    ) -> Dict[str, torch.Tensor]:
        """
        Input:
            pooled_normal: dict, 包含 'mu' 等
            voxel_coords_normal: [N_p, 4] = (b, z, y, x)
            pooled_query: dict, 包含 'mu' 等
            voxel_coords_query: [Q_p, 4] = (b, z, y, x)
        Return:
            neighbor_info: dict, 包含 neighbor_indices, neighbor_d2, neighbor_masks
        """
        device = voxel_coords_query.device
        Q_p = voxel_coords_query.size(0)
        N_p = voxel_coords_normal.size(0)

        # 预先计算 normal 的几何参数（用于 Mahalanobis 距离）
        mu_normal = pooled_normal['mu'].to(device)        # [N_p, 3]
        scale_normal = pooled_normal['scale'].to(device)  # [N_p, 3]
        rot_normal = pooled_normal['rotation'].to(device) # [N_p, 4]
        R_normal = self._quat_to_rotmat(rot_normal)       # [N_p, 3, 3]

        mu_query = pooled_query['mu'].to(device)          # [Q_p, 3]

        # 1) 对 normal 侧构建整数编码并排序（用于快速查找）
        code_normal = self._voxel_code(voxel_coords_normal)  # [N_p]
        code_sorted, idx_sorted = code_normal.sort()         # [N_p], [N_p]

        # 2) 构建 code -> [normal_idx,...] 的映射，避免重复 searchsorted
        code_to_indices: Dict[int, list] = {}
        for k in range(N_p):
            c = int(code_sorted[k].item())
            idx = int(idx_sorted[k].item())
            if c not in code_to_indices:
                code_to_indices[c] = [idx]
            else:
                code_to_indices[c].append(idx)

        # 3) 初始化输出
        neighbor_indices = torch.full((Q_p, self.max_neighbors), -1, device=device, dtype=torch.long)
        neighbor_d2 = torch.zeros((Q_p, self.max_neighbors), device=device, dtype=torch.float32)
        neighbor_masks = torch.zeros((Q_p, self.max_neighbors), device=device, dtype=torch.bool)

        # 4) 对每个 query 查找邻居
        b_q, z_q, y_q, x_q = voxel_coords_query.unbind(dim=1)  # [Q_p] each

        for q_idx in range(Q_p):
            bq, zq, yq, xq = b_q[q_idx].item(), z_q[q_idx].item(), y_q[q_idx].item(), x_q[q_idx].item()
            mu_q = mu_query[q_idx:q_idx+1]  # [1, 3]

            # 在窗口内枚举所有可能的体素坐标
            candidates = []
            for dz in range(-self.window_size, self.window_size + 1):
                for dy in range(-self.window_size, self.window_size + 1):
                    for dx in range(-self.window_size, self.window_size + 1):
                        zn, yn, xn = zq + dz, yq + dy, xq + dx
                        # 检查边界
                        if (0 <= zn < self.D and 0 <= yn < self.W and 0 <= xn < self.H):
                            # 编码该体素
                            code_candidate = bq * self.scale_xyz + zn * self.scale_yz + yn * self.scale_z + xn
                            # 使用 hash map 查找（code 已包含 batch 维度，无需额外检查）
                            idx_list = code_to_indices.get(int(code_candidate), None)
                            if idx_list is not None:
                                candidates.extend(idx_list)

            if len(candidates) == 0:
                continue

            # 5) 计算候选 normal 的 Mahalanobis 距离
            candidate_tensor = torch.tensor(candidates, device=device, dtype=torch.long)  # [C]
            mu_candidates = mu_normal[candidate_tensor]          # [C, 3]
            scale_cand = scale_normal[candidate_tensor]          # [C, 3]
            R_cand = R_normal[candidate_tensor]                  # [C, 3, 3]

            # delta = mu_candidates - mu_q (广播)
            delta = mu_candidates - mu_q                         # [C, 3]

            # delta_local = R^T * delta
            R_cand_T = R_cand.transpose(1, 2)                    # [C, 3, 3]
            delta_local = torch.einsum('cij,cj->ci', R_cand_T, delta)  # [C, 3]

            # Mahalanobis 距离平方
            d2_candidates = (delta_local ** 2 / (scale_cand ** 2 + 1e-8)).sum(dim=-1)  # [C]

            # 6) 取 top-K（距离最小的 K 个）
            K_actual = min(self.max_neighbors, len(candidates))
            topk_values, topk_indices = torch.topk(d2_candidates, k=K_actual, largest=False)  # [K_actual]

            # 填充到输出
            neighbor_indices[q_idx, :K_actual] = candidate_tensor[topk_indices]
            neighbor_d2[q_idx, :K_actual] = topk_values
            neighbor_masks[q_idx, :K_actual] = True

        return {
            'neighbor_indices': neighbor_indices,  # [Q_p, K]
            'neighbor_d2': neighbor_d2,            # [Q_p, K]
            'neighbor_masks': neighbor_masks       # [Q_p, K]
        }


class GaussianKernelCrossAttention(nn.Module):
    """
    基于高斯核的局部 Cross-Attention

    功能：
    - Query: pooled query gaussians embedding E_q
    - Key/Value: pooled normal gaussians embedding E_n
    - 范围：由 neighbor_info 指定的局部邻域（每个 query 对应 K 个 normal）
    - logits = (Q·K) / sqrt(d) - gamma * d2  （加入几何距离 bias）

    注意：当前实现为单头版本，nhead 参数预留用于后续扩展
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        geom_gamma: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.geom_gamma = geom_gamma

        # Q/K/V 投影（单头实现，后续可扩展为多头）
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        E_q: torch.Tensor,                    # [Q_p, d_model]
        E_n: torch.Tensor,                    # [N_p, d_model]
        neighbor_info: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            E_q: [Q_p, d_model]              # Query 嵌入
            E_n: [N_p, d_model]              # Normal 嵌入
            neighbor_info:
                'neighbor_indices': [Q_p, K]  # normal 索引
                'neighbor_d2':      [Q_p, K]  # 几何距离平方
                'neighbor_masks':   [Q_p, K]  # 有效掩码
        Return:
            query_out: [Q_p, d_model]        # 增强后的 query 特征
            attn:      [Q_p, K]              # 注意力权重（便于可视化/调试）
        """
        Q_p, d = E_q.shape
        K = neighbor_info['neighbor_indices'].size(1)

        neighbor_indices = neighbor_info['neighbor_indices']  # [Q_p, K]
        neighbor_d2 = neighbor_info['neighbor_d2']            # [Q_p, K]
        neighbor_masks = neighbor_info['neighbor_masks']      # [Q_p, K]

        # 1) Q/K/V 投影
        Qp = self.q_proj(E_q)  # [Q_p, d_model]

        # 2) 从 E_n 中 gather 出局部 K/V（处理 -1 索引）
        safe_idx = torch.where(neighbor_masks, neighbor_indices, torch.zeros_like(neighbor_indices))
        safe_idx = torch.clamp(safe_idx, min=0, max=E_n.size(0) - 1)  # 防止越界
        K_local = E_n[safe_idx]  # [Q_p, K, d_model]
        V_local = E_n[safe_idx]  # [Q_p, K, d_model]

        Kp = self.k_proj(K_local)  # [Q_p, K, d_model]
        Vp = self.v_proj(V_local)  # [Q_p, K, d_model]

        # 3) 计算注意力 logits：特征相似度 - 几何距离 bias
        logits_feat = (Qp.unsqueeze(1) * Kp).sum(dim=-1) / math.sqrt(d)  # [Q_p, K]
        logits = logits_feat - self.geom_gamma * neighbor_d2              # [Q_p, K]

        # 4) 掩码无效邻居
        logits = logits.masked_fill(~neighbor_masks, float('-inf'))

        # 5) 检查哪些 query 有邻居，避免 softmax(all -inf) → NaN
        no_neighbor = ~neighbor_masks.any(dim=1)   # [Q_p]，完全没有邻居的 query
        has_neighbor = ~no_neighbor                # [Q_p]，有至少一个邻居的 query

        # 初始化注意力为全 0
        attn = torch.zeros_like(logits)  # [Q_p, K]

        # 只对有邻居的行做 softmax
        if has_neighbor.any():
            attn_valid = torch.softmax(logits[has_neighbor], dim=-1)  # [Q_valid, K]
            attn[has_neighbor] = attn_valid
        # 对于 no_neighbor 行，保持 attn 为全 0（不进行 softmax）

        # 6) 加权聚合 Value
        out = torch.einsum('qk,qkd->qd', attn, Vp)  # [Q_p, d_model]

        # 7) 加上 Query 残差
        out = out + E_q

        # 8) 输出投影
        out = self.out_proj(out)  # [Q_p, d_model]

        return out, attn


class GaussianDetrHead(nn.Module):
    """
    简单 DETR 风格的检测 head

    功能：
    - 输入：每个 query 的 embedding [Q_p, d_model]
    - 输出：类别 logits + box 参数

    结构：
    - class_head: Linear → ReLU → Linear(num_classes)
    - box_head: Linear → ReLU → Linear(box_dim)
    """
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        box_dim: int = 7   # 例如 (cx, cy, cz, w, h, l, yaw)
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.box_dim = box_dim

        # 类别预测头
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_classes, bias=True)
        )

        # 框预测头
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, box_dim, bias=True)
        )

    def forward(self, query_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            query_feat: [Q_p, d_model]      # Query 特征
        Return:
            cls_logits: [Q_p, num_classes]  # 类别 logits
            box_pred:   [Q_p, box_dim]      # 框参数预测
        """
        cls_logits = self.class_head(query_feat)  # [Q_p, num_classes]
        box_pred = self.box_head(query_feat)      # [Q_p, box_dim]

        return cls_logits, box_pred


class GaussianToBoxesDETR(nn.Module):
    """
    主模块：高斯到 DETR 的检测分支

    功能：
    - 对 normal 和 query 高斯分别进行体素聚合（ContinuousGaussianVFE）
    - 构建 Query→Normal 的邻居关系（基于 voxel hash）
    - 将高斯嵌入到统一空间（GaussianEmbedding）
    - 执行局部 cross-attention（GaussianKernelCrossAttention）
    - 预测类别和框参数（GaussianDetrHead）

    输入：
        normal_gaussians: dict
            {
              'mu': [N,3],
              'scale':[N,3],
              'rotation':[N,4],
              'features':[N,C_feat],
              'semantic':[N,C_sem],
              'batch_idx':[N] (optional)
            }
        query_gaussians: dict
            结构同上，但 semantic 通常为 0

    输出：
        outputs = {
            'cls_logits': [Q_p, num_classes],
            'box_pred':   [Q_p, box_dim],
            'intermediates': {
                'pooled_normal': dict,
                'pooled_query': dict,
                'voxel_coords_normal': [N_p,4],
                'voxel_coords_query': [Q_p,4],
                'neighbor_info': dict,
                'attn': [Q_p, K],
            }
        }
    """
    def __init__(
        self,
        model_cfg: Dict,
        voxel_size: Tuple[float, float, float],
        grid_size: Tuple[int, int, int],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        feature_dim: int,
        semantic_dim: int,
        num_classes: int,
        box_dim: int = 7,
        d_model: int = 256,
        max_neighbors: int = 64,
        window_size: int = 1,
        **kwargs
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.num_classes = num_classes
        self.box_dim = box_dim
        self.d_model = d_model
        self.max_neighbors = max_neighbors
        self.window_size = window_size

        # ---------------- 1) 连续体素聚合（复用 ContinuousGaussianVFE） ----------------
        self.vfe = ContinuousGaussianVFE(
            feature_dim=feature_dim,
            voxel_size=voxel_size,
            grid_size=grid_size,
            pc_range=point_cloud_range,
            aggregation=model_cfg.get('VFE_AGGREGATION', 'mean'),
            quat_pooling=model_cfg.get('QUAT_POOLING', 'antipodal'),
            return_abs_height=False  # DETR 分支不需要高度
        )

        # ---------------- 2) Query→Normal 邻居构建器 ----------------
        self.neighbor_builder = QueryToNormalVoxelNeighborBuilder(
            grid_size=grid_size,
            max_neighbors=max_neighbors,
            window_size=window_size
        )

        # ---------------- 3) 高斯嵌入模块 ----------------
        self.embedding = GaussianEmbedding(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            d_model=d_model
        )

        # ---------------- 4) 高斯核 Cross-Attention ----------------
        self.cross_attn = GaussianKernelCrossAttention(
            d_model=d_model,
            nhead=model_cfg.get('NHEAD', 4),
            geom_gamma=model_cfg.get('GEOM_GAMMA', 1.0)
        )

        # ---------------- 5) DETR 检测头 ----------------
        self.head = GaussianDetrHead(
            d_model=d_model,
            num_classes=num_classes,
            box_dim=box_dim
        )

    def forward(
        self,
        normal_gaussians: Dict[str, torch.Tensor],
        query_gaussians: Dict[str, torch.Tensor],
        batch_dict: Optional[Dict] = None
    ) -> Dict[str, object]:
        """
        Returns:
            outputs: dict, 包含 'cls_logits', 'box_pred', 'intermediates'
        """
        intermediates = {}

        # 1) 分别对 normal / query 调用 VFE
        pooled_normal, voxel_coords_normal = self.vfe(normal_gaussians)
        pooled_query, voxel_coords_query = self.vfe(query_gaussians)

        intermediates['pooled_normal'] = pooled_normal
        intermediates['pooled_query'] = pooled_query
        intermediates['voxel_coords_normal'] = voxel_coords_normal
        intermediates['voxel_coords_query'] = voxel_coords_query

        # 2) 构建 Query→Normal 邻居关系
        neighbor_info = self.neighbor_builder(
            pooled_normal=pooled_normal,
            voxel_coords_normal=voxel_coords_normal,
            pooled_query=pooled_query,
            voxel_coords_query=voxel_coords_query
        )
        intermediates['neighbor_info'] = neighbor_info

        # 3) 将高斯嵌入到统一空间
        E_n = self.embedding(pooled_normal)  # [N_p, d_model]
        E_q = self.embedding(pooled_query)   # [Q_p, d_model]

        # 4) 执行局部 cross-attention
        query_out, attn = self.cross_attn(
            E_q=E_q,
            E_n=E_n,
            neighbor_info=neighbor_info
        )
        intermediates['attn'] = attn  # [Q_p, K]

        # 5) 预测类别和框参数
        cls_logits, box_pred = self.head(query_out)

        # 6) 组装输出
        outputs = {
            'cls_logits': cls_logits,      # [Q_p, num_classes]
            'box_pred': box_pred,          # [Q_p, box_dim]
            'intermediates': intermediates
        }

        return outputs

def _create_random_gaussians(
    num_gaussians: int,
    batch_size: int,
    feature_dim: int,
    semantic_dim: int,
    pc_range: Tuple[float, float, float, float, float, float],
    device: str = "cpu",
    is_query: bool = False,
):
    """
    构造一批随机高斯，用于单元测试：
    - mu 均匀采样在 point_cloud_range 内
    - scale 在一个合理正范围内
    - rotation 随机四元数并归一化
    - features 随机
    - semantic: normal 随机，query 置 0
    - batch_idx: 随机分配到 [0, batch_size)
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    mu = torch.empty(num_gaussians, 3, device=device)
    mu[:, 0] = torch.rand(num_gaussians, device=device) * (x_max - x_min) + x_min
    mu[:, 1] = torch.rand(num_gaussians, device=device) * (y_max - y_min) + y_min
    mu[:, 2] = torch.rand(num_gaussians, device=device) * (z_max - z_min) + z_min

    # scale > 0
    scale = 0.5 + torch.rand(num_gaussians, 3, device=device)  # [0.5, 1.5]

    # 随机四元数
    rot = torch.rand(num_gaussians, 4, device=device) - 0.5
    rot = F.normalize(rot, p=2, dim=-1)

    features = torch.randn(num_gaussians, feature_dim, device=device)

    if is_query:
        semantic = torch.zeros(num_gaussians, semantic_dim, device=device)
    else:
        semantic = torch.randn(num_gaussians, semantic_dim, device=device)

    batch_idx = torch.randint(0, batch_size, (num_gaussians,), device=device, dtype=torch.long)

    gaussians = {
        "mu": mu,
        "scale": scale,
        "rotation": rot,
        "features": features,
        "semantic": semantic,
        "batch_idx": batch_idx,
    }
    return gaussians


def _compute_grid_size(voxel_size, pc_range):
    """
    根据 voxel_size 和 point_cloud_range 计算 (H, W, D)
    约定：x -> H, y -> W, z -> D
    """
    vx, vy, vz = voxel_size
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    H = int((x_max - x_min) / vx)
    W = int((y_max - y_min) / vy)
    D = int((z_max - z_min) / vz)
    return (H, W, D)





def test_gaussian_to_detr(device: str = "cpu"):
    torch.manual_seed(42)

    # 基本配置
    batch_size = 2
    num_normal = 200
    num_query = 64
    feature_dim = 16
    semantic_dim = 4
    d_model = 64
    num_classes = 10
    box_dim = 7

    voxel_size = (0.5, 0.5, 0.5)
    point_cloud_range = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)
    grid_size = _compute_grid_size(voxel_size, point_cloud_range)

    model_cfg = {
        "VFE_AGGREGATION": "mean",
        "QUAT_POOLING": "antipodal",
        "NHEAD": 4,
        "GEOM_GAMMA": 1.0,
    }

    # 构造随机高斯
    normal_gaussians = _create_random_gaussians(
        num_gaussians=num_normal,
        batch_size=batch_size,
        feature_dim=feature_dim,
        semantic_dim=semantic_dim,
        pc_range=point_cloud_range,
        device=device,
        is_query=False,
    )
    query_gaussians = _create_random_gaussians(
        num_gaussians=num_query,
        batch_size=batch_size,
        feature_dim=feature_dim,
        semantic_dim=semantic_dim,
        pc_range=point_cloud_range,
        device=device,
        is_query=True,
    )

    # 构建模型
    model = GaussianToDetr(
        model_cfg=model_cfg,
        voxel_size=voxel_size,
        grid_size=grid_size,
        point_cloud_range=point_cloud_range,
        feature_dim=feature_dim,
        semantic_dim=semantic_dim,
        num_classes=num_classes,
        box_dim=box_dim,
        d_model=d_model,
        max_neighbors=32,
        window_size=1,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(normal_gaussians, query_gaussians)

    cls_logits = outputs["cls_logits"]  # [Q_p, num_classes]
    box_pred = outputs["box_pred"]      # [Q_p, box_dim]
    inter = outputs["intermediates"]

    pooled_query = inter["pooled_query"]
    Q_p = pooled_query["mu"].size(0)

    print("==== GaussianToDetr Test ====")
    print(f"pooled_query count (Q_p): {Q_p}")
    print("cls_logits shape:", cls_logits.shape)
    print("box_pred shape:", box_pred.shape)

    # 基本 shape 检查
    assert cls_logits.shape == (Q_p, num_classes)
    assert box_pred.shape == (Q_p, box_dim)

    # 邻居信息合法性检查
    neighbor_info = inter["neighbor_info"]
    neighbor_indices = neighbor_info["neighbor_indices"]
    neighbor_masks = neighbor_info["neighbor_masks"]
    neighbor_d2 = neighbor_info["neighbor_d2"]

    assert neighbor_indices.shape[0] == Q_p
    assert neighbor_masks.shape == neighbor_indices.shape
    assert neighbor_d2.shape == neighbor_indices.shape

    # mask 为 True 的位置 index 必须 >= 0
    assert torch.all(neighbor_indices[neighbor_masks] >= 0)

    # 输出中没有 NaN/Inf
    assert torch.isfinite(cls_logits).all()
    assert torch.isfinite(box_pred).all()

    print("All checks passed.\n")


if __name__ == "__main__":
    test_gaussian_to_detr(device="cpu")
    # 如需在 GPU 上测试：
    # test_gaussian_to_detr(device="cuda:0")
