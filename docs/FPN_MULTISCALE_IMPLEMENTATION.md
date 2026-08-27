# FPN 多尺度方案实现分析

## 一、现状与目标

### 1.1 当前架构

```
输入图像 256×704
    ↓
SimpleCNN (conv + 3×MaxPool)
    ↓
特征图 32×88 (单尺度)
    ↓
Detection Head (语义分割) + TPV Projector (LSS)
```

- **Vehicle/RSU**: 目标尺度适中，32×88 可接受
- **Drone**: 目标极小（高空俯视），在 32×88 上几乎丢失

### 1.2 目标

为 Drone 使用更高分辨率特征（64×176），为 Vehicle/RSU 保持 32×88，实现 **按 agent 类型选择特征尺度**。

---

## 二、数据流依赖分析

### 2.1 下游模块对特征尺寸的要求

| 模块 | 对 H×W 的要求 | 说明 |
|------|---------------|------|
| **Detection Head** | 任意 | Conv2d 支持任意空间尺寸 |
| **TPV Projector** | 任意 | `_create_frustum(agent_type, H, W)` 按需创建，depthnet 为 Conv2d |
| **语义监督标签** | 与特征一致 | `_build_semantic_supervision_from_image_gt` 支持传入 `feat_h, feat_w` |

结论：**下游均可支持按 agent 使用不同分辨率**，无需固定 32×88。

### 2.2 SimpleCNN 中间分辨率

```
输入: 256×704
  → pool1: 128×352
  → pool2: 64×176   ← P2 (用于 drone)
  → pool3: 32×88   ← P3 (用于 vehicle/rsu)
```

---

## 三、实现方案

### 方案 A：按 Agent 选择尺度（推荐）

**思路**：Backbone 输出 P2(64×176) 和 P3(32×88)，按 agent_type 选择使用哪一档。

```
                    ┌─────────────────────┐
                    │   SimpleCNN 共享     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
        ┌──────────┐                     ┌──────────┐
        │ P2 64×176│                     │ P3 32×88 │
        │ (pool2后)│                     │ (pool3后)│
        └────┬─────┘                     └────┬─────┘
             │                                 │
    drone ───┤                     vehicle/rsu─┤
             ▼                                 ▼
    Detection@64×176                  Detection@32×88
    TPV@64×176                         TPV@32×88
```

### 方案 B：FPN 融合 + 单尺度输出

**思路**：P2 与 P3 做 top-down 融合，输出统一 32×88，但融合特征包含多尺度信息。

- 优点：下游接口不变
- 缺点：融合可能削弱小目标，Drone 收益有限

### 方案 C：FPN 多尺度 + 多 Head

**思路**：P2、P3 各自有检测头，损失在各自尺度上计算并加权求和。

- 优点：各尺度独立监督
- 缺点：参数量增加，实现更复杂

---

## 四、方案 A 详细实现（推荐）

### 4.1 修改 GaussianImageFeatureExtractor

**核心**：拆分 `conv_layers`，在 pool2 后和 pool3 后分别输出特征。

```python
# 将 conv_layers 拆成 stage1 (到 pool2) 和 stage2 (pool3)
self.stage1 = nn.Sequential(
    nn.Conv2d(4, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
    nn.MaxPool2d(2, 2),  # 256→128
    nn.MaxPool2d(2, 2),  # 128→64
    # 输出: 64×176
)
self.stage2 = nn.Sequential(
    nn.MaxPool2d(2, 2),  # 64×176 → 32×88
)
self.fusion_P2 = nn.Conv2d(256, 128, 1)  # P2 通道对齐
self.fusion_P3 = nn.Conv2d(256, 128, 1)  # P3 通道对齐
```

**forward 逻辑**：

```python
def forward(self, agent_data, agent_type=None):
    imgs = agent_data['batch_merged_cam_inputs']['imgs']
    B, N, C, H, W = imgs.shape
    x = imgs.view(B * N, C, H, W)
    
    feat_P2 = self.stage1(x)           # [B*N, 256, 64, 176]
    feat_P3 = self.stage2(feat_P2)    # [B*N, 256, 32, 88]
    
    P2 = self.fusion_P2(feat_P2)      # [B*N, 128, 64, 176]
    P3 = self.fusion_P3(feat_P3)      # [B*N, 128, 32, 88]
    
    return {
        'P2': P2.view(B, N, 128, 64, 176),
        'P3': P3.view(B, N, 128, 32, 88),
    }
```

### 4.2 修改 GaussianImageBackbone.forward（预训练版）

```python
# Agent 与尺度映射
AGENT_FEATURE_SCALE = {
    'drone': 'P2',   # 64×176
    'vehicle': 'P3',
    'rsu': 'P3',     # 32×88
}

for agent_type in self.agent_types:
    if agent_type not in batch_dict or 'batch_merged_cam_inputs' not in batch_dict[agent_type]:
        continue
    
    agent_data = batch_dict[agent_type]
    multi_scale_feats = self.image_backbone(agent_data)
    
    scale_key = AGENT_FEATURE_SCALE.get(agent_type, 'P3')
    image_features = multi_scale_feats[scale_key]
    B, N, C, H, W = image_features.shape
    
    # Detection head（支持任意 H,W）
    semantic_logits = self.detection_head.forward_from_features(
        image_features, feat_h=H, feat_w=W
    )
    
    # 语义监督（使用相同的 feat_h, feat_w）
    semantic_targets = self._build_semantic_supervision_from_image_gt(
        batch_dict, agent_type, B, N, H, W
    )
```

### 4.3 修改 GaussianDetectionHead

去掉固定 `image_shape` 的断言，支持动态尺寸：

```python
def forward_from_features(self, image_features, feat_h=None, feat_w=None):
    B, N, C_feat, H_feat, W_feat = image_features.shape
    if feat_h is None:
        feat_h, feat_w = self.image_shape[0], self.image_shape[1]
    # 移除 assert，允许多尺度输入
    x = image_features.view(B * N, C_feat, H_feat, W_feat)
    logits = self.lightweight_cls_head(x)  # [B*N, M, H_feat, W_feat]
    # ... 后续逻辑保持不变，用 H_feat, W_feat 替代 32, 88
```

### 4.4 修改 TPV Projector（正式训练）

TPV 已支持可变 H×W，只需传入对应 scale 的特征即可：

```python
# backbone2d_semantic.py 的 forward 中
scale_key = AGENT_FEATURE_SCALE.get(agent_type, 'P3')
image_features = multi_scale_feats[scale_key]
# ... 直接传入 tpv_projector，其内部会根据 image_feat 的 H,W 创建 frustum
tpv_results = self.tpv_projector(agent_type, image_features, ...)
```

### 4.5 配置项

```yaml
# config.yaml
model:
  args:
    BACKBONE_2D:
      USE_FPN_MULTISCALE: True
      AGENT_FEATURE_SCALE:
        drone: P2    # 64×176
        vehicle: P3
        rsu: P3      # 32×88
      IMAGE_SHAPE_P2: [64, 176]
      IMAGE_SHAPE_P3: [32, 88]
```

---

## 五、预训练与正式训练的衔接

### 5.1 预训练

- 只训练 `image_backbone` + `detection_head`
- backbone 输出 `{'P2', 'P3'}`，按 agent 选 scale
- checkpoint 中保留 `image_backbone.*` 和 `detection_head.*`

### 5.2 正式训练加载权重

- `load_pretrained_weights` 仍按 `image_backbone.` 和 `detection_head.` 过滤
- 新结构下这两部分名称不变，可直接加载
- 若旧 checkpoint 为单尺度，可用 `strict=False`，新增的 `stage2`、`fusion_P2` 等随机初始化

### 5.3 兼容旧权重的策略

```python
def load_pretrained_weights(self, pretrained_path, strict=False):
    # 使用 strict=False 以允许新增的 P2 相关参数
    # 旧权重中的 conv_layers 可部分映射到 stage1
```

---

## 六、可选：FPN Top-Down 增强

若希望 P2 融合 P3 的语义信息，可增加轻量 top-down：

```python
# 在 backbone 内
self.lateral_P2 = nn.Conv2d(256, 128, 1)
self.lateral_P3 = nn.Conv2d(256, 128, 1)
self.upsample_P3_to_P2 = nn.Upsample(scale_factor=2, mode='bilinear')

# forward
P3_lat = self.lateral_P3(feat_P3)
P2_lat = self.lateral_P2(feat_P2)
P2_enriched = P2_lat + self.upsample_P3_to_P2(P3_lat)  # top-down
```

Drone 使用 `P2_enriched`，Vehicle/RSU 使用 `P3_lat`。

---

## 七、实现检查清单（已完成）

- [x] `GaussianImageFeatureExtractor` 拆分为 stage1/stage2（SimpleCNN）或 layer1/layer2（ResNet101），输出 P2 和 P3
- [x] `GaussianImageBackbone.forward` 按 agent_type 选择 scale
- [x] `GaussianDetectionHead.forward_from_features` 支持动态 feat_h, feat_w
- [x] `_build_semantic_supervision_from_image_gt` 传入 agent 对应的 feat_h, feat_w
- [ ] `backbone2d_semantic.py` 中正式训练流程同步上述逻辑（预训练模块已实现）
- [x] 配置中添加 `USE_FPN_MULTISCALE` 与 `AGENT_FEATURE_SCALE`
- [x] 预训练脚本改为按 agent 分别计算 loss（支持不同分辨率）

## 八、已修改文件

- `opencood/models/gaussian_modules/backbone2d_semantic_pretraining.py`
- `opencood/tools/train_backbone2d_semantic_pretraining.py`
- `AirV2X-Perception-Checkpoints/airv2x_intermediate_gaussian/config_backbone2d_pretraining.yaml`

---

## 九、预期收益与代价

| 项目 | 说明 |
|------|------|
| Drone 小目标 | 64×176 相对 32×88 分辨率提升 4 倍，小目标可保留 |
| 显存 | Drone 分支分辨率更高，显存略有增加 |
| 计算量 | Drone 的 64×176 路径计算量约为 P3 的 4 倍 |
| 兼容性 | 通过 `strict=False` 加载，可与旧单尺度权重共存 |
