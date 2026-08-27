# AirV2X Gaussian Pipeline Task Spec

## 1. Task Background

This project targets **cooperative 3D detection on AirV2X** with heterogeneous agents:
- vehicle
- RSU
- drone

The core idea is to maintain a **global Gaussian state** anchored by fused LiDAR voxels, and use multi-view images to shape each Gaussian with:
- image-plane morphology support
- ray-direction depth uncertainty
- iterative intra-agent refinement
- later inter-agent interaction

---

## 2. What the two features mean

After intra-agent refinement, each Gaussian feature is split into two branches:

### 2.1 Self feature

**Self feature** represents the Gaussian's **own stable local state** after local refinement.

It is mainly used for:
- preserving the Gaussian's local semantic and geometric identity
- local self-attention within nearby windows / neighbors
- supporting stable Gaussian state update
- keeping intra-agent consistency

Intuition:
- this branch focuses on **what this Gaussian already knows about itself and its local neighborhood**

Notation:
\[
f_i^{self}
\]

---

### 2.2 Cross feature

**Cross feature** represents the Gaussian's **interaction-oriented feature** for collecting complementary evidence from other sources.

It is mainly used for:
- cross-attention with other agents / other view groups
- absorbing heterogeneous evidence
- complementing missing morphology or support
- building collaboration-specific representation

Intuition:
- this branch focuses on **what this Gaussian should ask from other observations**

Notation:
\[
f_i^{cross}
\]

---

### 2.3 Why split them

A single feature is forced to do two jobs at once:
1. maintain the Gaussian's own stable representation
2. actively query heterogeneous evidence

This can make the representation entangled.

So we split:
- **self feature** = stable local state
- **cross feature** = interaction/query state

This makes the later local self-attention and cross-attention cleaner.

---

## 3. Updated pipeline with the new logic

Your latest logic adds an important step:

After the **first Gaussian generation**, the Gaussian is already projected to image, uses heatmap trend to run a deformable/cross attention operation, gets a first Gaussian feature, updates its just-estimated parameters once, and then is projected again to image for another **intra-agent refinement round** before going into inter-agent interaction.

This means the overall pipeline has **two intra-agent image-conditioned refinement stages**:

1. **initial Gaussian shaping**
2. **second intra-agent re-projection refinement**

That updated logic is described below.

---

## 4. Full pipeline

### Stage 0. Inputs

Inputs include:

#### 0.1 Multi-agent LiDAR
LiDAR point clouds from:
- vehicles
- RSUs
- drones

All LiDAR is transformed into a common global frame.

#### 0.2 Multi-agent images
All images from all available cameras of all agents.

#### 0.3 Masks
We maintain:
- **agent-wise voxel masks**: produced in the dataloader **only** when using this Gaussian pipeline model (see §18.4). Tensor layout / keys TBD in skeleton follow-up.
- **valid-view masks** for Gaussian-to-image projection: required for Stage 4+; CUDA/MambaFusion acceleration remains an optional optimization (interface-friendly first).

#### 0.4 Unified Interface Keys
To avoid ambiguous aliases in the skeleton code, the current pipeline uses one fixed key name for each feature interface.

Under each `agent` sub-dict, use:
- `batch_merged_cam_inputs["imgs"]`: `[B, num_views, 3, H_img, W_img]`
- `batch_merged_cam_inputs["intrinsics"]`: `[B, num_views, 3, 3]` or `[B, num_views, 4, 4]`
- `batch_merged_cam_inputs["extrinsics"]`: `[B, num_views, 4, 4]`
- `batch_merged_cam_inputs["post_rots"]`: `[B, num_views, 3, 3]`
- `batch_merged_cam_inputs["post_trans"]`: `[B, num_views, 3]`
- `image_feature`: the single image feature map used by Gaussian observation sampling with shape `[B, num_views, C_img, H_feat, W_feat]`
- `heatmap_feature`: the image-plane heatmap / semantic support feature map with shape `[B, num_views, C_hm, H_feat, W_feat]`
- `label_map`: categorical image-plane label map on the proposal grid with shape `[B, num_views, H_feat, W_feat]`; foreground pixels store class ids and background is `-1`
- `semantic_feature`: the optional semantic feature map if separated from heatmap support, shape `[B, num_views, C_sem, H_feat, W_feat]`
- `voxel_coords`: voxel coordinates for lidar-anchor Gaussian projection
- `voxel_features`: voxel-aligned lidar features

Under top-level `batch_dict`, use:
- `lidar_mask[agent]`: agent-wise voxel selection mask with shape `[B, num_voxels]`; row `b` marks which global voxels should be projected to the `b`-th local agent of that type

Under `batch_dict["gaussian_pipeline"]`, use:
- `gaussian_candidates[agent]`: a short list of minimal Gaussian candidate entries used across files; each entry keeps only the candidate feature / mean / 2D support matrix / projected coords / `local_agent_ids` / agent-local `view_ids` / `group_ids`, plus the depth statistics needed by first-round scoring
- `projection_masks[agent]`: per-agent image-plane projection masks; the current minimal implementation uses `lidar_coverage_mask` with shape `[B, num_views, H_feat, W_feat]` on the same proposal grid as `label_map`

No alias keys, fallback branches, or shape-compatibility code should be added in code. If an interface name or tensor shape changes later, update the spec first and then update the code to the new single contract.

#### 0.5 Agent-specific Depth Config

The shared depth predictor must read one explicit `AGENT_DEPTH_CONFIG` block from the Gaussian pipeline model config.
Do not infer these settings from `VTRANSFORM` or other modules in code.

Minimal config structure:

```yaml
AGENT_DEPTH_CONFIG:
  vehicle:
    DBOUND: [2.0, 50.0, 1.0]
    depth_input_dim: 128
    depth_hidden_dim: 128
  rsu:
    DBOUND: [2.0, 50.0, 0.5]
    depth_input_dim: 128
    depth_hidden_dim: 128
  drone:
    DBOUND: [6.0, 150.0, 0.5]
    depth_input_dim: 128
    depth_hidden_dim: 128
```

Notes:
- each agent has its own LSS depth discretization and its own depth head parameters
- `DBOUND: [depth_start, depth_max, step]` defines that agent's depth bins
- `depth_num` can be derived from `DBOUND`; if desired, it can also be passed explicitly in the agent sub-config
- both lidar-hit and image-only branches must reuse the same shared predictor instance, but route through the current agent's own config / head
- image-only generation uses the upstream `label_map` directly instead of any top-k heatmap selection
- local 2D support covariance is estimated from a same-label local patch; patch size is configured once by `local_patch_size`

---

### Stage 1. Global LiDAR fusion and voxelization

All LiDAR points are transformed to a common frame and voxelized into a global voxel representation.

Outputs:
- global voxel centers
- global voxel features
- agent-wise voxel masks

Notation:
\[
\{(x_i, f_i^{lidar}, m_i^{agent})\}_{i=1}^N
\]

where:
- \(x_i\): voxel center
- \(f_i^{lidar}\): fused voxel feature
- \(m_i^{agent}\): indicates which agent contributes to this voxel

---

### Stage 2. Intra-agent LiDAR-to-image guidance

For each agent, use its own masked LiDAR voxels and project them to its own images.

Purpose:
- inject sparse depth hints into image features
- improve heatmap prediction
- improve LSS depth estimation

This stage is lightweight and does **not** serve as the main fusion stage.
Also,when project lidar to image, should encode the relative depth to the point, which could give a hint to lss 
Possible forms:
- sparse depth map
- relative depth encoding
- simple feature concatenation
- lightweight cross-attention
- add it as TODO,do not coding, just pass, add it as a interface(for the 2 modality fusion part)

Outputs:
- image features with LiDAR depth hints
- heatmap / semantic features
- LSS-ready depth features 

---

### Stage 3. Global anchor Gaussian initialization

Each global voxel initializes one anchor Gaussian.

At this point:
- center comes from global voxel center
- initial feature comes from LiDAR voxel feature
- covariance is not yet final; it will be determined by image observations

Notation:
\[
g_i^{anchor} = (\mu_i, f_i^{(0)})
\]
with
\[
\mu_i = x_i,\quad f_i^{(0)} = f_i^{lidar}
\]

---

### Stage 4. Multi-view valid mask construction

Each anchor Gaussian is projected to all candidate images.

For Gaussian \(g_i\) and image \(c\), projection gives:
\[
p_{i,c} \in \mathbb{R}^2
\]

A valid-view mask is built:
\[
m_{i,c}^{view}
\]

This mask can depend on:
- projection inside image range
- positive depth
- valid viewing angle
- non-empty semantic / heatmap support
- optional depth consistency

This determines which images are used to observe Gaussian \(g_i\).

Current minimal implementation note:
- besides the future Gaussian valid-view mask, the current code also exposes an agent-local image-plane binary `lidar_coverage_mask`
- for each projected lidar hit \((u, v)\) on one camera view, map it to the target feature grid and mark the 2x2 integer neighborhood:
  - `[(floor(u), floor(v)), (ceil(u), floor(v)), (floor(u), ceil(v)), (ceil(u), ceil(v))]`
- this mask is indexed by `local_agent_id` and the current agent-local `view_id`, so there is no cross-agent camera index mixing when each agent is projected independently
- later image-only proposal generation should suppress proposal creation on masked cells first, before adding more advanced filtering rules

---

## 5. First image-conditioned Gaussian generation

This is the **first Gaussian shaping stage**.

For each valid pair \((i,c)\), use projection point \(p_{i,c}\) to extract view-specific evidence.

### 5.1 Heatmap-trend-based 2D support

Around \(p_{i,c}\), use heatmap / semantic trend to estimate a 2D support covariance:
\[
\Sigma_{i,c}^{2D}
\]

This matrix describes:
- image-plane support range
- local elongation direction
- semantic shape tendency
- serve it as a interface,do not coding at the first time

Current skeleton note:
- the current minimal interface may use a coarse heatmap-center trend approximation only for pipeline bootstrapping
- this approximation should **not** be treated as the final heatmap support design

TODO (future version):
- replace the global heatmap-center approximation with a **local patch-based heatmap support estimator**
- for each projected point \(p_{i,c}\), crop a local heatmap / semantic neighborhood centered at \(p_{i,c}\)
- estimate the local support trend from that neighborhood instead of the full-view heatmap
- candidate realizations include:
  - local weighted centroid around \(p_{i,c}\)
  - local second-moment / covariance estimation
  - local PCA-style principal direction extraction
  - semantic-aware local support fitting
- the final \(\Sigma_{i,c}^{2D}\) should come from this local neighborhood trend and should only describe image-plane morphology support
- this heatmap support must **not** directly move the Gaussian absolute 3D center; it only provides tangent-plane support geometry

### 5.2 LSS depth uncertainty

At the same location, use the LSS-style depth prediction to estimate:
- soft depth mean
- depth variance

Construct ray-direction covariance:
\[
\Sigma_{i,c}^{ray} = \sigma_{i,c}^2\, r_{i,c} r_{i,c}^\top
\]
should base on code of paper Gaussianlss

Current implementation note:
- `pred_depth_mean` uses the soft expectation of the LSS depth distribution
- `r_{i,c}` is the unit vector from the current camera center to the projected lidar-anchor point
- the current minimal path keeps these depth statistics as local tensors during lidar candidate construction, and only writes the final minimal candidate entry into `batch_dict["gaussian_pipeline"]["gaussian_candidates"][agent]`
- lidar-hit projection is done per local agent `b` with `map_points_cuda`; every valid hit keeps both `local_agent_id` and agent-local `view_id`
- repeated lidar observations are grouped by global voxel identity, so the same global voxel can aggregate evidence from multiple vehicles and from multiple views of one vehicle

TODO (future version):
- add explicit `anchor_depth` for each projected lidar anchor under the current camera view
- build `relative_depth_mean = pred_depth_mean - anchor_depth`
- build `relative_depth_variance` as the depth uncertainty counterpart for the relative-depth branch
- keep `depth_entropy` out of the later tokenized `depth_descriptor`

### 5.3 Lift image-plane covariance to 3D tangent covariance

Using local back-projection Jacobian \(J_{i,c}\):
\[
\Sigma_{i,c}^{tan} = J_{i,c}\Sigma_{i,c}^{2D}J_{i,c}^\top
\]

Current minimal implementation note:
- the current code places the covariance builder in `geometry/`
- `\Sigma^{ray}` is built directly from `depth_variance` and `ray_direction`
- `\Sigma^{tan}` should be built with the explicit local back-projection Jacobian induced by camera intrinsics, camera-to-lidar extrinsics, post-augmentation rotation, and the current anchor depth
- concretely, at the current projected point, map normalized image-plane perturbations to image pixels, undo image post-rotation, back-project them with the camera intrinsic matrix, and rotate them into the lidar frame with the camera-to-lidar extrinsic rotation
- then use the local Jacobian for covariance propagation:
  - `\Sigma^{tan} = J \Sigma^{2D} J^\top`
- then form:
  - `\Sigma^{3D} = \Sigma^{tan} + \Sigma^{ray}`

Then:
\[
\Sigma_{i,c}^{3D} = \Sigma_{i,c}^{tan} + \Sigma_{i,c}^{ray}
\]

### 5.4 First DA / cross-attention on image support region

At this first shaping stage, after projection to image and heatmap-trend extraction, run a **DA-style cross attention / deformable aggregation** between the Gaussian query and the image support region.

This gives a first view-specific Gaussian feature:
\[
f_{i,c}^{(1)}
\]

For one global voxel / Gaussian, let the maximum number of valid projected views be:
\[
K \le 5
\]

For each valid view \(c\), first build a raw view observation token from:
- sampled image feature around the projected point / region
- heatmap / semantic support information
- LSS depth quality information

Then compress each raw token with a shared MLP into a 64-dim view token:
\[
t_{i,c}^{view} \in \mathbb{R}^{64}
\]

For Gaussian \(g_i\), stack up to \(K\) valid view tokens into:
\[
T_i \in \mathbb{R}^{K \times 64}
\]
with a valid-view mask for padded invalid views.

Before scoring confidence, run a lightweight self-attention over the \(K\) view tokens of the same Gaussian so that each view can see the other projected views of the same Gaussian.

This explicitly models:
- cross-view consistency
- cross-view competition
- complementary evidence across views

Then apply a shared score MLP on each attended view token to obtain a scalar score:
\[
q_{i,c}^{(1)}
\]

After masked softmax over valid views, obtain multi-view confidence weights:
\[
\alpha_{i,c}^{(1)} = \text{softmax}_c(q_{i,c}^{(1)})
\]

These confidence weights are used to fuse:
- view-specific Gaussian features
- view-specific covariance estimates
- later confidence-related update signals

Then aggregate all valid views to obtain the first Gaussian feature:
\[
f_i^{(1)} = \sum_c \alpha_{i,c}^{(1)} f_{i,c}^{(1)}
\]

and similarly the first fused covariance can be formed from the confidence-weighted view-specific covariance observations:
\[
\Sigma_i^{(1)} = \text{Fuse}_{cov}\left(\{\alpha_{i,c}^{(1)}, \Sigma_{i,c}^{3D}\}_c\right)
\]

This means each view confidence is not decided independently; it is determined jointly after comparing the current view against other valid views of the same Gaussian.  
For example, if one Gaussian is observed by one drone image and four vehicle images, the confidence module should be able to learn that the vehicle observations deserve larger weights in that sample when they are more mutually consistent.

### 5.5 First parameter update

Use the first Gaussian feature \(f_i^{(1)}\) and fused covariance \(\Sigma_i^{(1)}\) to update the newly obtained Gaussian parameters.

The update target can include:
- covariance refinement
- confidence refinement
- feature refinement
- optional small mean residual

Notation:
\[
g_i^{(1)} = \text{Update}_1(g_i^{anchor}, f_i^{(1)}, \Sigma_i^{(1)})
\]

This stage forms the **first complete Gaussian proposal**.

---

## 6. Second intra-agent re-projection refinement

This is the new logic you just added.

After the first Gaussian is formed and updated once, project the updated Gaussian back to the images again for another **intra-agent information acquisition and update**.

This is not the initial shaping anymore.  
This is a **second-round intra-agent refinement** using a better Gaussian state.

### 6.1 Re-projection

Project updated Gaussian \(g_i^{(1)}\) again to valid images:
\[
p_{i,c}^{(2)}
\]

### 6.2 Second-round intra-agent observation

Using the updated Gaussian state:
- re-query image support region
- re-estimate local semantic support
- optionally refine local morphology evidence
- collect stronger intra-agent image evidence

This gives:
\[
f_{i,c}^{(2)}
\]

### 6.3 Second-round aggregation

Aggregate view-specific second-round observations:
\[
f_i^{(2)} = \sum_c \alpha_{i,c}^{(2)} f_{i,c}^{(2)}
\]

### 6.4 Second-round Gaussian update

Use \(f_i^{(2)}\) to refine the Gaussian again:
\[
g_i^{(2)} = \text{Update}_2(g_i^{(1)}, f_i^{(2)})
\]

At this point, each Gaussian has completed:
- first image-conditioned generation
- one more intra-agent re-projection refinement

This becomes the Gaussian state that enters the interaction stage.

---

## 7. Multi-view covariance fusion

For each Gaussian, the final covariance after multi-view observation can be fused in two retained versions.

### Version A. Convex combination

\[
\Sigma_i = \sum_c \alpha_{i,c}\Sigma_{i,c}^{3D}
\]

Advantages:
- simple
- stable
- continuous
- easy to implement

### Version B. Information matrix fusion

\[
\Lambda_{i,c} = (\Sigma_{i,c}^{3D})^{-1}
\]
\[
\Lambda_i = \sum_c \alpha_{i,c}\Lambda_{i,c}
\]
\[
\Sigma_i = \Lambda_i^{-1}
\]

Advantages:
- closer to uncertainty fusion
- sharper views can dominate naturally

Both are retained in the design and will be screened experimentally later.

---

## 8. Image-only proposal Gaussian generation

For heatmap-salient regions without valid Gaussian coverage, generate image-only proposal Gaussians.

Each proposal includes:
- 3D mean from image position + soft depth mean
- covariance from image-plane support + ray uncertainty
- image feature as initial feature

These proposals are merged into the Gaussian set.
These generated gaussian should involved in first/second img projection interation
it may use the lidar feature that projected into img plane(lidar generated gaussian) TO CHECK
Current minimal interface note:
- image-only proposal generation should first read `batch_dict["gaussian_pipeline"]["projection_masks"][agent]["lidar_coverage_mask"]`
- any candidate image cell already marked by this binary mask should be skipped so duplicated lidar-covered proposals are not created
- image-only proposals use the same `[B, num_views, ...]` local-agent indexing as lidar-hit observations and do not participate in repeated-view group fusion; their `group_ids` stay unique
---

## 9. Main Gaussian state before interaction

After the second intra-agent refinement, each Gaussian has:

- center:
\[
\mu_i
\]
- covariance:
\[
\Sigma_i
\]
- main feature:
\[
f_i^{main}
\]

This is the input state for interaction:
\[
g_i = (\mu_i,\Sigma_i,f_i^{main})
\]

---

## 10. Dual-path feature split

Split the main feature into two branches:

\[
f_i^{self} = \text{MLP}_{self}(f_i^{main})
\]
\[
f_i^{cross} = \text{MLP}_{cross}(f_i^{main})
\]

### Meaning
- \(f_i^{self}\): stable local Gaussian representation
- \(f_i^{cross}\): interaction-oriented feature for other agents / view groups

---

## 11. Local self-attention and local cross-agent attention

Within local windows or local neighborhoods, run self-attention and cross attn:
\[
\tilde f_i^{self} = \text{LocalSelfAttn}(Q=f_i^{self},K=f_j^{self},V=f_j^{self})
\tilde f_i^{cross} = \text{LocalCrossAttn}(Q=f_i^{cross},K=f_j^{cross},V=f_j^{cross})
\]

Purpose:
- local consistency
- denoising
- local context enhancement

after local attn, update gaussian parameters
---

## 12. Global self-attention and Cross-agent attention

Run cross-attention using cross features:
\[
\tilde f_i^{self} = \text{SelfAttn}(Q=f_i^{self},K=f_j^{self},V=f_j^{self})
\tilde f_i^{cross} = \text{CrossAttn}(Q=f_i^{cross},K=f_j^{cross},V=f_j^{cross})
\]

This stage collects:
- heterogeneous agent evidence
- complementary morphology
- cross-view support

Optional additions:
- agent type embedding
- geometry-aware attention bias
- relative position / covariance bias

after global attn, update gaussian parameters
---

## 13. Feature fusion, update, and rendering

Fuse the two branches:
\[
f_i^{fused} = \text{MLP}_{fuse}([\tilde f_i^{self}, \tilde f_i^{cross}])
\]

Then use two heads:

### 13.1 Update head
Update Gaussian state:
- feature
- confidence
- optional small covariance residual

### 13.2 Render head
Generate rendering feature:
\[
f_i^{render}
\]

This is the feature used for BEV splatting.

---

## 14. Gaussian splatting to BEV

Render all final Gaussians to BEV latent space:
\[
F^{bev}_0 = \text{splat}(\{g_i\})
\]

Inputs:
- \(\mu_i\)
- \(\Sigma_i\)
- \(f_i^{render}\)

---

## 15. BEV backbone and final detection

Use BEV backbone / decoder / head to output:
- 3D detection boxes
- classification scores
- optional BEV outputs

---

## 16. Summary in one sentence

We first build a unified global voxel representation from multi-agent LiDAR, initialize anchor Gaussians, generate view-specific covariance and feature observations from all valid images using heatmap trend and LSS depth uncertainty, perform a first Gaussian generation followed by a second intra-agent re-projection refinement, then split Gaussian features into self and cross branches for local self-attention and cross-agent attention, and finally render the refined Gaussian set into BEV space for cooperative 3D detection.

---

## 17. Current candidate innovation points

### Innovation 1
**Global voxel-anchored, image-conditioned anisotropic Gaussian generation**

### Innovation 2
**Multi-view Gaussian observation aggregation with learnable covariance fusion**

### Innovation 3
**Dual-path Gaussian interaction with second-round intra-agent refinement**

