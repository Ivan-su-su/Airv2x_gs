import torch
import torch.nn as nn
import numpy as np
from opencood.models.mambafusion_modules.ops.pointnet2.pointnet2_stack.pointnet2_utils import three_nn
from opencood.models.mambafusion_modules.ops.win_coors.flattened_window_cuda import map_points as map_points_cuda
from opencood.models.mambafusion_modules.ops.win_coors.flattened_window_cuda import map_points_v2 as map_points_v2_cuda
# import Tuple
from typing import Tuple
# @torch.jit.script
def get_points(pc_range, sample_num, space_shape, coords=None):
    '''Generate points in specified range or voxels

    Args:
        pc_range (list(int)): point cloud range, (x1,y1,z1,x2,y2,z2)
        sample_num (int): sample point number in a voxel
        space_shape (list(int)): voxel grid shape, (w,h,d)
        coords (tensor): generate points in specified voxels, (N,3)

    Returns:
        points (tensor): generated points, (N,sample_num,3)
    '''
    sx, sy, sz = space_shape # [360, 360, 1]
    x1, y1, z1, x2, y2, z2 = pc_range # [-54.0, -54.0, -10.0, 54.0, 54.0, 10.0]
    if coords is None:
        coord_x = torch.linspace( # torch.Size([1, 360, 360, 1])
            0, sx-1, sx).view(1, -1, 1, 1).repeat(1, 1, sy, sz)
        coord_y = torch.linspace( # torch.Size([1, 360, 360, 1])
            0, sy-1, sy).view(1, 1, -1, 1).repeat(1, sx, 1, sz)
        coord_z = torch.linspace(   # torch.Size([1, 360, 360, 1])
            0, sz-1, sz).view(1, 1, 1, -1).repeat(1, sx, sy, 1)
        coords = torch.stack((coord_x, coord_y, coord_z), -1).view(-1, 3) # torch.Size([129600, 3]) 360*360*1=129600
    points = coords.clone().float()
    points[..., 0] = ((points[..., 0]+0.5)/sx)*(x2-x1) + x1 # 在x轴上均匀分布，然后映射到pc_range的范围内 +0.5是为了保证在x1和x2之间
    points[..., 1] = ((points[..., 1]+0.5)/sy)*(y2-y1) + y1
    points[..., 2] = ((points[..., 2]+0.5)/sz)*(z2-z1) + z1

    if sample_num == 1:
        points = points.unsqueeze(1)
    else:
        points = points.unsqueeze(1).repeat(1, sample_num, 1) # torch.Size([129600, 20, 3])
        points[..., 2] = torch.linspace(z1, z2, sample_num).unsqueeze(0) # 在z轴上均匀分布，然后映射到pc_range的范围内(-10 -> 10)
    return points

# @torch.jit.script
def map_points(points, lidar2image, image_aug_matrix, batch_size: int, image_shape: Tuple[int, int], expand_scale: float = 0.0):
    '''Map 3D points to image space.

    Args:
        points (tensor): Grid points in 3D space, shape (grid num, sample num,4).
        lidar2image (tensor): Transformation matrix from lidar to image space, shape (B, N, 4, 4).
        image_aug_matrix (tensor): Transformation of image augmentation, shape (B, N, 4, 4).
        batch_size (int): Sample number in a batch.
        image_shape (tuple(int)): Image shape, (height, width).

    Returns:
        points (tensor): 3d coordinates of points mapped in image space. shape (B,N,num,k,4)
        points_2d (tensor): 2d coordinates of points mapped in image space. (B,N,num,k,2)
        map_mask (tensor): Number of points per view (batch size x view num). (B,N,num,k,1)
    '''
    points = points.to(torch.float32) # torch.Size([32338, sample_num, 3])
    lidar2image = lidar2image.to(torch.float32) # torch.Size([1, B*N, 4, 4])
    image_aug_matrix = image_aug_matrix.to(torch.float32) # torch.Size([1, B*N, 4, 4])
    num_view = lidar2image.shape[1] # B*N
    points = torch.cat((points, torch.ones_like(points[..., :1])), -1) # torch.Size([32338, sample_num, 4]) 在最后一维增加一个1，转换为齐次坐标
    # map points from lidar to (aug) image space
    points = points.unsqueeze(0).unsqueeze(0).repeat(
        batch_size, num_view, 1, 1, 1).unsqueeze(-1) # torch.Size([1, 6, 32338, sample_num, 4, 1])
    grid_num, sample_num = points.shape[2:4] # 129600, sample_num
    lidar2image = lidar2image.view(batch_size, num_view, 1, 1, 4, 4).repeat( # 生成每一个点对应的变换矩阵 torch.Size([2, 6, 1, 1, 4, 4])
        1, 1, grid_num, sample_num, 1, 1) # torch.Size([2, 6, 32338, sample_num, 4, 4])
    image_aug_matrix = image_aug_matrix.view(
        batch_size, num_view, 1, 1, 4, 4).repeat(1, 1, grid_num, sample_num, 1, 1) # torch.Size([2, 6, 129600, sample_num, 4, 4])
    points_2d = torch.matmul(lidar2image, points).squeeze(-1) # 将这些点从激光雷达坐标系映射到相机坐标系 torch.Size([2, 6, 32338, sample_num, 4, 1]) -> torch.Size([2, 6, 32338, sample_num, 4])

    # recover image augmentation
    eps = 1e-5
    map_mask = (points_2d[..., 2:3] > eps) # 保留投影在相机坐标系下z轴大于0的点 torch.Size([2, 6, 32338, sample_num, 1])
    points_2d[..., 0:2] = points_2d[..., 0:2] / torch.maximum(
        points_2d[..., 2:3], torch.ones_like(points_2d[..., 2:3]) * eps) # torch.Size([2, 6, 32338, sample_num, 2]) 除以z轴，得到图像坐标系下的坐标
    points_2d[..., 2] = torch.ones_like(points_2d[..., 2]) # torch.Size([2, 6, 32338, sample_num, 1]) z轴设为1 得到齐次的图像坐标系下的坐标
    points_2d = torch.matmul( # image_aug_matrix包括了图像增强的变换矩阵，如旋转、平移、缩放等
        image_aug_matrix, points_2d.unsqueeze(-1)).squeeze(-1)[..., 0:2] # torch.Size([2, 6, 32338, sample_num, 2]) 乘以图像增强的变换矩阵
    points_2d[..., 0] /= image_shape[1] # torch.Size([2, 6, 32338, sample_num, 1]) 归一化到图像坐标系下的坐标
    points_2d[..., 1] /= image_shape[0] # torch.Size([2, 6, 32338, sample_num, 1]) 归一化到图像坐标系下的坐标

    # mask points out of range
    map_mask = (map_mask & (points_2d[..., 1:2] > 0.0 - expand_scale) 
                & (points_2d[..., 1:2] < 1.0 + expand_scale)
                & (points_2d[..., 0:1] < 1.0 + expand_scale)
                & (points_2d[..., 0:1] > 0.0 - expand_scale)) # torch.Size([2, 6, 32338, sample_num, 1]) 保留在图像范围内的点
    map_mask = torch.nan_to_num(map_mask).squeeze(-1)

    return points.squeeze(-1), points_2d, map_mask


def map_points_v2(
    points: torch.Tensor,
    lidar2image: torch.Tensor,
    image_aug_matrix: torch.Tensor,
    batch_size: int,
    image_shape: Tuple[int, int],
    expand_scale: float = 0.0
    ) -> torch.Tensor:
    """Map 3D points to image space and select first hit view (Python version).
    
    This function is a Python implementation of map_points_v2_cuda, providing
    the same functionality without requiring CUDA extensions.
    
    Args:
        points (tensor): Grid points in 3D space, shape (grid_num, sample_num, 3).
        lidar2image (tensor): Transformation matrix from lidar to image space, 
            shape (B, N, 4, 4).
        image_aug_matrix (tensor): Transformation of image augmentation, 
            shape (B, N, 4, 4).
        batch_size (int): Batch size.
        image_shape (tuple(int)): Image shape, (height, width).
        expand_scale (float): Expand scale for image boundary check. Default: 0.0.
    
    Returns:
        hit_points (tensor): Hit points with shape (grid_num, sample_num, 3).
            The last dimension contains [u_clamped, v_clamped, view_idx]:
            - u_clamped, v_clamped: Normalized 2D coordinates clamped to [0, 3]
            - view_idx: Index of the first view where the point is visible (0 to num_view-1)
    """
    points = points.to(torch.float32)
    lidar2image = lidar2image.to(torch.float32) # torch.Size([1, B*N, 4, 4])
    image_aug_matrix = image_aug_matrix.to(torch.float32) # torch.Size([1, B*N, 4, 4])
    num_view = lidar2image.shape[1] # B*N
    grid_num, sample_num = points.shape[:2]

    points_homo = torch.cat((points, torch.ones_like(points[..., :1])), -1).squeeze(1) # torch.Size([N_lidar, 4]) 在最后一维增加一个1，转换为齐次坐标
    # map points from lidar to (aug) image space
    points_homo = points_homo.unsqueeze(0).unsqueeze(0).repeat(
        batch_size, num_view, 1, 1) # torch.Size([1, B*N, N_lidar, 4])
    points_2d = torch.matmul(
        lidar2image.unsqueeze(2),         # [1, BN, 1, 4, 4]
        points_homo.unsqueeze(-1)         # [1, BN, N_lidar, 4, 1]
    ).squeeze(-1)                         # [1, BN, N_lidar, 4]

    # recover image augmentation
    eps = 1e-5
    map_mask = (points_2d[..., 2:3] > eps)   #保留投影在相机坐标系下z轴大于0的点
    points_2d[..., 0:2] = points_2d[..., 0:2] / torch.maximum(
        points_2d[..., 2:3], torch.ones_like(points_2d[..., 2:3]) * eps)
    points_2d[..., 2] = torch.ones_like(points_2d[..., 2])
    points_2d[..., 3] = torch.ones_like(points_2d[..., 3])
    # dwb 得到点在原始图像上的投影坐标
    u_ori = points_2d[..., 0] / 1280
    v_ori = points_2d[..., 1] / 720
    pixel_ori = torch.stack([u_ori, v_ori], dim=-1)  # [1, BN, N_lidar, 2]
    print(f"pixel_ori shape: {pixel_ori.shape}")

    # TODO: 先不加图像增强矩阵，在原始图像上进行可视化
    # point_img_aug = torch.matmul(
    #     image_aug_matrix.unsqueeze(2),    # [1, BN, 1, 4, 4]
    #     points_2d.unsqueeze(-1)
    # ).squeeze(-1)                       # [1, BN, N_lidar, 4]
    
    # u = point_img_aug[..., 0] / image_shape[1]
    # v = point_img_aug[..., 1] / image_shape[0]
    # pixel_coords = torch.stack([u, v], dim=-1)  # [1, BN, N_lidar, 2]
    # print(f"pixel_coords shape: {pixel_coords.shape}")
    # print(f"pixel_coords max: {pixel_coords[0, 0, :, 0].max()}, min: {pixel_coords[0, 0, :, 0].min()}")
    # print(f"pixel_coords max: {pixel_coords[0, 0, :, 1].max()}, min: {pixel_coords[0, 0, :, 1].min()}")

    u = points_2d[..., 0] / 1280
    v = points_2d[..., 1] / 720
    pixel_coords = torch.stack([u, v], dim=-1)  # [1, BN, N_lidar, 2]
    print(f"pixel_coords shape: {pixel_coords.shape}")

    # mask points out of range (使用归一化后的坐标进行检查)
    map_mask = (map_mask & (pixel_coords[..., 1:2] > 0.0 - expand_scale) 
                & (pixel_coords[..., 1:2] < 1.0 + expand_scale)
                & (pixel_coords[..., 0:1] < 1.0 + expand_scale)
                & (pixel_coords[..., 0:1] > 0.0 - expand_scale))
    map_mask = torch.nan_to_num(map_mask).squeeze(-1)   #[1, BN, N_lidar]

    # Step 2: Process hit points - select first hit view for each point
    # Note: CUDA kernel assumes batch_size=1, so we use batch_idx=0
    batch_idx = 0
    
    num_view = map_mask.shape[1]
    total_points = grid_num * sample_num
    map_mask_flat = map_mask[batch_idx].view(num_view, total_points)  # [num_view, grid_num * sample_num]
    points_2d_flat = pixel_coords[batch_idx].view(num_view, total_points, 2)  # [num_view, grid_num * sample_num, 2]
    
    # Find first hit view for each point using vectorized operations
    # Strategy: For each point, find the first view (lowest index) where mask is True
    # If no view hits, use view 0 (matches CUDA kernel behavior)
    
    # Create a priority matrix where True masks get higher priority for earlier views
    # Priority = mask * (num_view - view_idx), so earlier views (smaller view_idx) get higher priority
    view_indices = torch.arange(num_view, device=pixel_coords.device, dtype=torch.float32).unsqueeze(1)
    mask_with_priority = map_mask_flat.float() * (num_view - view_indices)
    
    # argmax will find the view with highest priority (first True view, or 0 if all False)
    first_hit_view_idx = torch.argmax(mask_with_priority, dim=0)  # [grid_num * sample_num]
    
    # Get the 2D coordinates for the selected view (first hit view, or view 0 if no hit)
    point_indices = torch.arange(total_points, device=pixel_coords.device)
    hit_coords = points_2d_flat[first_hit_view_idx, point_indices]  # [grid_num * sample_num, 2]
    hit_u = hit_coords[:, 0]
    hit_v = hit_coords[:, 1]
    
    # Clamp coordinates to [-1, 2] and then add 1 to get [0, 3] range
    # This matches the CUDA kernel: fminf(fmaxf(hit_u, -1.0f), 2.0f) + 1.0f
    hit_u_clamped = torch.clamp(hit_u, -1.0, 2.0) + 1.0
    hit_v_clamped = torch.clamp(hit_v, -1.0, 2.0) + 1.0
    # dwb 现在实际上是[1,2]之间的值
    
    # Reshape back to [grid_num, sample_num, 3]
    hit_points_out = torch.stack([
        hit_u_clamped,
        hit_v_clamped,
        first_hit_view_idx.float()
    ], dim=1).view(grid_num*sample_num, 3)
    
    return hit_points_out, pixel_ori


class MapImage2Lidar(nn.Module):
    '''Map image patch to lidar space'''

    def __init__(self, model_cfg, accelerate=False, use_map=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg.point_cloud_range # [-54.0, -54.0, -10.0, 54.0, 54.0, 10.0]
        self.voxel_size = model_cfg.voxel_size # [0.3, 0.3, 20.0] 
        self.sample_num = model_cfg.sample_num  # 20 sample num in a voxel
        self.space_shape = [
            int((self.pc_range[i+3]-self.pc_range[i])/self.voxel_size[i]) for i in range(3)] # [360, 360, 1]

        self.points = get_points(
            self.pc_range, self.sample_num, self.space_shape).cuda()
        self.accelerate = accelerate
        if self.accelerate:
            self.cache = None
        self.use_map = use_map

    def forward(self, batch_dict):
        '''Get the coordinates of image patch in 3D space.

        Returns:
            image2lidar_coords_zyx (tensor): The coordinates of image features 
            (batch size x view num) in 3D space.
            nearest_dist (tensor): The distance between each image feature 
            and the nearest mapped 3d grid point in image space.
        '''
        # accelerate by caching when the mapping relationship changes little
        if self.accelerate and self.cache is not None:
            image2lidar_coords_zyx, nearest_dist = self.cache
            return image2lidar_coords_zyx, nearest_dist
        # Ensure lidar2image and image_aug_matrix exist from nested cam inputs if needed
        batch_dict = self._ensure_lidar2image_and_aug(batch_dict)
        img = batch_dict['camera_imgs'] # [batch, view, 3, H, W]
        batch_size, num_view, _, h, w = img.shape
        points = self.points.clone() # torch.Size([129600, 20, 3]) 根据给定的点云范围、采样数量和空间形状，生成一个均匀分布的点集，用于映射图像块到激光雷达空间
        lidar2image = batch_dict['lidar2image'] # (B, V, 4, 4)
        image_aug_matrix = batch_dict['img_aug_matrix'] # (B, V, 4, 4)

        with torch.no_grad():
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug'] # torch.Size([2, 6, 4, 4]) 从激光雷达空间到图像空间的变换矩阵
            # get mapping points in image space
            points_3d, points_2d, map_mask = map_points(
                points, lidar2image, image_aug_matrix, batch_size, (h, w)) # torch.Size([2, 6, 129600, 20, 4]) torch.Size([2, 6, 129600, 20, 2]) torch.Size([2, 6, 129600, 20, 1])
            mapped_points_2d = points_2d[map_mask] # torch.Size([3779754, 2])
            mapped_points_3d = points_3d[map_mask] # torch.Size([3779754, 4])
            mapped_view_cnts = map_mask.view(
                batch_size, num_view, -1).sum(-1).view(-1).int() # torch.Size([batch * 6]) 每个batch中每个视角的点的数量
            mapped_points = torch.cat(
                [mapped_points_2d, torch.zeros_like(mapped_points_2d[:, :1])], dim=-1) # torch.Size([3779754, 3]) 在z轴上增加一个0
            mapped_coords_3d = mapped_points_3d[:, :3]

            # shape (H*W,2), [[x1,y1],...]
            patch_coords_perimage = batch_dict['patch_coords'][batch_dict['patch_coords'][:, 0] == 0, 2:].clone(
            ).float() # [33792, 4] -> [2816, 2] 从patch_coords中取出每个图像块的坐标
            patch_coords_perimage[:, 0] = (
                patch_coords_perimage[:, 0] + 0.5) / batch_dict['hw_shape'][1] # 归一化到图像坐标系下的坐标
            patch_coords_perimage[:, 1] = (
                patch_coords_perimage[:, 1] + 0.5) / batch_dict['hw_shape'][0] # 归一化到图像坐标系下的坐标1

            # get image patch coords
            patch_points = patch_coords_perimage.unsqueeze(
                0).repeat(batch_size * num_view , 1, 1).view(-1, 2) # torch.Size([33792, 2])
            patch_points = torch.cat( # torch.Size([33792, 3])
                [patch_points, torch.zeros_like(patch_points[:, :1])], dim=-1)
            patch_view_cnts = (torch.ones_like(
                mapped_view_cnts) * (batch_dict['hw_shape'][0] * batch_dict['hw_shape'][1])).int() # torch.Size([batch * 6]) 每个batch中每个视角的点的数量

            # find the nearest 3 mapping points and keep the closest
            _, idx = three_nn(patch_points.to(torch.float32), patch_view_cnts, mapped_points.to( # 每个图像块中的点到最近的三个映射点的索引
                torch.float32), mapped_view_cnts) # torch.Size([33792, 3]) torch.Size([33792, 3])
            idx = idx[:, :1].repeat(1, 3).long() #? torch.Size([33792, 3]) 为什么只保留最近的一个点的索引
            # take 3d coords of the nearest mapped point of each image patch as its 3d coords
            image2lidar_coords_xyz = torch.gather(mapped_coords_3d, 0, idx) # torch.Size([33792, 3])

            # calculate distance between each image patch and the nearest mapping point in image space
            neighbor_2d = torch.gather(mapped_points, 0, idx) # torch.Size([33792, 3])
            nearest_dist = (patch_points[:, :2]-neighbor_2d[:, :2]).abs() # torch.Size([33792, 2]) 计算图像块中心点到最近的映射点的距离
            nearest_dist[:, 0] *= batch_dict['hw_shape'][1] # torch.Size([33792, 2])
            nearest_dist[:, 1] *= batch_dict['hw_shape'][0] # torch.Size([33792, 2])

            # 3d coords -> voxel grids
            image2lidar_coords_xyz[..., 0] = (image2lidar_coords_xyz[..., 0] - self.pc_range[0]) / (
                self.pc_range[3]-self.pc_range[0]) * self.space_shape[0] - 0.5 # 归一化到激光雷达空间下的坐标
            image2lidar_coords_xyz[..., 1] = (image2lidar_coords_xyz[..., 1] - self.pc_range[1]) / (
                self.pc_range[4]-self.pc_range[1]) * self.space_shape[1] - 0.5 # 归一化到激光雷达空间下的坐标
            image2lidar_coords_xyz[..., 2] = 0.

            image2lidar_coords_xyz[..., 0] = torch.clamp(
                image2lidar_coords_xyz[..., 0], min=0, max=self.space_shape[0]-1) # 限制在激光雷达空间的范围内 0 -> 360
            image2lidar_coords_xyz[..., 1] = torch.clamp(
                image2lidar_coords_xyz[..., 1], min=0, max=self.space_shape[1]-1) # 限制在激光雷达空间的范围内 0 -> 360

            # reorder to z,y,x
            image2lidar_coords_zyx = image2lidar_coords_xyz[:, [2, 1, 0]] # torch.Size([33792, 3])
        if self.accelerate:
            self.cache = (image2lidar_coords_zyx, nearest_dist)
        return image2lidar_coords_zyx, nearest_dist # torch.Size([33792, 3]) torch.Size([33792, 2])


class MapLidar2Image(nn.Module):
    '''Map Lidar points to image space'''

    def __init__(self, model_cfg, accelerate=False, use_map=False, use_denoise=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg.point_cloud_range
        self.voxel_size = model_cfg.voxel_size
        self.sample_num = model_cfg.sample_num
        self.space_shape = [
            int((self.pc_range[i+3]-self.pc_range[i])/self.voxel_size[i]) for i in range(3)]
        self.accelerate = accelerate
        if self.accelerate:
            raise NotImplementedError
            self.full_lidar2image_coors_zyx = None
            # only support one point in a voxel
            self.points = get_points(
                self.pc_range, self.sample_num, self.space_shape).cuda()
        self.use_map = use_map
        self.use_denoise = use_denoise

    def pre_compute(self, batch_dict):
        '''Precalculate the coords of all voxels mapped on the image'''
        batch_dict = self._ensure_lidar2image_and_aug(batch_dict)
        image = batch_dict['camera_imgs']
        lidar2image = batch_dict['lidar2image']
        image_aug_matrix = batch_dict['img_aug_matrix']
        hw_shape = batch_dict['hw_shape']

        image_shape = image.shape[-2:]
        assert image.shape[0] == 1, 'batch size should be 1 in pre compute'
        batch_idx = torch.zeros(
            self.space_shape[0]*self.space_shape[1], device=image.device)
        with torch.no_grad():
            # get reference points, only in voxels.
            points = self.points.clone()
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug']
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, image_aug_matrix, batch_idx, image_shape)

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
            self.full_lidar2image_coors_zyx = lidar2image_coords_xyz[:, [
                2, 0, 1]]

    def map_lidar2image(self, points, lidar2image, image_aug_matrix, batch_idx, image_shape, coords_ori,agent_num = None,num_view = None):
        '''Map Lidar points to image space.

        Args:
            points (tensor): batch lidar points shape (voxel num, sample num,4).
            lidar2image (tensor): Transformation matrix from lidar to image space, shape (B, N, 4, 4).
            image_aug_matrix (tensor): Transformation of image augmentation, shape (B, N, 4, 4).
            batch_idx (tensor): batch id for all points in batch
            image_shape (Tuple(int, int)): Image shape, (height, width).

        Returns:
            batch_hit_points: 2d coordinates of lidar points mapped in image space. 
        '''
        
        if not agent_num and not num_view:
            points_batch = points.float()
            lidar2image_batch = lidar2image.float()
            image_aug_matrix_batch = image_aug_matrix.float()
            # hit_points, pixel_ori = map_points_v2(points_batch, lidar2image_batch, image_aug_matrix_batch, 1, (image_shape[0], image_shape[1]), 0)
            
            # TODO: 在这里改
            hit_points_raw = map_points_v2_cuda(points_batch, lidar2image_batch, image_aug_matrix_batch, 1, image_shape[0], image_shape[1], 0)
            hit_points = hit_points_raw[0].squeeze(1)
            return hit_points
        else:
            # 原有的循环逻辑：按batch分别处理
            batch_size = (batch_idx[-1] + 1).int() 
            batch_hit_points = []
            for b in range(batch_size):
                # 确保数据类型为Float，避免混合精度训练中的类型不匹配问题
                points_batch = points[batch_idx == b].float()
                lidar2image_batch = lidar2image[b:b+1].float()
                image_aug_matrix_batch = image_aug_matrix[b:b+1].float()
                
                hit_points = map_points_v2_cuda(points_batch, lidar2image_batch, image_aug_matrix_batch, 1, image_shape[0], image_shape[1], 0)[0].squeeze(1)
                
                batch_hit_points.append(hit_points)
            batch_hit_points = torch.cat(batch_hit_points, dim=0)
            return batch_hit_points

    def map_lidar2image_multi_view(self, points, lidar2image, image_aug_matrix, batch_idx, image_shape):
        """Map lidar voxels to all valid camera views with CUDA.

        Returns:
            batch_hit_points: [num_hits, 3], each row is [u, v, view_idx].
            point_ids: [num_hits], linear point ids inside the provided ``points`` tensor.
        """
        points_batch = points.float()
        lidar2image_batch = lidar2image.float()
        image_aug_matrix_batch = image_aug_matrix.float()
        sample_num = points_batch.shape[1]

        _, points_2d, map_mask = map_points_cuda(
            points_batch,
            lidar2image_batch,
            image_aug_matrix_batch,
            1,
            image_shape[0],
            image_shape[1],
            0
        )

        points_2d = points_2d.squeeze(0)  # [num_view, grid_num, sample_num, 2]
        map_mask = map_mask.squeeze(0)    # [num_view, grid_num, sample_num]
        hit_indices = torch.nonzero(map_mask, as_tuple=False)

        if hit_indices.numel() == 0:
            empty_hit_points = points_batch.new_empty((0, 3))
            empty_point_ids = torch.empty((0,), device=points_batch.device, dtype=torch.long)
            return empty_hit_points, empty_point_ids

        view_ids = hit_indices[:, 0].long()
        grid_ids = hit_indices[:, 1].long()
        sample_ids = hit_indices[:, 2].long()

        hit_coords = points_2d[view_ids, grid_ids, sample_ids]
        hit_coords = torch.clamp(hit_coords, -1.0, 2.0) + 1.0
        hit_points = torch.cat(
            [hit_coords, view_ids.to(hit_coords.dtype).unsqueeze(1)],
            dim=1
        )
        point_ids = grid_ids * sample_num + sample_ids
        return hit_points, point_ids.long()

    def forward(self, batch_dict, use_multi_scale=False, space_shape=None):
        '''Get the coordinates of lidar poins in image space.

        Returns:
            lidar2image_coords_zyx (tensor): The coordinates of lidar points in 3D space.
        '''
        batch_dict = self._ensure_lidar2image_and_aug(batch_dict)
        img = batch_dict['camera_imgs'] # torch.Size([batch_size, view, 3, H, W])
        coords = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone() # torch.Size([num_of_voxel(39590), 4]) batch_idx, x, y, z
        if 'ori_coords_height' in batch_dict:
        #    ori_coords_height = (batch_dict['ori_coords_height'] + 0.5).to(torch.int32).clamp(min=0, max=31).reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
           ori_coords_height = batch_dict['ori_coords_height'].reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31

           coords = torch.cat([coords[:, :-1], ori_coords_height], dim=1)
        
        # 将相机外参从 agent 对齐到 ego：先把 ego 点转换到 agent 坐标，再用相机外参
        
        lidar2image = batch_dict['lidar2image']
        img_aug_matrix = batch_dict['img_aug_matrix']
        hw_shape = batch_dict['hw_shape'] # (32, 88) image shape / 8
        img_shape = img.shape[-2:] # (256, 704)
        batch_idx = coords[:, 0] # torch.Size([num_of_voxel])
        with torch.no_grad():
            # get reference points, only in voxels.
            space_shape = self.space_shape
            points = get_points(self.pc_range, self.sample_num,
                                space_shape, coords[:, 1:]) # 获取每个体素中的采样点 torch.Size([num_of_voxel, 1, 3])                    
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug'] # torch.Size([2, 6, 4, 4])
            # get mapping points in image space
            # print(f"points shape: {points.shape}")
            # print(f"lidar2image shape: {lidar2image.shape}")
            # print(f"img_aug_matrix shape: {img_aug_matrix.shape}"))
            lidar2image_coords_xyz, voxel_hit_indices = self.map_lidar2image_multi_view(
                points, lidar2image, img_aug_matrix, batch_idx, img_shape)
            voxel_hit_indices = voxel_hit_indices.to(coords.device)
            # [hit_u_clamped, hit_v_clamped, first_hit_view_idx]
            # hit_u_clamped = clamp(u, -1.0, 2.0) + 1.0   → 范围变为 [0, 3]
            # hit_v_clamped = clamp(v, -1.0, 2.0) + 1.0   → 范围变为 [0, 3]
            # 实际上对于图像内的有效点，u,v ∈ [0,1]，clamp 后 +1 → [1, 2]


            # DEBUG: 打印投影后的归一化坐标分布（在乘以hw_shape之前）
            # if lidar2image_coords_xyz.shape[0] > 0:
            #     norm_x = lidar2image_coords_xyz[:, 0].cpu().numpy()
            #     norm_y = lidar2image_coords_xyz[:, 1].cpu().numpy()
            #     view_idx = lidar2image_coords_xyz[:, 2].cpu().numpy()
            #     print(f"投影后归一化坐标分布:")
            #     print(f"  x范围: [{norm_x.min():.4f}, {norm_x.max():.4f}], 均值={norm_x.mean():.4f}, 中位数={np.median(norm_x):.4f}")
            #     print(f"  y范围: [{norm_y.min():.4f}, {norm_y.max():.4f}], 均值={norm_y.mean():.4f}, 中位数={np.median(norm_y):.4f}")
            #     print(f"  view_idx范围: [{view_idx.min():.0f}, {view_idx.max():.0f}], 唯一值={np.unique(view_idx).shape[0]}个")

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1] # 归一化到图像坐标系下的坐标
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]  # 归一化到图像坐标系下的坐标
            
            lidar2image_coords_zyx = lidar2image_coords_xyz[:, [2, 0, 1]] # torch.Size([num_hits, 3]) view_idx, x, y
        if use_multi_scale: 
            use_multi_name_list = ['x_conv3'] # ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']
            if 'ori_coords_height' in batch_dict:
                use_multi_name_coords_list = ['ori_coords_height_coords3'] # ['ori_coords_height_coords1', 'ori_coords_height_coords2', 'ori_coords_height_coords3', 'ori_coords_height_coords4']
            lidar2image_coords_zyx_list = []
            for i, name in enumerate(use_multi_name_list):
                indices = batch_dict['multi_scale_3d_features'][name].indices[:, [0, 3, 2, 1]].clone()
                if 'ori_coords_height' in batch_dict:
                    # ori_coords_height_tmp = (batch_dict[use_multi_name_coords_list[i]] + 0.5).to(torch.int32).clamp(min=0, max=31).reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
                    ori_coords_height_tmp = batch_dict[use_multi_name_coords_list[i]].reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
                    indices = torch.cat([indices[:, :-1], ori_coords_height_tmp], dim=1)
                    space_shape = [704, 200, 32]
                else:
                    space_shape = batch_dict['multi_scale_3d_features'][name].spatial_shape[::-1]
                with torch.no_grad():
                    points = get_points(self.pc_range, self.sample_num, space_shape, indices[:, 1:])
                    lidar2image_coords_xyz = self.map_lidar2image(
                        points, lidar2image, img_aug_matrix, indices[:, 0], img_shape, indices)
                    lidar2image_coords_xyz[:, 0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
                    lidar2image_coords_xyz[:, 1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
                    lidar2image_coords_zyx_tmp = lidar2image_coords_xyz[:, [2, 0, 1]]
                    lidar2image_coords_bzyx = torch.cat([indices[:, 0:1], lidar2image_coords_zyx_tmp], dim=1)
                lidar2image_coords_zyx_list.append(lidar2image_coords_bzyx)
            return lidar2image_coords_zyx, lidar2image_coords_zyx_list, voxel_hit_indices
        return lidar2image_coords_zyx, None, voxel_hit_indices #TODO to fix 

    def _ensure_lidar2image_and_aug(self, batch_dict):
        """Populate batch_dict['lidar2image'] and ['img_aug_matrix'] from nested
        batch_merged_cam_inputs when not present. Supports agents_as_views collapsing.
        Expected keys under batch_merged_cam_inputs: 'imgs', 'intrinsics', 'extrinsics',
        'post_rots', 'post_trans'.
        extrinsics can be cam_from_lidar (T_cam_lidar) or lidar_from_cam (T_lidar_cam).
        If 'EXTRINSICS_IS_LIDAR_TO_CAM' flag is present in batch_dict or model_cfg,
        it will be used to disambiguate; otherwise defaults to cam_from_lidar if reasonable.
        """
        if ('lidar2image' in batch_dict) and ('img_aug_matrix' in batch_dict):
            return batch_dict
        if 'batch_merged_cam_inputs' not in batch_dict:
            return batch_dict
        cam_inputs = batch_dict['batch_merged_cam_inputs']
        imgs = cam_inputs.get('imgs', None)
        Ks = cam_inputs.get('intrinsics', None)
        # Ks = Ks # 缩放因子 TODO：是否需要考虑缩放因子
        Ext = cam_inputs.get('extrinsics', None)
        post_rots = cam_inputs.get('post_rots', None)
        post_trans = cam_inputs.get('post_trans', None)
        assert imgs is not None and Ks is not None and Ext is not None, 'Missing camera imgs/intrinsics/extrinsics'
        B, N = imgs.shape[0], imgs.shape[1]
        device = imgs.device
        # Ensure 4x4 intrinsics (homogeneous)
        if Ks.shape[-2:] == (3, 3):
            Ks_4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            Ks_4[:, :, :3, :3] = Ks
        else:
            Ks_4 = Ks
        # print("--------------------------------*************************")
        # print(f"Ks_4 shape: {Ks_4.shape}")
        # Ensure 4x4 extrinsics
        if Ext.shape[-2:] != (4, 4):
            Ext_4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            Ext_4[:, :, :3, :3] = Ext
            Ext_4[:, :, :3, 3] = Ext[:, :, :, 3] if Ext.dim() == 4 else Ext[:, :, 3]
        else:
            Ext_4 = Ext
        Ext_4 = torch.inverse(Ext_4)

        # Extrinsics is T_cam_lidar (camera from lidar), same as where2comm
        # No need to check EXTRINSICS_IS_LIDAR_TO_CAM, use directly
        # Ext_4 is already T_cam_lidar, same definition as where2comm's rots and trans
        # Ensure 4x4 post_rots and post_trans
        if post_rots.shape[-2:] == (3, 3):
            post_rots_4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            post_rots_4[:, :, :3, :3] = post_rots
        else:
            post_rots_4 = post_rots
        if post_trans.shape[-1] == 3:
            post_trans_4 = torch.zeros(B, N, 4, device=device)
            post_trans_4[:, :, :3] = post_trans
            post_trans_4[:, :, 3] = 1
        else:
            post_trans_4 = post_trans
        A_post = torch.eye(4, device=device).view(1,1,4,4).repeat(B,N,1,1)
        A_post[:, :, :3, :3] = post_rots
        A_post[:, :, :3,  3] = post_trans
        
        # Compute lidar2image transformation
        # lidar2image = post_rots_4 @ (Ks_4 @ Ext_4)
        Ks_Ext = torch.bmm(Ks_4.view(-1, 4, 4), Ext_4.view(-1, 4, 4))  # K @ T_cam_lidar
        lidar2image = Ks_Ext
        
        # DEBUG: 验证lidar2image矩阵计算
        # print(f"[_ensure_lidar2image_and_aug] lidar2image矩阵计算验证:")
        # print(f"  Ks_4 shape: {Ks_4.shape}, Ext_4 shape: {Ext_4.shape}, post_rots_4 shape: {post_rots_4.shape}")
        # print(f"  计算流程: lidar2image = post_rots_4 @ (Ks_4 @ Ext_4)")
        '''
        if B == 1 and N > 0:
            # 打印第一个view的各个矩阵
            Ks_sample = Ks_4[0, 0].cpu().numpy()
            Ext_sample = Ext_4[0, 0].cpu().numpy()
            post_rots_sample = post_rots_4[0, 0].cpu().numpy()
            lidar2image_sample = lidar2image[0, 0].cpu().numpy()
            print(f"  第一个view (batch 0, view 0)的矩阵:")
            print(f"    Ks_4 (内参):\n{Ks_sample}")
            print(f"    Ext_4 (外参 T_cam_lidar):\n{Ext_sample}")
            print(f"    post_rots_4 (图像增强):\n{post_rots_sample}")
            print(f"    lidar2image (最终):\n{lidar2image_sample}")
            # 验证矩阵有效性
            if lidar2image_sample.shape == (4, 4):
                is_valid = np.allclose(lidar2image_sample[3, :], [0,0,0,1])
                print(f"    矩阵有效性检查: 最后一行是否为[0,0,0,1] = {is_valid}")
                if not is_valid:
                    print(f"    ⚠️  警告: lidar2image矩阵最后一行不是[0,0,0,1]，实际为: {lidar2image_sample[3, :]}")
        '''

        # Compute image augmentation matrix
        img_aug_matrix = A_post.view(1, -1, 4, 4)
        # 如果camera_imgs是[1, B*N]格式（BN合并），将lidar2image也合并成[1, B*N, 4, 4]
        # 这样可以与camera_imgs的shape保持一致，方便后续处理
        lidar2image = lidar2image.view(1, -1, 4, 4)

        if 'agent_to_ego_transform' in batch_dict:
            agent_to_ego = batch_dict['agent_to_ego_transform']  # [B, 4, 4]
            ego_to_agent = torch.inverse(agent_to_ego) # [B,1,4,4]
            # print(f"  ego_to_agent shape: {ego_to_agent.shape}") [1,30,4,4]
            # print(f"  lidar2image shape: {lidar2image.shape}") [1,30,4,4]
            lidar2image_lidar = torch.matmul(lidar2image, ego_to_agent)
        batch_dict['lidar2image'] = lidar2image_lidar
        batch_dict['lidar2image_cam'] = lidar2image
        batch_dict['img_aug_matrix'] = img_aug_matrix
        return batch_dict


class MapLidar2Image2(nn.Module):
    '''Map Lidar points to image space'''

    def __init__(self, model_cfg, accelerate=False, use_map=False, use_denoise=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg.point_cloud_range
        self.voxel_size = model_cfg.voxel_size
        self.sample_num = model_cfg.sample_num
        self.space_shape = [
            int((self.pc_range[i+3]-self.pc_range[i])/self.voxel_size[i]) for i in range(3)]
        self.accelerate = accelerate
        if self.accelerate:
            raise NotImplementedError
            self.full_lidar2image_coors_zyx = None
            # only support one point in a voxel
            self.points = get_points(
                self.pc_range, self.sample_num, self.space_shape).cuda()
        self.use_map = use_map
        self.use_denoise = use_denoise

    def pre_compute(self, batch_dict):
        '''Precalculate the coords of all voxels mapped on the image'''
        image = batch_dict['camera_imgs']
        lidar2image = batch_dict['lidar2image']
        image_aug_matrix = batch_dict['img_aug_matrix']
        hw_shape = batch_dict['hw_shape']

        image_shape = image.shape[-2:]
        assert image.shape[0] == 1, 'batch size should be 1 in pre compute'
        batch_idx = torch.zeros(
            self.space_shape[0]*self.space_shape[1], device=image.device)
        with torch.no_grad():
            # get reference points, only in voxels.
            points = self.points.clone()
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug']
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, image_aug_matrix, batch_idx, image_shape)

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
            self.full_lidar2image_coors_zyx = lidar2image_coords_xyz[:, [
                2, 0, 1]]

    def map_lidar2image(self, points, lidar2image, image_aug_matrix, batch_idx, image_shape, coords_ori):
        '''Map Lidar points to image space.

        Args:
            points (tensor): batch lidar points shape (voxel num, sample num,4).
            lidar2image (tensor): Transformation matrix from lidar to image space, shape (B, N, 4, 4).
            image_aug_matrix (tensor): Transformation of image augmentation, shape (B, N, 4, 4).
            batch_idx (tensor): batch id for all points in batch
            image_shape (Tuple(int, int)): Image shape, (height, width).

        Returns:
            batch_hit_points: 2d coordinates of lidar points mapped in image space. 
        '''
        # [SANITY] 输入形状与矩阵有效性断言
        assert lidar2image.ndim == 4, f"lidar2image must be 4D, got {lidar2image.ndim}D"
        assert lidar2image.shape[-2:] == (4, 4), f"lidar2image last 2 dims must be (4,4), got {lidar2image.shape[-2:]}"
        assert image_aug_matrix.ndim == 4, f"img_aug_matrix must be 4D, got {image_aug_matrix.ndim}D"
        assert image_aug_matrix.shape[-2:] == (4, 4), f"img_aug_matrix last 2 dims must be (4,4), got {image_aug_matrix.shape[-2:]}"
        
        # 检查lidar2image最后一行是否为[0,0,0,1]
        last_row = lidar2image[..., 3, :]
        expected_last_row = torch.zeros_like(last_row)
        expected_last_row[..., 3] = 1.0
        if not torch.allclose(last_row, expected_last_row, atol=1e-3):
            print(f"[SANITY] [WARN] lidar2image最后一行不是[0,0,0,1]")
            print(f"  实际值: {last_row[0, 0]}")
            print(f"  说明：可能传入的是'包含K的投影矩阵'或row/col major弄反")
        
        # 检查img_aug_matrix的平移位置
        trans_col = image_aug_matrix[..., :3, 3]  # 平移应该在最后一列的前3行
        trans_row = image_aug_matrix[..., 3, :3]  # 不应该在最后一行的前3列
        if torch.any(trans_row.abs() > 1e-3):
            print(f"[SANITY] [WARN] img_aug_matrix的平移可能放错位置")
            print(f"  [:3,3] (正确位置) 非零值: {torch.any(trans_col.abs() > 1e-3)}")
            print(f"  [3,:3] (错误位置) 非零值: {torch.any(trans_row.abs() > 1e-3)}")
            print(f"  约定：齐次变换用最后一列做平移")
        
        num_view = lidar2image.shape[1] # 6
        batch_size = (batch_idx[-1] + 1).int() 
        batch_hit_points = []
        for b in range(batch_size):
            if self.use_denoise:
                _, points_2d, map_mask = map_points( # 将点从激光雷达坐标系映射到图像坐标系 torch.Size([1, 6, num_of_voxel, 1, 2]) torch.Size([1, 6, num_of_voxel, 1])
                    points[batch_idx == b], lidar2image[b:b+1], image_aug_matrix[b:b+1], 1, image_shape, expand_scale=0.2)
            else:
                _, points_2d, map_mask = map_points( # 将点从激光雷达坐标系映射到图像坐标系 torch.Size([1, 6, num_of_voxel, 1, 2]) torch.Size([1, 6, num_of_voxel, 1])
                    points[batch_idx == b], lidar2image[b:b+1], image_aug_matrix[b:b+1], 1, image_shape)
            points_2d = points_2d.squeeze(3) # torch.Size([1, 6, num_of_voxel, 2])
            # set point not hit image as hit 0
            map_mask = map_mask.squeeze(3).permute(0, 2, 1).view(-1, num_view) # torch.Size([num_of_voxel, 6])
            hit_mask = map_mask.any(dim=-1) # torch.Size([num_of_voxel]) 有一个视角命中就算命中

            map_mask[~hit_mask, 0] = True # torch.Size([num_of_voxel, 6]) 不命中的都放视角0
            # get hit view id
            hit_view_ids = torch.nonzero(map_mask) # torch.Size([num_of_hitvoxel, 2]) 第一列是batch_idx，第二列是view_idx，由于一个voxel可能被多个视角命中，所以会有多行
            # select first view if hit multi view
            hit_poins_id = hit_view_ids[:, 0] # torch.Size([num_of_hitvoxel]) 
            shift_hit_points_id = torch.roll(hit_poins_id, 1) # torch.Size([num_of_hitvoxel])
            shift_hit_points_id[0] = -1
            first_mask = (hit_poins_id - shift_hit_points_id) > 0 # torch.Size([num_of_hitvoxel]) 保留第一个命中的视角
            unique_hit_view_ids = hit_view_ids[first_mask, 1:] # torch.Size([num_of_hitvoxel, 1]) 保留第一个命中的视角的voxel索引
            num = points_2d.shape[2] # num_of_voxel
            assert len(unique_hit_view_ids) == num, 'some points not hit view!'
            # get coords in hit view
            points_2d = points_2d.permute(0, 2, 1, 3).flatten(0, 1) # torch.Size([num_of_voxel, 6, 2]) 
            hit_points_2d = points_2d[range( # torch.Size([num_of_voxel, 2]) 保留命中的视角的点
                num), unique_hit_view_ids.squeeze()]
            # if self.use_denoise and hit_mask.sum() < num:
            #     coords_ori_current_batch = coords_ori[batch_idx == b]
            #     hit_points_2d[~hit_mask] = (coords_ori_current_batch[~hit_mask, :2] - 180) / coords_ori_current_batch.new_tensor([image_shape[1], image_shape[0]])
            #     unique_hit_view_ids[~hit_mask] = 6
            # clamp value range and adjust to postive for set partition
            hit_points_2d = torch.clamp(hit_points_2d, -1, 2) + 1 # torch.Size([num_of_voxel, 2])
            hit_points = torch.cat([hit_points_2d, unique_hit_view_ids], -1) # torch.Size([num_of_voxel, 3])
            batch_hit_points.append(hit_points)
        batch_hit_points = torch.cat(batch_hit_points, dim=0)
        return batch_hit_points

    def forward(self, batch_dict, use_multi_scale=False, space_shape=None):
        '''Get the coordinates of lidar poins in image space.

        Returns:
            lidar2image_coords_zyx (tensor): The coordinates of lidar points in 3D space.
        '''
        if self.accelerate:
            raise NotImplementedError
            if self.full_lidar2image_coors_zyx is None:
                self.pre_compute(batch_dict)
            # accelerate by index table
            coords_xyz = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone()
            unique_index = coords_xyz[:, 1] * \
                self.space_shape[1] + coords_xyz[:, 2]
            lidar2image_coords_zyx = self.full_lidar2image_coors_zyx[unique_index.long(
            )]
            return lidar2image_coords_zyx
        batch_dict = self._ensure_lidar2image_and_aug(batch_dict)
        img = batch_dict['camera_imgs'] # torch.Size([batch_size, view, 3, H, W])
        coords = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone() # torch.Size([num_of_voxel(39590), 4]) batch_idx, x, y, z
        if 'ori_coords_height' in batch_dict:
        #    ori_coords_height = (batch_dict['ori_coords_height'] + 0.5).to(torch.int32).clamp(min=0, max=31).reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
           ori_coords_height = batch_dict['ori_coords_height'].reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
           space_shape = [360, 360, 32]
           coords = torch.cat([coords[:, :-1], ori_coords_height], dim=1)
        lidar2image = batch_dict['lidar2image']
        img_aug_matrix = batch_dict['img_aug_matrix']
        hw_shape = batch_dict['hw_shape'] # (32, 88) image shape / 8

        img_shape = img.shape[-2:] # (256, 704)
        batch_idx = coords[:, 0] # torch.Size([num_of_voxel])
        with torch.no_grad():
            # get reference points, only in voxels.
            space_shape = self.space_shape if space_shape is None else space_shape
            points = get_points(self.pc_range, self.sample_num,
                                space_shape, coords[:, 1:]) # 获取每个体素中的采样点 torch.Size([num_of_voxel, 1, 3])
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug'] # torch.Size([2, 6, 4, 4])
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, img_aug_matrix, batch_idx, img_shape, coords) # torch.Size([num_of_voxel, 3])

            # DEBUG: 打印投影后的归一化坐标分布（在乘以hw_shape之前）
            if lidar2image_coords_xyz.shape[0] > 0:
                norm_x = lidar2image_coords_xyz[:, 0].cpu().numpy()
                norm_y = lidar2image_coords_xyz[:, 1].cpu().numpy()
                view_idx = lidar2image_coords_xyz[:, 2].cpu().numpy()
                print(f"[unitr_utils] 投影后归一化坐标分布:")
                print(f"  x范围: [{norm_x.min():.4f}, {norm_x.max():.4f}], 均值={norm_x.mean():.4f}, 中位数={np.median(norm_x):.4f}")
                print(f"  y范围: [{norm_y.min():.4f}, {norm_y.max():.4f}], 均值={norm_y.mean():.4f}, 中位数={np.median(norm_y):.4f}")
                print(f"  view_idx范围: [{view_idx.min():.0f}, {view_idx.max():.0f}], 唯一值={np.unique(view_idx).shape[0]}个")
                # 统计超出[0,1]范围的点数
                x_out_of_range = ((norm_x < 0) | (norm_x > 1)).sum()
                y_out_of_range = ((norm_y < 0) | (norm_y > 1)).sum()
                if x_out_of_range > 0 or y_out_of_range > 0:
                    print(f"  超出[0,1]范围: x有{x_out_of_range}个({x_out_of_range/norm_x.shape[0]*100:.2f}%), y有{y_out_of_range}个({y_out_of_range/norm_y.shape[0]*100:.2f}%)")
                    # 打印超出范围的坐标样本
                    out_mask = (norm_x < 0) | (norm_x > 1) | (norm_y < 0) | (norm_y > 1)
                    if out_mask.sum() > 0:
                        out_samples = lidar2image_coords_xyz[out_mask][:10].cpu().numpy()
                        print(f"  超出范围坐标样本(前10个): {out_samples}")

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1] # 归一化到图像坐标系下的坐标
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]  # 归一化到图像坐标系下的坐标
            
            # DEBUG: 打印乘以hw_shape后的坐标分布
            if lidar2image_coords_xyz.shape[0] > 0:
                x_coords = lidar2image_coords_xyz[:, 0].cpu().numpy()
                y_coords = lidar2image_coords_xyz[:, 1].cpu().numpy()
                print(f"[unitr_utils] 乘以hw_shape后坐标分布:")
                print(f"  x范围: [{x_coords.min():.1f}, {x_coords.max():.1f}], 均值={x_coords.mean():.1f}, 中位数={np.median(x_coords):.1f}, hw_shape[1]={hw_shape[1]}")
                print(f"  y范围: [{y_coords.min():.1f}, {y_coords.max():.1f}], 均值={y_coords.mean():.1f}, 中位数={np.median(y_coords):.1f}, hw_shape[0]={hw_shape[0]}")
                # 统计超出hw_shape范围的点数
                x_out = (x_coords >= hw_shape[1]).sum()
                y_out = (y_coords >= hw_shape[0]).sum()
                if x_out > 0 or y_out > 0:
                    print(f"  超出hw_shape范围: x>={hw_shape[1]}有{x_out}个({x_out/x_coords.shape[0]*100:.2f}%), y>={hw_shape[0]}有{y_out}个({y_out/y_coords.shape[0]*100:.2f}%)")
            
            lidar2image_coords_zyx = lidar2image_coords_xyz[:, [2, 0, 1]] # torch.Size([num_of_voxel, 3]) view_idx, x, y
        if use_multi_scale: 
            use_multi_name_list = ['x_conv3'] # ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']
            if 'ori_coords_height' in batch_dict:
                use_multi_name_coords_list = ['ori_coords_height_coords3'] # ['ori_coords_height_coords1', 'ori_coords_height_coords2', 'ori_coords_height_coords3', 'ori_coords_height_coords4']
            lidar2image_coords_zyx_list = []
            for i, name in enumerate(use_multi_name_list):
                indices = batch_dict['multi_scale_3d_features'][name].indices[:, [0, 3, 2, 1]].clone()
                if 'ori_coords_height' in batch_dict:
                    # ori_coords_height_tmp = (batch_dict[use_multi_name_coords_list[i]] + 0.5).to(torch.int32).clamp(min=0, max=31).reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
                    ori_coords_height_tmp = batch_dict[use_multi_name_coords_list[i]].reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
                    indices = torch.cat([indices[:, :-1], ori_coords_height_tmp], dim=1)
                    space_shape = [360, 360, 32]
                else:
                    space_shape = batch_dict['multi_scale_3d_features'][name].spatial_shape[::-1]
                with torch.no_grad():
                    points = get_points(self.pc_range, self.sample_num, space_shape, indices[:, 1:])
                    lidar2image_coords_xyz = self.map_lidar2image(
                        points, lidar2image, img_aug_matrix, indices[:, 0], img_shape, indices)
                    lidar2image_coords_xyz[:, 0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
                    lidar2image_coords_xyz[:, 1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
                    lidar2image_coords_zyx_tmp = lidar2image_coords_xyz[:, [2, 0, 1]]
                    lidar2image_coords_bzyx = torch.cat([indices[:, 0:1], lidar2image_coords_zyx_tmp], dim=1)
                lidar2image_coords_zyx_list.append(lidar2image_coords_bzyx)
            return lidar2image_coords_zyx, lidar2image_coords_zyx_list
        return lidar2image_coords_zyx, None

    def _ensure_lidar2image_and_aug(self, batch_dict):
        """Populate batch_dict['lidar2image'] and ['img_aug_matrix'] from nested
        batch_merged_cam_inputs when not present. Supports agents_as_views collapsing.
        Expected keys under batch_merged_cam_inputs: 'imgs', 'intrinsics', 'extrinsics',
        'post_rots', 'post_trans'.
        extrinsics can be cam_from_lidar (T_cam_lidar) or lidar_from_cam (T_lidar_cam).
        If 'EXTRINSICS_IS_LIDAR_TO_CAM' flag is present in batch_dict or model_cfg,
        it will be used to disambiguate; otherwise defaults to cam_from_lidar if reasonable.
        """
        if ('lidar2image' in batch_dict) and ('img_aug_matrix' in batch_dict):
            return batch_dict
        if 'batch_merged_cam_inputs' not in batch_dict:
            return batch_dict
        cam_inputs = batch_dict['batch_merged_cam_inputs']
        imgs = cam_inputs.get('imgs', None)
        Ks = cam_inputs.get('intrinsics', None)
        Ext = cam_inputs.get('extrinsics', None)
        post_rots = cam_inputs.get('post_rots', None)
        post_trans = cam_inputs.get('post_trans', None)
        assert imgs is not None and Ks is not None and Ext is not None, 'Missing camera imgs/intrinsics/extrinsics'
        B, N = imgs.shape[0], imgs.shape[1]
        device = imgs.device
        # Ensure 4x4 intrinsics (homogeneous)
        if Ks.shape[-2:] == (3, 3):
            Ks_4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            Ks_4[:, :, :3, :3] = Ks
        else:
            Ks_4 = Ks
        # Ensure 4x4 extrinsics
        if Ext.shape[-2:] != (4, 4):
            raise ValueError('Extrinsics must be 4x4')
        # Extrinsics is T_cam_lidar (camera from lidar), same as where2comm
        # No need to check EXTRINSICS_IS_LIDAR_TO_CAM, use directly
        # Ext is already T_cam_lidar, same definition as where2comm's rots and trans
        T_cam_lidar = Ext
        # lidar2image = K @ T_cam_lidar
        lidar2image = torch.matmul(Ks_4, T_cam_lidar)
        # Build image augmentation matrix from post_rots/trans if available
        if post_rots is not None and post_trans is not None:
            # post_rots [B, N, 3, 3], post_trans [B, N, 3]
            aug = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            aug[:, :, :3, :3] = post_rots
            aug[:, :, :3, 3] = post_trans
            img_aug_matrix = aug
        else:
            img_aug_matrix = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # Handle agents_as_views collapsing
        agents_as_views = batch_dict.get('agents_as_views', False)
        if agents_as_views:
            lidar2image = lidar2image.view(1, B * N, 4, 4)
            img_aug_matrix = img_aug_matrix.view(1, B * N, 4, 4)
        batch_dict['lidar2image'] = lidar2image
        batch_dict['img_aug_matrix'] = img_aug_matrix
        return batch_dict