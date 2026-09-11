"""Independent Gaussian representation, covariance, and projection helpers."""

from opencood.models.gaussian_modules_0822.base.covariance import (
    assemble_covariance,
    ensure_psd_symmetric,
    transform_covariance,
    transform_gaussians,
)
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_filter import (
    GaussianRangeFilter,
)
from opencood.models.gaussian_modules_0822.base.gaussian_geometry_encoder import (
    GaussianGeometryEncoder,
)
from opencood.models.gaussian_modules_0822.base.gaussian_init import (
    covariance_to_scale_quaternion,
    init_gaussian_from_axes,
)
from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    quaternion_to_rotation_matrix,
    reconstruct_covariance,
    rotation_matrix_to_quaternion,
    transform_gaussians as transform_gaussians_srq,
)
from opencood.models.gaussian_modules_0822.base.projection import (
    CAMERA_GEOMETRY_KEY,
    attach_camera_geometry,
    backproject_optical_z,
    build_camera_geometry,
    extract_agent_to_ego,
    flatten_camera_geometry,
    image_direction_to_tangent,
    lift_optical_z_to_lidar,
    project_lidar_to_image,
    projection_mats,
    transform_to_ego,
)
from opencood.models.gaussian_modules_0822.base.utils import (
    empty_long,
    principal_eigenvector_2x2,
    safe_normalize,
)

__all__ = [
    "CAMERA_GEOMETRY_KEY",
    "GaussianRangeFilter",
    "GaussianSet",
    "GaussianGeometryEncoder",
    "assemble_covariance",
    "attach_camera_geometry",
    "backproject_optical_z",
    "build_camera_geometry",
    "covariance_to_scale_quaternion",
    "ensure_psd_symmetric",
    "extract_agent_to_ego",
    "flatten_camera_geometry",
    "image_direction_to_tangent",
    "init_gaussian_from_axes",
    "lift_optical_z_to_lidar",
    "project_lidar_to_image",
    "projection_mats",
    "quaternion_to_rotation_matrix",
    "reconstruct_covariance",
    "rotation_matrix_to_quaternion",
    "transform_covariance",
    "transform_gaussians",
    "transform_gaussians_srq",
    "transform_to_ego",
    "empty_long",
    "principal_eigenvector_2x2",
    "safe_normalize",
]
