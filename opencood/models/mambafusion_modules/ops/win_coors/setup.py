import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split("."), src) for src in sources],
    )


setup(
    name="win_coors",
    cmdclass={"build_ext": BuildExtension},
    ext_modules=[
        make_cuda_ext(
            name="flattened_window_cuda",
            module="opencood.models.mambafusion_modules.ops.win_coors",
            sources=[
                "src/flattened_window.cpp",
                "src/flattened_window_kernel.cu",
            ],
        ),
    ],
)
