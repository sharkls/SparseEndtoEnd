import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def make_cuda_ext(
    name,                   # 扩展模块名称
    module,                 # 模块路径
    sources,                # C++源文件列表
    sources_cuda=[],        # CUDA源文件列表
    extra_args=[],          # 额外编译参数
    extra_include_path=[],  # 额外包含路径
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        # # CUDA相关配置
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [     # 设置CUDA编译器(nvcc)的特定参数
            "-D__CUDA_NO_HALF_OPERATORS__",             # 禁用半精度(FP16)操作，确保兼容性
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        # 非CUDA配置
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="e2e_deformable_aggregation_ext",
        ext_modules=[
            make_cuda_ext(
                "e2e_deformable_aggregation_ext",
                module=".",
                sources=[
                    f"src/deformable_aggregation.cpp",
                    f"src/deformable_aggregation_cuda.cu",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
