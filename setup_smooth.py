from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='att_smooth',
    ext_modules=[
        CUDAExtension('att_smooth_cuda', [
            'new_src/att_smooth_cuda.cpp',
            'new_src/att_smooth_cuda_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
