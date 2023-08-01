from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ads_conv_cuda10',
    ext_modules=[
        CUDAExtension('ads_conv_cuda', [
            'src/ads_conv_cuda.cpp',
            'src/ads_conv_cuda_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
