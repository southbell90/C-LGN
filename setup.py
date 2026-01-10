from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Minimal setup.py so users can do: pip install -e .
# This builds the CUDA extension (requires NVCC + a CUDA-enabled PyTorch).

setup(
    name='convdifflogic_pytorch',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='convdifflogic_cuda',
            sources=[
                'convdifflogic_pytorch/cuda/convdifflogic.cpp',
                'convdifflogic_pytorch/cuda/convdifflogic_kernel.cu',
            ],
            extra_compile_args={'nvcc': ['-lineinfo']},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch>=1.6.0', 'numpy'],
)