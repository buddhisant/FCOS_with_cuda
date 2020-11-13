import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extensions = [CUDAExtension(name="fcos_cuda.ops",
                            sources=glob.glob("./fcos_cuda/src/*"),
                            include_dirs=[os.path.abspath("./fcos_cuda/src")]),]

setup(
    name='fcos_cuda',
    version="1.0.0",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
)
