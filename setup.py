from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy as np

from Cython.Compiler.Options import get_directive_defaults

get_directive_defaults()['binding'] = True
get_directive_defaults()['linetrace'] = True

extensions = [
    Extension("weights", ["spectacle/weights.pyx"], 
        define_macros=[('CYTHON_TRACE', '1')])
]

setup(
    name='spectacle',
    version='0.1',
    author='Christopher Lovell',
    cmdclass={'build_ext': build_ext},
    packages=['spectacle'],
    ext_modules=cythonize(extensions),
    include_dirs = [np.get_include()],
)

