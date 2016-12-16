from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy as np

# NPPATH path to numpy includes
np_path = '/'.join(np.__file__.split('/')[:-1]) + '/core/include'

# setup(name='ICython',packages=['Sierpe','Core'],
#       ext_modules=cythonize('ICython/*/*.pyx'),
#       include_dirs=[np_path])

setup(packages=['Core,Sierpe'], ext_modules=cythonize('ICython**/*.pyx'),
      include_dirs=[np_path])

#to run: python icompile.py build_ext --inplace
