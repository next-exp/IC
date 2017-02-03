from setuptools import setup
from Cython.Build import cythonize
from os import path
from subprocess import check_output
import numpy
numpy_include = path.join(path.dirname(numpy.__file__), 'core/include')



setup(name         = 'invisible cities',
      version      = check_output('git describe --tags --always'.split()).decode(),
      description  = 'NEXT blah blah',
      url          = 'https://github.com/nextic/IC',
      author       = 'NEXT collaboration',
      author_email = 'nextic@TODO.org',
      license      = 'TODO',
      packages     = ['invisible_cities'],
      ext_modules  = cythonize('invisible_cities**/*.pyx'),
      include_dirs = [numpy_include],
)
