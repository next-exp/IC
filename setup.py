from setuptools import setup
from Cython.Build import cythonize
from os import path
from subprocess import check_output
import numpy

numpy_include = path.join(path.dirname(numpy.__file__), 'core/include')


version = ( check_output('git describe --tags --always'.split())
          .decode()
          .split("-")[0]
          .removeprefix("v")
          )

files      = "invisible_cities/*/*.pyx"
directives = dict(language_level=3, embedsignature=True)

setup(name         = 'invisible cities',
      version      = version,
      description  = 'NEXT reconstruction software',
      url          = 'https://github.com/nextic/IC',
      author       = 'NEXT collaboration',
      author_email = 'nextic@TODO.org',
      license      = 'MIT',
      packages     = ['invisible_cities'],
      ext_modules  = cythonize(files, compiler_directives=directives),
      include_dirs = [numpy.get_include()]
)
