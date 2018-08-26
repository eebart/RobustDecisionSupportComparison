from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
  ext_modules=cythonize(['dps.pyx',
                         'planned_adaptive.pyx',
                         'intertemporal.pyx']),
  include_dirs=[numpy.get_include()]
)
