from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

bbox = Extension('bbox', ['bbox.pyx'], include_dirs=[numpy.get_include()])
nms  = Extension('nms', ['nms.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([bbox, nms]))
