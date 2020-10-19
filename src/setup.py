#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from distutils.core import setup, Extension

info = {}
with open("astrocv/info.py", "r") as finfo:
    exec(finfo.read(), info)

readme = []
with open("README.md", "r") as freadme:
    readme = list(filter(lambda l: len(l.replace('=', '').strip()) > 0, freadme.readlines()))

description, long_description = info["__package_name__"], ""
if len(readme) > 0:
    description, long_description = readme[0].strip(), "".join(readme[1:])

USE_SSE = True
USE_OMP = True
USE_OPTIMIZATION = False


extra_compile_args = []
extra_link_args = []

libraries = []
define_macros = [('USE_SSE', int(USE_SSE)),
                 ('USE_OMP', int(USE_OMP))]

if USE_OPTIMIZATION:
    extra_compile_args += ['-O2', '-Ofast', '-falign-loops',
                           '-ffinite-math-only', '-finline', '-fpeel-loops',
                           '-fprefetch-loop-arrays', '-ftree-loop-optimize']

if USE_OMP:
    if sys.platform in ['win32']:
        extra_compile_args += ['-openmp']
    else:
        libraries += ['gomp']
        extra_compile_args += ['-fopenmp']
        extra_link_args += ['-fopenmp']


castrocvmodule = Extension(name='castrocv',
                           sources=['astrocv/castrocv/castrocvmodule.c',
                                    'astrocv/castrocv/image_info.c',
                                    'astrocv/castrocv/image_fun.c',
                                    'astrocv/castrocv/search_objects.c',
                                    'astrocv/castrocv/matrix.c',
                                    'astrocv/castrocv/graphics.c',
                                    'astrocv/castrocv/sampling.c',
                                    'astrocv/castrocv/convolve.c'],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         define_macros=define_macros,
                           libraries=libraries,
                         )


'''ctrackermodule = Extension(name='ctracker',
                           sources=['astrocv/ctracker/ctrackermodule.cpp',
                                    'astrocv/ctracker/Tracker.cpp',
                                    'astrocv/ctracker/track.cpp',
                                    'astrocv/ctracker/KalmanFilterMatr.cpp',
                                    'astrocv/ctracker/HungarianAlg/HungarianAlg.cpp',],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         define_macros=define_macros,
                           libraries=libraries,
                         )'''

scripts_dir = 'scripts'
scripts = []
for fname in os.listdir(scripts_dir):
    if fname.startswith('astrocv-'):
        scripts.append("%s/%s" % (scripts_dir, fname))

setup(name=info["__package_name__"],
      version=info["__version__"],
      description=description,
      long_description=long_description,
      license='MIT',
      classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
      ],
      packages=['astrocv', 'astrocv.tracker'],
      scripts=scripts,
      ext_package='astrocv',
      ext_modules=[castrocvmodule]
)
