#!/usr/bin/env python

from setuptools import setup

setup(name='Voodoo',
      version='1.0',
      description='Neural network cloud droplet retrieval',
      author='Willi Schimmel',
      author_email='willi.schimmel@uni-leipzig.de',
      url='https://github.com/remsens-lim/Voodoo',
      license='MIT',
      packages=[],
      install_requires=[
          'torch', 
          'xarray',
          'rpgpy',
          'toml',
      ],
     )
