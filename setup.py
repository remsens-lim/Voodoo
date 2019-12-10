
from setuptools import setup, find_packages

version = {}
with open("libVoodoo/version.py") as f:
    exec(f.read(), version)

with open('README.md') as f:
    readme = f.read()

setup(
    name='voodoo',
    version=version['__version__'],
    description='Python package for prediction of lidar backscatter and depolarization using cloud radar Doppler spectra.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Willi Schimmel',
    author_email='willi.schimmel@uni-leipzig.de',
    url='https://github.com/KarlJohnsonnn/Voodoo',
    license='MIT License',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['numpy>=1.16', 'scipy>=1.2', 'netCDF4>=1.4.2', 'tensorflow>=2.0',
                      'matplotlib>=3.0.2', 'requests>=2.21'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
