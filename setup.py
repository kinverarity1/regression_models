"""Setup script for regression_models

Based on:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

from distutils.core import setup

from regression_models import __version__

with open(path.join(path.dirname(__file__), "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()

with open(path.join(path.dirname(__file__), "README.md"), "r") as f:
    README = f.read()

setup(
    name='regression_models',

    version=__version__,

    description='Very basic wrapper around scipy\'s regression modelling functions',
    long_description=README,

    url="https://github.com/kinverarity1/regression_models",

    author="Kent Inverarity",
    author_email="kinverarity@hotmail.com",

    license="MIT",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Customer Service",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Filesystems",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],

    keywords="science maths",

    packages=["regression_models", ],

    install_requires=requirements,

    entry_points={
        'console_scripts': [
        ],
    }
    )
