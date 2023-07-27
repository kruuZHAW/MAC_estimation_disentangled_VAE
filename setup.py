#! /usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="trajectory_pairs_generation",
    version="0.0.0",
    description="Description",
    author="Timothé Krauth",
    author_email="",
    url="https://github.com/kruuZHAW",
    install_requires=[
        "pytorch-lightning",
        # "traffic",
        "numba",
        # "sphinx",
        # "sphinx_rtd_theme",
        # "sphinx-copybutton",
        # "pyvinecopulib",
    ],
    packages=find_packages(),
)
