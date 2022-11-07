# Copyright (c) Tudor Oancea, EPFL Racing Team Driverless, 2022
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()
    # remove lines that start with #
    requirements = [
        r
        for r in requirements
        if not (r.startswith("#") or r.startswith("-e git+") or r.startswith("git+"))
    ]

setup(
    name="pyGLIS",
    version="2.2.1",
    description="A Python implementation of the derivative-free global optimization package GLIS developed by ",
    url="https://github.com/EPFL-RT-Driverless/pyGLIS",
    author="Tudor Oancea, Auguste Poiroux",
    license="MIT",
    classifiers=[
        "Development Status :: 1 - Planning",
        # "Development Status :: 2 - Pre-Alpha",
        # "Development Status :: 3 - Alpha",
        # "Development Status :: 4 - Beta",
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Private :: Do Not Upload",
    ],
    packages=find_packages(include=["pyGLIS"]),
    install_requires=requirements,
)
