from setuptools import setup, find_packages

setup(
    name="pyGLIS",
    version="1.0.0",
    description="A Python implementation of the derivative-free global optimization package GLIS developed by ",
    url="https://github.com/EPFL-RT-Driverless/pyGLIS",
    author="Tudor Oancea",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Private :: Do Not Upload",
    ],
    packages=find_packages(include=["pyGLIS"]),
    install_requires=["numpy", "scipy", "pyDOE", "pyswarm", "nlopt"],
)