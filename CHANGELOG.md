# v2.2.3

added `COLCON_IGNORE` file to ignore this package in colcon builds (for brains repo)

# v2.2.2

moved GPyOpt and pyswarm from requirements.txt to requirements_dev.txt

# v2.2.1

fixed dimension in cobyla failure handling

# v2.2.0

Simplified a lot the interface by removing useless features:
- IDW functions
- linear and nonlinear constraints
Also fixed bugs on scaling of variables during data dumping and data loading

# v2.1.1

:heavy_minus_sign: Removed useless deps (in particular `nlopt`)

# v2.1.0

Usage of minimize method from scipy
Implement data serialization
Cleanup
# v2.0.0

updated everything to match `python_boilerplate` v2.0.2
