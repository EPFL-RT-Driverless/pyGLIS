# pyGLIS

This package is a Python re-implementation of the GLIS algorithm for derivative-free global optimization using surrogate functions developed by A. Bemporad.
You can find the original paper and implementation [here](http://cse.lab.imtlucca.it/~bemporad/publications).

This re-implementation was intended to be used for parameter tuning for the autonomous driving controllers (primary Stanley and MPC based) created at the EPFL Racing Team.

## Features
The main features are the same as the GLIS implementation in the [version 2.4](http://cse.lab.imtlucca.it/~bemporad/glis/download/glis_v2.4.zip): 
- global minimization of a general function
- support for box, linear and/or nonlinear constraints
- support for Particle Swarm Optimization (PSO) or DIRECT algorithms for solving the subproblems

We have not (yet) implemented the GLISp and C-GLIS(p) variants since they were not of primary interest for our application.

## Future changes
- switch the PSO library to [pyswarms](https://pyswarms.readthedocs.io/en/latest/index.html)
- add data serialization after each iteration of the algorithm to allow the procedure to restart if there was a crash due to an external problem (e.g. sever shutdown) during the execution.

