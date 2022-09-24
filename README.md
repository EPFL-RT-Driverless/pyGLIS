# pyGLIS

This package is a Python re-implementation of the GLIS algorithm for derivative-free global optimization using surrogate functions developed by A. Bemporad.
You can find the original paper and implementation [here](http://cse.lab.imtlucca.it/~bemporad/publications).

This re-implementation was intended to be used for parameter tuning for the autonomous driving controllers (primary Stanley and MPC based) created at the EPFL Racing Team.

## Features
The main features are the same as the GLIS implementation in the [version 2.4](http://cse.lab.imtlucca.it/~bemporad/glis/download/glis_v2.4.zip): 
- global minimization of a general function
- support for box, linear and/or nonlinear constraints

We have not (yet) implemented the GLISp and C-GLIS(p) variants since they were not of primary interest for our application.

## Future changes
