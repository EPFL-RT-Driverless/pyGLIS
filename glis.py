#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless
# (C) 2019 A. Bemporad, July 6, 2019
# Solve (GL)obal optimization problems using (I)nverse distance weighting and radial basis function (S)urrogates.

import contextlib
import io
from time import perf_counter

import nlopt  # https://nlopt.readthedocs.io
import numpy as np
from pyDOE import lhs  # https://pythonhosted.org/pyDOE/
from pyswarm import pso  # https://pythonhosted.org/pyswarm/
from scipy.optimize import linprog


class GLIS:
    def __init__(
        self,
        f=None,
        nvars=1,
        lb=None,
        ub=None,
        maxevals=20,
        alpha=1,
        delta=0.5,
        nsamp=None,
        useRBF=True,
        rbf=lambda x1, x2: 1 / (1 + 0.25 * np.vecsum((x1 - x2) ** 2)),
        scalevars=True,
        svdtol=1e-6,
        Aineq=None,
        bineq=None,
        g=None,
        shrink_range=1,
        constraint_penalty=1000,
        feasible_sampling=False,
        globoptsol="direct",
        display=0,
        PSOiters=500,
        PSOswarmsize=20,
        epsDeltaF=1e-4,
        isUnknownFeasibilityConstrained=False,
        isUnknownSatisfactionConstrained=False,
        g_unkn_fun=[],
        s_unkn_fun=[],
    ):
        """
        Generates default problem structure for IDW-RBF Global Optimization.

         :param f: Callable, objective function
         :param lb: np.ndarray, lower bounds on the optimization variables
         :param ub: np.ndarray, upper bounds on the optimization variables
         :param maxevals: int, maximum number of function evaluations
         :param alpha: float, weight on function uncertainty variance measured by IDW
         :param delta: float, weight on distance from previous samples
         :param nsamp: int, number of initial samples
         :param useRBF: int, 1 = use RBFs, 0 = use IDW interpolation
         :param rbf: Callable, inverse quadratic RBF function (only used if useRBF=1)
         :param scalevars: int, 1=scale problem variables, 0 = don't scale
         :param svdtol: double, tolerance used to discard small singular values
         :param Aineq: np.ndarray, matrix A defining linear inequality constraints
         :param bineq: np.ndarray, right-hand side of constraints A*x <= b
         :param g: Callable, constraint function. Example: g = lambda x: x[0]**2+x[1]**2-1
         :param shrink_range: int, 0=disable shrinking lb and ub to bounding box of feasible set
         :param constraint_penalty: float, penalty term on violation of linear inequality and nonlinear constraint
         :param feasible_sampling: bool, if True, initial samples are forced to be feasible
         :param globoptsol: str, nonlinear solver used during acquisition. Interfaced solvers are: "direct" DIRECT from NLopt tool (nlopt.readthedocs.io) and "pswarm" PySwarm solver (pythonhosted.org/pyswarm/)
         :param display: int, verbosity level (0=minimum)
         :param PSOiters: int, number of iterations in PSO solver
         :param PSOswarmsize: int, swarm size in PSO solver
         :param epsDeltaF: float, minimum value used to scale the IDW distance function·
        """
        # default values for unspecified parameters
        if ub is None:
            ub = -np.ones((nvars, 1))
        if lb is None:
            lb = np.ones((nvars, 1))
        if nsamp is None:
            nsamp = 2 * nvars

        # Assertions
        assert globoptsol in ["direct", "pswarm"], "Unknown solver"
        assert (
            maxevals >= nsamp
        ), "Max number of function evaluations is too low. You specified {} maxevals and {} nsamp.".format(
            maxevals, nsamp
        )

        # store parameters
        self.nvars = nvars
        self.f = f
        self.lb = lb
        self.ub = ub
        self.maxevals = maxevals
        self.alpha = alpha
        self.delta = delta
        self.nsamp = nsamp
        self.useRBF = useRBF
        self.rbf = rbf
        self.scalevars = scalevars
        self.svdtol = svdtol
        self.Aineq = Aineq
        self.bineq = bineq
        self.g = g
        self.shrink_range = shrink_range
        self.constraint_penalty = constraint_penalty
        self.feasible_sampling = feasible_sampling
        self.globoptsol = globoptsol
        self.display = display
        self.PSOiters = PSOiters
        self.PSOswarmsize = PSOswarmsize
        self.epsDeltaF = epsDeltaF
        self.isUnknownFeasibilityConstrained = isUnknownFeasibilityConstrained
        self.isUnknownSatisfactionConstrained = isUnknownSatisfactionConstrained
        self.g_unkn_fun = g_unkn_fun
        self.s_unkn_fun = s_unkn_fun

        # check what constraints are specified
        if bineq is not None and Aineq is not None:
            num_cols_A = Aineq.shape[1]
            assert (
                bineq.shape == (num_cols_A,) and Aineq.shape[0] == nvars
            ), "Inconsistent dimensions for Aineq and bineq : Aineq.shape = {} and bineq.shape = {}".format(
                Aineq.shape, bineq.shape
            )
        else:
            raise ValueError("Both Aineq and bineq must be specified")

        isNLConstrained = g0 != 0
        if not isLinConstrained and not isNLConstrained:
            feasible_sampling = False

        f = f0
        g = g0
        dd = (ub - lb) / 2  # compute dd,d0 even if scalevars=0 so to return them
        d0 = (ub + lb) / 2
        if scalevars:
            # Rescale problem variables in [-1,1]
            f = lambda x: f0(x * dd + d0)

            lb = -np.ones(nvar)
            ub = np.ones(nvar)

            if isLinConstrained:
                bineq = bineq - Aineq.dot(d0)
                Aineq = Aineq.dot(np.diag(dd.flatten("C")))

            if isNLConstrained:
                g = lambda x: g0(x * dd + d0)

        # set solver options
        if globoptsol == "pswarm":
            DIRECTopt = []
        else:  # globoptsol == "direct"
            DIRECTopt = nlopt.opt(nlopt.GN_DIRECT, 2)
            DIRECTopt.set_lower_bounds(lb.flatten("C"))
            DIRECTopt.set_upper_bounds(ub.flatten("C"))
            DIRECTopt.set_ftol_abs(1e-5)
            DIRECTopt.set_maxeval(2000)
            DIRECTopt.set_xtol_rel(1e-5)

        if shrink_range == 1:
            # possibly shrink lb,ub to constraints
            if not isNLConstrained and isLinConstrained:
                flin = np.zeros((nvar, 1))

                for i in range(nvar):
                    flin[i] = 1
                    res = linprog(flin, Aineq, bineq, bounds=(lb, ub))
                    aux = max(lb[i], res.fun)
                    lb[i] = aux
                    flin[i] = -1
                    res = linprog(flin, Aineq, bineq, bounds=(lb, ub))
                    aux = min(ub[i], -res.fun)
                    ub[i] = aux
                    flin[i] = 0

            elif isNLConstrained:
                NLpenaltyfun = lambda x: np.sum(np.square(np.maximum(g(x), 0)))
                if isLinConstrained:
                    LINpenaltyfun = lambda x: np.sum(
                        np.square(np.maximum((Aineq.dot(x) - bineq).flatten("C"), 0))
                    )
                else:
                    LINpenaltyfun = lambda x: 0

                for i in range(0, nvar):
                    obj_fun = lambda x: x[i] + 1.0e4 * (
                        NLpenaltyfun(x) + LINpenaltyfun(x)
                    )
                    if globoptsol == "pswarm":
                        if display == 0:
                            with contextlib.redirect_stdout(io.StringIO()):
                                z, cost = pso(
                                    obj_fun,
                                    lb,
                                    ub,
                                    swarmsize=30,
                                    minfunc=1e-8,
                                    maxiter=2000,
                                )
                        else:
                            z, cost = pso(
                                obj_fun,
                                lb,
                                ub,
                                swarmsize=30,
                                minfunc=1e-8,
                                maxiter=2000,
                            )
                    else:  # globoptsol=="direct":
                        DIRECTopt.set_min_objective(lambda x, grad: obj_fun(x)[0])
                        z = DIRECTopt.optimize(z.flatten("C"))
                    lb[i] = max(lb[i], z[i])

                    obj_fun = lambda x: -x[i] + 1.0e4 * (
                        NLpenaltyfun(x) + LINpenaltyfun(x)
                    )

                    if globoptsol == "pswarm":
                        if display == 0:
                            with contextlib.redirect_stdout(io.StringIO()):
                                z, cost = pso(
                                    obj_fun,
                                    lb,
                                    ub,
                                    swarmsize=30,
                                    minfunc=1e-8,
                                    maxiter=2000,
                                )
                        else:
                            z, cost = pso(
                                obj_fun,
                                lb,
                                ub,
                                swarmsize=30,
                                minfunc=1e-8,
                                maxiter=2000,
                            )
                    else:  # globoptsol=="direct":
                        DIRECTopt.set_min_objective(lambda x, grad: obj_fun(x)[0])
                        z = DIRECTopt.optimize(z.flatten("C"))
                    ub[i] = min(ub[i], z[i])

        X = np.zeros((maxevals, nvar))
        F = np.zeros((maxevals, 1))
        z = (lb + ub) / 2

        if not feasible_sampling:
            X[0:nsamp, :] = lhs(nvar, nsamp, "m")
            X[0:nsamp, :] = (
                X[0:nsamp, :] * (np.ones((nsamp, 1)) * (ub - lb))
                + np.ones((nsamp, 1)) * lb
            )
        else:
            nn = nsamp
            nk = 0
            while nk < nsamp:
                XX = lhs(nvar, nn, "m")
                XX = XX * (np.ones((nn, 1)) * (ub - lb)) + np.ones((nn, 1)) * lb

                ii = np.ones(nn, dtype=bool)
                for i in range(nn):
                    if isLinConstrained:
                        ii[i] = np.all(Aineq.dot(XX[i, :]) <= bineq.ravel("C"))
                    if isNLConstrained:
                        ii[i] = ii[i] and np.all(g(XX[i, :]) <= 0)

                nk = np.sum(ii)
                if nk == 0:
                    nn = 20 * nn
                elif nk < nsamp:
                    nn = np.ceil(np.min(20, 1.1 * nsamp / nk) * nn)

            ii = np.nonzero(ii)[0]
            X[0:nsamp, :] = XX[np.nonzero(ii), :]

        if useRBF:
            M = np.zeros((maxevals, maxevals))  # preallocate the entire matrix
            for i in range(nsamp):
                for j in range(i, nsamp):
                    mij = rbf(
                        X[
                            i,
                        ],
                        X[
                            j,
                        ],
                    )
                    M[i, j] = mij
                    M[j, i] = mij
        else:
            M = []
