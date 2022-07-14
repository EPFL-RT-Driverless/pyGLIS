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
        shrink_range=True,
        constraint_penalty=1000,
        feasible_sampling=False,
        globoptsol="direct",
        display=False,
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
        assert svdtol > 0.0, "svdtol must be positive but svdtol = {}".format(svdtol)
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
            self.isLinConstrained = True
        else:
            raise ValueError("Both Aineq and bineq must be specified")

        self.isNLConstrained = self.g is not None
        if not self.isLinConstrained and not self.isNLConstrained:
            self.feasible_sampling = False

        # scaling variables in [-1,1]
        if self.scalevars:
            self.dd = (self.ub - self.lb) / 2
            self.d0 = (self.ub + self.lb) / 2
            self.lb = -np.ones(self.nvar)
            self.ub = np.ones(self.nvar)
            self.f = lambda x: self.f(x * self.dd + self.d0)

            if self.isLinConstrained:
                self.bineq = self.bineq - self.Aineq.dot(self.d0)
                self.Aineq = self.Aineq.dot(np.diag(self.dd.flatten("C")))

            if self.isNLConstrained:
                self.g = lambda x: self.g(x * self.dd + self.d0)

        # set solver options for the minimization of the acquisition function
        if self.globoptsol == "pswarm":
            self.DIRECTopt = None
        else:  # self.globoptsol == "direct"
            self.DIRECTopt = nlopt.opt(nlopt.GN_DIRECT, 2)
            self.DIRECTopt.set_lower_bounds(self.lb.flatten("C"))
            self.DIRECTopt.set_upper_bounds(self.ub.flatten("C"))
            self.DIRECTopt.set_ftol_abs(1e-5)
            self.DIRECTopt.set_maxeval(2000)
            self.DIRECTopt.set_xtol_rel(1e-5)

        # shrink lb,ub
        if self.shrink_range:
            if not self.isNLConstrained and self.isLinConstrained:
                flin = np.zeros((self.nvar, 1))

                for i in range(self.nvar):
                    flin[i] = 1.0
                    res = linprog(
                        flin, self.Aineq, self.bineq, bounds=(self.lb, self.ub)
                    )
                    self.lb[i] = np.max(self.lb[i], res.fun)
                    flin[i] = -1.0
                    res = linprog(
                        flin, self.Aineq, self.bineq, bounds=(self.lb, self.ub)
                    )
                    self.ub[i] = np.min(self.ub[i], -res.fun)
                    flin[i] = 0.0

            elif self.isNLConstrained:
                NLpenaltyfun = lambda x: np.sum(np.square(np.maximum(self.g(x), 0.0)))
                if self.isLinConstrained:
                    LINpenaltyfun = lambda x: np.sum(
                        np.square(np.maximum(self.Aineq.dot(x) - self.bineq, 0.0))
                    )
                else:
                    LINpenaltyfun = lambda x: 0.0

                for i in range(self.nvar):
                    obj_fun = lambda x: x[i] + 1.0e4 * (
                        NLpenaltyfun(x) + LINpenaltyfun(x)
                    )
                    if self.globoptsol == "pswarm":
                        if self.display:
                            # TODO : use the specified params PSOswarmsize and PSOiters ?
                            z, cost = pso(
                                obj_fun,
                                lb,
                                ub,
                                swarmsize=30,
                                minfunc=1e-8,
                                maxiter=2000,
                            )
                        else:
                            with contextlib.redirect_stdout(io.StringIO()):
                                z, cost = pso(
                                    obj_fun,
                                    lb,
                                    ub,
                                    swarmsize=30,
                                    minfunc=1e-8,
                                    maxiter=2000,
                                )
                    else:  # globoptsol=="direct":
                        self.DIRECTopt.set_min_objective(lambda x, grad: obj_fun(x)[0])
                        z = self.DIRECTopt.optimize(z.flatten("C"))

                    lb[i] = max(lb[i], z[i])

                    obj_fun = lambda x: -x[i] + 1.0e4 * (
                        NLpenaltyfun(x) + LINpenaltyfun(x)
                    )

                    if self.globoptsol == "pswarm":
                        if self.display:
                            z, cost = pso(
                                obj_fun,
                                lb,
                                ub,
                                swarmsize=30,
                                minfunc=1e-8,
                                maxiter=2000,
                            )
                        else:
                            with contextlib.redirect_stdout(io.StringIO()):
                                z, cost = pso(
                                    obj_fun,
                                    lb,
                                    ub,
                                    swarmsize=30,
                                    minfunc=1e-8,
                                    maxiter=2000,
                                )
                    else:  # globoptsol=="direct":
                        self.DIRECTopt.set_min_objective(lambda x, grad: obj_fun(x)[0])
                        z = self.DIRECTopt.optimize(z.flatten())

                    self.ub[i] = np.min(self.ub[i], z[i])

        self.X = np.zeros((self.maxevals, self.nvar))
        self.F = np.zeros(self.maxevals)
        z = (self.lb + self.ub) / 2

        if not feasible_sampling:
            # generate the samples using Latin Hypercube Sampling (generates values in [0,1])
            self.X[0 : self.nsamp, :] = lhs(
                n=self.nvar, samples=self.nsamp, criterion="m"
            )
            # re-scale the sample to our bounds defined by self.lb and self.ub
            self.X[0 : self.nsamp, :] = (
                self.X[0 : self.nsamp, :]
                * (np.ones((self.nsamp, 1)) * (self.ub - self.lb))
                + np.ones((self.nsamp, 1)) * self.lb
            )
        else:
            tpr_nsamp = self.nsamp
            nbr_feasible_samples = 0
            while nbr_feasible_samples < self.nsamp:
                XX = lhs(n=self.nvar, samples=tpr_nsamp, crieterion="m")
                XX = (
                    XX * (np.ones((tpr_nsamp, 1)) * (self.ub - self.lb))
                    + np.ones((tpr_nsamp, 1)) * self.lb
                )

                # find indices of sample points where all the constraints are satisfied.
                sample_idx = np.ones(tpr_nsamp, dtype=bool)
                for i in range(tpr_nsamp):
                    if self.isLinConstrained:
                        sample_idx[i] = np.all(self.Aineq.dot(XX[i, :]) <= self.bineq)
                    if self.isNLConstrained:
                        sample_idx[i] = sample_idx[i] and np.all(
                            self.g(XX[i, :]) <= 0.0
                        )

                # check if we havve enough feasible points. If not increase the number of samples and try again
                nbr_feasible_samples = np.sum(sample_idx)
                if nbr_feasible_samples == 0:
                    tpr_nsamp = 20 * tpr_nsamp
                elif nbr_feasible_samples < self.nsamp:
                    tpr_nsamp = np.ceil(
                        np.min(20, 1.1 * self.nsamp / nbr_feasible_samples) * tpr_nsamp
                    )

            feasible_samples_idx = np.nonzero(sample_idx)[0]
            self.X[0 : self.nsamp, :] = XX[feasible_samples_idx, :]

        # pre-allocate the Gram matrix for the RBF functions
        if useRBF:
            self.M = np.zeros((self.maxevals, self.maxevals))
            for i in range(self.nsamp):
                for j in range(i, self.nsamp):
                    rbf_kernel_value = self.rbf(
                        self.X[i, :],
                        self.X[j, :],
                    )
                    self.M[i, j] = rbf_kernel_value
                    self.M[j, i] = rbf_kernel_value
        else:
            self.M = None

    def get_rbf_weights(self, NX):
        # Solve M*W = F using SVD
        assert self.M.shape == (
            NX,
            NX,
        ), "Gram matrix M has wrong shape: {} instead of {}".format(
            self.M.shape, (NX, NX)
        )
        U, dS, V = np.linalg.vd(self.M)
        non_nnegligeable_singular_values_idx = np.nonzero(dS >= self.svdtol)[0]
        ns = np.max(non_nnegligeable_singular_values_idx) + 1
        W = np.transpose(V[0:ns, :]).dot(
            np.diag(1 / dS[0:ns].flatten()).dot((U[:, 0:ns].T).dot(self.F[0:NX]))
        )

        return W

    def get_delta_adpt(X, constraint_set, delta_const_default):
        ind = constraint_set.shape[0]
        sqr_error_feas = zeros((ind, 1))
        for i in range(0, ind):
            xx = X[i, :]
            Xi = vstack((X[0:i, :], X[i + 1 : ind, :]))
            constraint_set_i = vstack(
                (
                    constraint_set[
                        0:i,
                    ],
                    constraint_set[
                        i + 1 : ind,
                    ],
                )
            )
            Feas_xx = constraint_set[i]
            d = npsum((Xi - xx) ** 2, axis=-1)
            w = npsum(-d) / d
            sw = sum(w)
            ghat = npsum(constraint_set_i.T * w) / sw
            sqr_error_feas[i] = (ghat - Feas_xx) ** 2

        std_feas = (sum(sqr_error_feas) / (ind - 1)) ** (1 / 2)
        delta_adpt = (1 - std_feas) * delta_const_default

        return delta_adpt

    def facquisition(
        xx,
        X,
        F,
        N,
        alpha,
        delta_E,
        dF,
        W,
        rbf,
        useRBF,
        isUnknownFeasibilityConstrained,
        isUnknownSatisfactionConstrained,
        Feasibility_unkn,
        SatConst_unkn,
        delta_G,
        delta_S,
        iw_ibest,
        maxevals,
    ):
        # Acquisition function to minimize to get next sample

        d = npsum(
            (
                X[
                    0:N,
                ]
                - xx
            )
            ** 2,
            axis=-1,
        )

        ii = where(d < 1e-12)
        if ii[0].size > 0:
            fhat = F[ii[0]][0]
            dhat = 0
            if isUnknownFeasibilityConstrained:
                Ghat = Feasibility_unkn[ii]
            else:
                Ghat = 1
            if isUnknownSatisfactionConstrained:
                Shat = SatConst_unkn[ii]
            else:
                Shat = 1
        else:
            w = exp(-d) / d
            sw = sum(w)

            if useRBF:
                v = rbf(X[0:N, :], xx)
                fhat = v.ravel().dot(W.ravel())
            else:
                fhat = (
                    npsum(
                        F[
                            0:N,
                        ]
                        * w
                    )
                    / sw
                )

            if maxevals <= 30:
                # for comparision, used in the original GLIS and when N_max <= 30 in C-GLIS
                dhat = delta_E * atan(1 / sum(1 / d)) * 2 / pi * dF + alpha * sqrt(
                    sum(
                        w
                        * (
                            F[
                                0:N,
                            ]
                            - fhat
                        ).flatten("c")
                        ** 2
                    )
                    / sw
                )
            else:
                dhat = delta_E * (
                    (1 - N / maxevals) * atan((1 / sum(1.0 / d)) / iw_ibest)
                    + N / maxevals * atan(1 / sum(1.0 / d))
                ) * 2 / pi * dF + alpha * sqrt(
                    sum(
                        w
                        * (
                            F[
                                0:N,
                            ]
                            - fhat
                        ).flatten("c")
                        ** 2
                    )
                    / sw
                )

            # to account for the unknown constraints
            if isUnknownFeasibilityConstrained:
                Ghat = npsum(Feasibility_unkn[0:N].T * w) / sw
            else:
                Ghat = 1

            if isUnknownSatisfactionConstrained:
                Shat = npsum(SatConst_unkn[0:N].T * w) / sw
            else:
                Shat = 1

        f = fhat - dhat + (delta_G * (1 - Ghat) + delta_S * (1 - Shat)) * dF

        return f
