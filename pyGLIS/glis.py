#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless
#  Solve (GL)obal optimization problems using (I)nverse distance weighting and radial basis function (S)urrogates.

import contextlib
from enum import Enum
import io
from time import perf_counter
from typing import Callable

import nlopt  # https://nlopt.readthedocs.io
import numpy as np
from pyDOE import lhs  # https://pythonhosted.org/pyDOE/
from pyswarm import pso  # https://pythonhosted.org/pyswarm/
from scipy.optimize import linprog


class GLIS:
    class SubproblemSolver(Enum):
        pso = 0
        direct = 1

    # function to minimize and its number of variables
    f: Callable
    nvar: int

    # constraints: bounding box, linear and nonlinear constraints
    lb: np.ndarray
    ub: np.ndarray
    Aineq: np.ndarray
    bineq: np.ndarray
    g: Callable

    # user defined options for GLIS
    nsamp: int
    maxevals: int
    alpha: float
    delta: float
    useRBF: bool
    rbf: Callable
    scalevars: bool
    svdtol: float
    shrink_range: bool
    constraint_penalty: float
    feasible_sampling: bool
    globoptsol: SubproblemSolver
    verbose: bool
    PSOiters: int
    PSOswarmsize: int
    epsilon_DeltaF: float

    # variables used during execution of the solver
    f0: Callable  # specified objective function that will be used either to define the actual objective function as is, or after rescaling.
    g0: Callable  # specified nonlinear constraints that will be used either to define the actual nonlinear constraints as is, or after rescaling.
    isLinConstrained: bool
    isNLConstrained: bool
    X: np.ndarray
    F: np.ndarray

    def __init__(
        self,
        f=None,
        nvar=1,
        lb=None,
        ub=None,
        Aineq=None,
        bineq=None,
        g=None,
        nsamp=None,
        maxevals=20,
        alpha=1,
        delta=0.5,
        useRBF=True,
        rbf=lambda x1, x2: 1 / (1 + 0.25 * np.sum(np.square(x1 - x2), axis=-1)),
        scalevars=True,
        svdtol=1e-6,
        shrink_range=False,
        constraint_penalty=1000.0,
        feasible_sampling=True,
        globoptsol=SubproblemSolver.pso,
        verbose=False,
        PSOiters=1000,
        PSOswarmsize=30,
        epsilon_DeltaF=1.0e-4,
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
         :param useRBF: bool, wether to use RBF functions or IDW functions when constructing the surrogate function
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
         :param epsilon_DeltaF: bruh
         :type epsilon_DeltaF: float
        """
        # default values for unspecified parameters
        if ub is None:
            ub = -np.ones((nvar, 1))
        if lb is None:
            lb = np.ones((nvar, 1))
        if nsamp is None:
            nsamp = 2 * nvar

        # Test the coherence of the provided configuration
        assert svdtol > 0.0, "svdtol must be positive but svdtol = {}".format(svdtol)
        assert alpha > 0.0, "alpha must be positive but alpha = {}".format(alpha)
        assert delta > 0.0, "delta must be positive but delta = {}".format(delta)
        assert (
            maxevals >= nsamp
        ), "Max number of function evaluations is too low. You specified {} maxevals and {} nsamp.".format(
            maxevals, nsamp
        )

        def test_callable(callable, callable_name):
            try:
                callable(np.zeros(nvar))
            except Exception as e:
                raise ValueError(
                    "{} does not handle inputs of the appropriate size nvar={}".format(
                        callable_name, nvar
                    )
                )

        test_callable(f, "f")
        if g is not None:
            test_callable(g, "g")

        # check what constraints are specified
        if bineq is not None and Aineq is not None:
            num_cols_A = Aineq.shape[1]
            assert (
                bineq.shape == (num_cols_A,) and Aineq.shape[0] == nvar
            ), "Inconsistent dimensions for Aineq and bineq : Aineq.shape = {} and bineq.shape = {}".format(
                Aineq.shape, bineq.shape
            )
        elif not (bineq is None and Aineq is None):
            raise ValueError("Both Aineq and bineq must be specified")

        # store parameters
        self.nvar = nvar
        self.f0 = f
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
        self.g0 = g
        self.shrink_range = shrink_range
        self.constraint_penalty = constraint_penalty
        self.feasible_sampling = feasible_sampling
        self.globoptsol = globoptsol
        self.verbose = verbose
        self.PSOiters = PSOiters
        self.PSOswarmsize = PSOswarmsize
        self.epsilon_DeltaF = epsilon_DeltaF

        self.initialize_sample_points()

    def initialize_sample_points(self):
        # check what constraints were specified
        self.isLinConstrained = self.bineq is not None and self.Aineq is not None
        self.isNLConstrained = self.g0 is not None
        if not self.isLinConstrained and not self.isNLConstrained:
            self.feasible_sampling = False

        # scaling variables in [-1,1]
        if self.scalevars:
            self.dd = (self.ub - self.lb) / 2
            self.d0 = (self.ub + self.lb) / 2
            self.lb = -np.ones(self.nvar)
            self.ub = np.ones(self.nvar)
            self.f = lambda x: self.f0(x * self.dd + self.d0)

            if self.isLinConstrained:
                self.bineq = self.bineq - self.Aineq.dot(self.d0)
                self.Aineq = self.Aineq.dot(np.diag(self.dd.flatten("C")))

            if self.isNLConstrained:
                self.g = lambda x: self.g0(x * self.dd + self.d0)
        else:
            self.f = self.f0
            self.g = self.g0

        # set solver options for the minimization of the acquisition function
        if self.globoptsol == GLIS.SubproblemSolver.pso:
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
                    # TODO: use another NLP solver for shrinking the bounding box ?
                    obj_fun = lambda x: x[i] + 1.0e4 * (
                        NLpenaltyfun(x) + LINpenaltyfun(x)
                    )
                    z = self.minimize(obj_fun)
                    self.lb[i] = np.max(self.lb[i], z[i])

                    obj_fun = lambda x: -x[i] + 1.0e4 * (
                        NLpenaltyfun(x) + LINpenaltyfun(x)
                    )
                    z = self.minimize(obj_fun)
                    self.ub[i] = np.min(self.ub[i], z[i])

        self.X = np.zeros((self.maxevals, self.nvar))
        self.F = np.zeros(self.maxevals)
        z = (self.lb + self.ub) / 2

        if not self.feasible_sampling:
            # generate the initial samples using regular Latin Hypercube Sampling (generating values in [0,1])
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
            # generate the initial samples using a modified Latin Hypercube Sampling (generating values in [0,1])
            tpr_nsamp = self.nsamp
            nbr_feasible_samples = 0
            while nbr_feasible_samples < self.nsamp:
                XX = lhs(n=self.nvar, samples=tpr_nsamp, crieterion="m")
                XX = (
                    XX * (np.ones((tpr_nsamp, 1)) * (self.ub - self.lb))
                    + np.ones((tpr_nsamp, 1)) * self.lb
                )

                # find indices of sample points where all the constraints are satisfied.
                # TODO: create a function for this
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
        if self.useRBF:
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

    def minimize(self, obj_fun):
        if self.globoptsol == GLIS.SubproblemSolver.pso:
            if self.verbose:
                z, _ = pso(
                    obj_fun,
                    self.lb,
                    self.ub,
                    swarmsize=self.PSOswarmsize,
                    minfunc=1e-8,
                    maxiter=self.PSOiters,
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    z, _ = pso(
                        obj_fun,
                        self.lb,
                        self.ub,
                        swarmsize=self.PSOswarmsize,
                        minfunc=1e-8,
                        maxiter=self.PSOiters,
                    )
        else:  # globoptsol=="direct":
            self.DIRECTopt.set_min_objective(lambda x, grad: obj_fun(x)[0])
            z = self.DIRECTopt.optimize(z.flatten("C"))
        return z

    def get_rbf_weights(self, NX):
        # Solve M*W = F using SVD
        U, dS, V = np.linalg.svd(self.M[np.ix_(range(NX), range(NX))])
        non_nnegligeable_singular_values_idx = np.nonzero(dS >= self.svdtol)[0]
        ns = np.max(non_nnegligeable_singular_values_idx) + 1
        W = np.transpose(V[0:ns, :]).dot(
            np.diag(1 / dS[0:ns].flatten()).dot((U[:, 0:ns].T).dot(self.F[0:NX]))
        )

        return W

    def facquisition(self, x, N, dF, W):
        # Acquisition function to minimize to get next sample

        d = np.sum(np.square(self.X[0:N, :] - x.reshape(1, -1)), axis=1)

        ii = np.nonzero(d < 1e-12)[0]
        if ii.size > 0:
            fhat = self.F[ii]
            s = 0.0
            z = 0.0
        else:
            w = np.exp(-d) / d
            sw = np.sum(w)

            if self.useRBF:
                v = self.rbf(self.X[:N, :], x)
                fhat = v.ravel().dot(W.ravel())
            else:
                fhat = np.sum(self.F[:N] * w) / sw

            s = np.sqrt(np.sum(w * np.square(self.F[0:N] - fhat)) / sw)
            z = 2.0 / np.pi * np.arctan(1.0 / np.sum(1.0 / d))

        f = fhat - self.delta * s - self.alpha * dF * z

        return f

    def run(self):
        time_iter = []
        time_f_eval = []
        time_opt_acquisition = []
        time_fit_surrogate = []

        # evaluate the function f at the initial samples
        for i in range(self.nsamp):
            time_fun_eval_start = perf_counter()
            self.F[i] = self.f(self.X[i, :])
            time_fun_eval_i = perf_counter() - time_fun_eval_start
            time_iter.append(time_fun_eval_i)
            time_f_eval.append(time_fun_eval_i)
            time_opt_acquisition.append(0.0)
            time_fit_surrogate.append(0.0)

        # get RBF weights fot the initial samples
        # TODO : maybe include these in __init__? Depends on the implementation of the serialization
        if self.useRBF:
            rbf_weights = self.get_rbf_weights(self.nsamp)
        else:
            rbf_weights = None

        # find the optimal point among the FEASIBLE initial samples
        fbest = np.inf
        zbest = np.zeros(self.nsamp)
        for i in range(self.nsamp):
            isfeas = True
            if self.isLinConstrained:
                isfeas = isfeas and np.all(self.Aineq.dot(self.X[i, :]) <= self.bineq)
            if self.isNLConstrained:
                isfeas = isfeas and np.all(self.g(X[i, :]) <= 0)
            if isfeas and fbest > self.F[i]:
                fbest = self.F[i]
                zbest = self.X[i, :]

        Fmax = np.max(self.F[0 : self.nsamp])
        Fmin = np.min(self.F[0 : self.nsamp])

        N = self.nsamp

        # we iterate to construct new values
        while N < self.maxevals:

            time_iter_start = perf_counter()

            dF = np.maximum(
                (np.max(self.F[0 : self.nsamp]) - np.min(self.F[0 : self.nsamp])),
                self.epsilon_DeltaF,
            )

            # compute penalty function
            if self.isLinConstrained or self.isNLConstrained:
                linear_constraint_penalty_function = lambda x: np.sum(
                    np.square(np.maximum(self.Aineq.dot(x) - self.bineq, 0.0))
                )
                nonlinear_constraint_penalty_function = lambda x: np.sum(
                    np.square(np.maximum(self.g(x), 0.0))
                )
                constraint_penalty_function = (
                    lambda x: self.constraint_penalty
                    * dF
                    * (
                        (
                            linear_constraint_penalty_function
                            if self.isLinConstrained
                            else 0.0
                        )
                        + (
                            nonlinear_constraint_penalty_function
                            if self.isNLConstrained
                            else 0.0
                        )
                    )
                )
            else:
                constraint_penalty_function = lambda x: 0.0

            # define the acquisition function based on the current surrogate function
            # and sample points
            acquisition = lambda x: (
                self.facquisition(x, N, dF, rbf_weights)
                + constraint_penalty_function(x)
            )

            # minimize the acquisition function to get a new sample point z
            time_opt_acq_start = perf_counter()
            if self.globoptsol == GLIS.SubproblemSolver.pso:
                with contextlib.redirect_stdout(io.StringIO()):
                    x_next, _ = pso(
                        acquisition,
                        self.lb,
                        self.ub,
                        swarmsize=self.PSOswarmsize,
                        minfunc=dF * 1e-8,
                        maxiter=self.PSOiters,
                    )

            else:
                self.DIRECTopt.set_min_objective(lambda x, grad: acquisition(x)[0])
                x_next = self.DIRECTopt.optimize(x_next.flatten())

            time_opt_acquisition.append(perf_counter() - time_opt_acq_start)

            # evaluate the objective function f at the new sample point z
            time_fun_eval_start = perf_counter()
            f_x_next = self.f(x_next)
            time_f_eval.append(perf_counter() - time_fun_eval_start)

            # update everything
            N = N + 1
            self.X[N - 1, :] = x_next
            self.F[N - 1] = f_x_next

            Fmax = np.max((Fmax, f_x_next))
            Fmin = np.min((Fmin, f_x_next))

            time_fit_surrogate_start = perf_counter()
            if self.useRBF:
                # Just update last row and column of M
                for k in range(N):
                    mij = self.rbf(
                        self.X[k, :],
                        self.X[N - 1, :],
                    )
                    self.M[k, N - 1] = mij
                    self.M[N - 1, k] = mij

                rbf_weights = self.get_rbf_weights(N)

            time_fit_surrogate.append(perf_counter() - time_fit_surrogate_start)

            if fbest > f_x_next:
                fbest = f_x_next.copy()
                zbest = x_next.copy()

            if self.verbose:
                print("N = %4d, cost = %7.4f, best = %7.4f" % (N, f_x_next, fbest))
                string = ""
                for j in range(self.nvar):
                    aux = zbest[j]
                    if self.scalevars:
                        aux = aux * self.dd[j] + self.d0[j]

                    string = string + " x" + str(j + 1) + " = " + ("%7.4f" % aux)
                print(string)

            time_iter.append(perf_counter() - time_iter_start)

        # end of the sample acquisition

        # find best result and return it
        fopt = fbest.copy()
        xopt = zbest.copy()
        if self.scalevars:
            # Scale variables back
            xopt = xopt * self.dd + self.d0
            self.X = self.X * (np.ones((N, 1)) * self.dd) + np.ones((N, 1)) * self.d0

        out = {
            "xopt": xopt,
            "fopt": fopt,
            "X": self.X,
            "F": self.F,
            "W": rbf_weights,
            "time_iter": np.array(time_iter),
            "time_opt_acquisition": np.array(time_opt_acquisition),
            "time_fit_surrogate": np.array(time_fit_surrogate),
            "time_f_eval": np.array(time_f_eval),
        }

        return out
