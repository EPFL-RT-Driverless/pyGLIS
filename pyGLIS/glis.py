#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless
#  Solve (GL)obal optimization problems using (I)nverse distance weighting and radial basis function (S)urrogates.
import os
import warnings
import csv
from time import perf_counter
from typing import Callable, Union

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.stats.qmc import LatinHypercube
from typing import Optional


class GLIS:
    # optimization problem data
    f: Callable
    nvar: int
    lb: np.ndarray  # lower bounds
    ub: np.ndarray  # upper bounds

    # user defined options for GLIS
    nsamp: int  # number of initial samples
    maxevals: int  # max number of iterations
    alpha: float  # exploitation parameter
    delta: float  # exploration parameter
    rbf_function: Callable  # radial basis function
    scaling: bool  # whether to scale the problem variables between -1 and 1 or not
    epsilon_svd: float  # threshold for singular values in SVD
    verbose: bool  # whether to print information or not
    epsilon_DeltaF: float  # threshold for change in objective function value

    # variables used during execution of GLIS
    original_f: Callable  # specified objective function that will be used either to
    # define the actual objective function as is, or after rescaling.
    original_lb: np.ndarray
    original_ub: np.ndarray
    dd: np.ndarray  # (self.ub - self.lb) / 2
    d0: np.ndarray  # (self.ub + self.lb) / 2l;
    X: np.ndarray  # values of the samples
    F: np.ndarray  # values of the objective functions at the samples
    rbf_weights: np.ndarray  # RBF weights used to compute the surrogate function
    M: np.ndarray  # Gram matrix of the RBF function
    csv_dump_file: str  # path to the csv file where the results will be dumped
    time_iter: list[float]  # time spent in each iteration
    time_f_eval: list[float]  # time spent evaluating the objective function
    time_opt_acquisition: list[float]  # time spent optimizing the acquisition function
    time_fit_surrogate: list[float]  # time spent fitting the surrogate function
    fbest: float  # best value of the objective function
    xbest: np.ndarray  # best sample

    def __init__(
        self,
        f: Callable = None,
        nvar: int = 1,
        lb: Union[float, np.ndarray] = 1.0,
        ub: Union[float, np.ndarray] = -1.0,
        nsamp: Optional[int] = None,
        maxevals: int = 20,
        alpha: float = 1.0,
        delta: float = 0.5,
        epsilon_svd: float = 1e-6,
        epsilon_DeltaF: float = 1.0e-4,
        rbf_function: Optional[Callable] = lambda x1, x2: 1
        / (1 + 0.25 * np.sum(np.square(x1 - x2), axis=-1)),
        scaling: bool = True,
        verbose: bool = False,
        csv_dump_file: str = "out.csv",
        load_previous_results: bool = False,
        **kwargs,
    ):
        """
        Generates default problem structure for IDW-RBF Global Optimization.

        Inputs:
        -------
        :param f: objective function
        :type f: Callable
        :param nvar: number of variables
        :type nvar: int
        :param lb: lower bounds on the optimization variables
        :type lb: np.ndarray
        :param ub: upper bounds on the optimization variables
        :type ub: np.ndarray
        :param maxevals: maximum number of function evaluations
        :type maxevals: int
        :param alpha: weight on function uncertainty variance measured by IDW
        :type alpha: float
        :param delta: weight on distance from previous samples
        :type delta: float
        :param epsilon_svd: tolerance used to discard small singular values
        :type epsilon_svd: float
        :param epsilon_DeltaF: weird thingy called somewhere
        :type epsilon_DeltaF: float
        :param nsamp: number of initial samples
        :type nsamp: int
        :param use_rbf: whether to use RBF functions or IDW functions when constructing the surrogate function
        :type use_rbf: bool
        :param rbf_function: inverse quadratic RBF function (only used if use_rbf=1)
        :type rbf_function: Callable
        :param scaling: whether to scale the problem to have variables between -1 and 1.
        :type scaling: bool
        :param Aineq: matrix A defining linear inequality constraints
        :type Aineq: np.ndarray
        :param bineq: right-hand side of constraints A*x <= b
        :type bineq: np.ndarray
        :param g: constraint function. Example: g = lambda x: x[0]**2+x[1]**2-1
        :type g: Callable
        :param constraint_penalty: penalty term on violation of linear inequality and nonlinear constraint
        :type constraint_penalty: float
        :param feasible_sampling: if True, initial samples are forced to be feasible
        :type feasible_sampling: bool
        :param verbose:
        :type verbose: bool
        :param csv_dump_file: file in which all the iteration results are stored
        :type csv_dump_file: str
        :param load_previous_results: whether or not the previous results in csv_dump_file should be used in the
        optimisation process
        :type load_previous_results: bool
        """
        #  check input ================================================================================================
        if type(ub) == float:
            ub = ub * np.ones(nvar)
        if type(lb) == float:
            lb = lb * np.ones(nvar)
        if nsamp is None:
            nsamp = 2 * nvar

        assert (
            epsilon_svd > 0.0
        ), "epsilon_svd must be positive but epsilon_svd = {}".format(epsilon_svd)
        assert alpha > 0.0, "alpha must be positive but alpha = {}".format(alpha)
        assert delta > 0.0, "delta must be positive but delta = {}".format(delta)
        assert nsamp >= 1, "nsamp must be at least 1 but nsamp = {}".format(nsamp)
        assert (
            maxevals >= nsamp
        ), "Max number of function evaluations is too low. You specified {} maxevals and {} nsamp.".format(
            maxevals, nsamp
        )

        if not os.path.isabs(csv_dump_file):
            csv_dump_file = os.path.join(os.getcwd(), csv_dump_file)

        # store parameters ===========================================================================================
        self.nvar = nvar
        self.original_f = f
        self.original_lb = lb
        self.original_ub = ub
        self.lb = lb
        self.ub = ub
        self.maxevals = maxevals
        self.alpha = alpha
        self.delta = delta
        self.nsamp = nsamp
        self.rbf_function = rbf_function
        self.scaling = scaling
        self.epsilon_svd = epsilon_svd
        self.verbose = verbose
        self.epsilon_DeltaF = epsilon_DeltaF
        self.csv_dump_file = csv_dump_file
        self.time_iter = []
        self.time_f_eval = []
        self.time_opt_acquisition = []
        self.time_fit_surrogate = []

        # initialize algorithm =======================================================================================
        # scaling variables in [-1,1]
        if self.scaling:
            self.dd = np.asarray((self.ub - self.lb) / 2)
            self.d0 = np.asarray((self.ub + self.lb) / 2)
            self.lb = -np.ones(self.nvar)
            self.ub = np.ones(self.nvar)
            self.f = lambda x: self.original_f(x * self.dd + self.d0)
        else:
            self.f = self.original_f
            self.g = self.original_g

        self.X = np.zeros((self.maxevals, self.nvar))
        self.F = np.zeros(self.maxevals)

        if load_previous_results:
            print("[GLIS]: Loading previous results from {}".format(self.csv_dump_file))
            data = np.loadtxt(self.csv_dump_file, delimiter=",")
            assert (
                len(data) >= 3
            ), "csv_dump_file must contain at least 3 rows (the first two for the lb and ub, and the following ones for the iterations)"
            assert (
                len(data) <= self.maxevals + 2
            ), "csv_dump_file contains too many rows for the specified maxevals={}".format(
                self.maxevals
            )
            # load lb and ub from the first two lines
            lb = data[0, 1:]
            ub = data[1, 1:]
            assert np.allclose(lb, self.original_lb), "lb does not match"
            assert np.allclose(ub, self.original_ub), "ub does not match"
            self.F[: len(data) - 2] = data[2:, 0]
            self.X[: len(data) - 2, :] = data[2:, 1:]
            self.N = len(data) - 2
            # compute best value so far
            minid = np.argmin(self.F[: self.N])
            self.fbest = self.F[minid]
            self.xbest = self.X[minid, :]
            print("[GLIS]: Loaded {} previous results".format(self.N))
        else:
            print("[GLIS]: Creating {} initial samples".format(self.nsamp))
            # generate the initial samples using regular Latin Hypercube Sampling (generating values in [0,1])
            sampler = LatinHypercube(d=self.nvar)
            self.X[: self.nsamp, :] = sampler.random(self.nsamp)
            # re-scale the sample to our bounds defined by self.lb and self.ub
            self.X[0 : self.nsamp, :] = (
                self.X[0 : self.nsamp, :]
                * (np.ones((self.nsamp, 1)) * (self.ub - self.lb))
                + np.ones((self.nsamp, 1)) * self.lb
            )

            # dump the lower bounds and upper bounds to the csv file
            np.savetxt(
                self.csv_dump_file,
                np.vstack(
                    (np.append(0.0, self.original_lb), np.append(0.0, self.original_ub))
                ),
                delimiter=",",
            )

            # evaluate the function f at the initial samples
            self.fbest = np.inf
            self.xbest = np.zeros(self.nvar)
            self.N = 0
            for i in range(self.nsamp):
                time_fun_eval_start = perf_counter()
                self.F[i] = self.f(self.X[i, :])
                time_fun_eval_i = perf_counter() - time_fun_eval_start
                self.time_iter.append(time_fun_eval_i)
                self.time_f_eval.append(time_fun_eval_i)

                if self.F[i] < self.fbest:
                    self.fbest = self.F[i]
                    self.xbest = self.X[i, :]

                self._print_iteration()
                self._dump_iteration()

                self.N += 1

            print("[GLIS]: finished creating {} initial samples".format(self.nsamp))

        # pre-allocate the Gram matrix for the RBF functions
        self.M = np.zeros((self.maxevals, self.maxevals))
        for i in range(self.N):
            for j in range(i, self.N):
                rbf_kernel_value = self.rbf_function(
                    self.X[i, :],
                    self.X[j, :],
                )
                self.M[i, j] = rbf_kernel_value
                self.M[j, i] = rbf_kernel_value

        # get RBF weights fot the initial samples
        self.rbf_weights = self._get_rbf_weights(self.N)

    def run(self):
        """
        Runs the optimization algorithm and returns the best found solution

        :returns: a dict with the following entries:
            xopt: minimizer x
            fopt: minimum value of the objective function
            X: all the samples generated
            F: all the function values generated
            W: the associated rbf weights
            time_iter: the time taken for each iteration
            time_opt_acquisition: the time taken for the optimization of the acquisition function
            time_fit_surrogate: the time taken for fitting the surrogate model
            time_f_eval: the time taken for evaluating the objective function
        """
        Fmax = np.max(self.F[0 : self.N])
        Fmin = np.min(self.F[0 : self.N])

        # we iterate to construct new values
        while self.N < self.maxevals:
            time_iter_start = perf_counter()

            dF = np.maximum(Fmax - Fmin, self.epsilon_DeltaF)

            # define constraints for the scipy minimize method
            # constraints = []
            # constraints.append({"type": "ineq", "fun": lambda x: x - self.lb})
            # constraints.append({"type": "ineq", "fun": lambda x: self.ub - x})
            # if self.isLinConstrained:
            #     constraints.append(
            #         {"type": "ineq", "fun": lambda x: self.bineq - self.Aineq.dot(x)}
            #     )
            # if self.isNLConstrained:
            #     constraints.append({"type": "ineq", "fun": lambda x: -self.g(x)})

            # # define the acquisition function based on the current surrogate function
            # # and sample points
            # def acquisition(x):
            #     return

            # minimize the acquisition function to get a new sample point x_next
            time_opt_acq_start = perf_counter()
            x_next = self.lb
            res: OptimizeResult = minimize(
                fun=lambda x: self._facquisition(x, self.N, dF, self.rbf_weights),
                x0=x_next,
                method="COBYLA",
                constraints=[
                    {"type": "ineq", "fun": lambda x: x - self.lb},
                    {"type": "ineq", "fun": lambda x: self.ub - x},
                ],
            )
            self.time_opt_acquisition.append(perf_counter() - time_opt_acq_start)
            if res.success:
                x_next = res.x
            else:
                warnings.warn(
                    "Optimization failed with COBYLA: {}\n maxcv={}".format(
                        res.message, res.maxcv
                    )
                )
                # if the optimization of the acquisition function fails, we just
                # sample a random point
                x_next = self.lb + (self.ub - self.lb) * np.random.rand(*self.lb.shape)

            # evaluate the objective function f at the new sample point x_next
            time_fun_eval_start = perf_counter()
            f_x_next = self.f(x_next)
            self.time_f_eval.append(perf_counter() - time_fun_eval_start)

            # update everything
            self.X[self.N, :] = x_next
            self.F[self.N] = f_x_next
            Fmax = np.max((Fmax, f_x_next))
            Fmin = np.min((Fmin, f_x_next))

            time_fit_surrogate_start = perf_counter()
            # Just update last row and column of the Gram matrix M
            for k in range(self.N):
                mij = self.rbf_function(
                    self.X[k, :],
                    self.X[self.N, :],
                )
                self.M[k, self.N] = mij
                self.M[self.N, k] = mij

            self.rbf_weights = self._get_rbf_weights(self.N + 1)
            self.time_fit_surrogate.append(perf_counter() - time_fit_surrogate_start)

            if self.fbest > f_x_next:
                self.fbest = f_x_next.copy()
                self.xbest = x_next.copy()

            self._print_iteration()
            self._dump_iteration()

            self.N += 1

            self.time_iter.append(perf_counter() - time_iter_start)

        # end of the sample acquisition

        # # find best result and return it
        # fopt = self.fbest
        # if self.scaling:
        #     # Scale variables back
        #     xopt = self.xbest * self.dd + self.d0
        #     self.X = (
        #         self.X * (np.ones((self.N, 1)) * self.dd)
        #         + np.ones((self.N, 1)) * self.d0
        #     )
        # else:
        #     xopt = self.xbest

        return {
            "xopt": self.xbest if not self.scaling else self.xbest * self.dd + self.d0,
            "fopt": self.fbest,
            "X": self.X
            if not self.scaling
            else self.X * (np.ones((self.N, 1)) * self.dd)
            + np.ones((self.N, 1)) * self.d0,
            "F": self.F,
            "W": self.rbf_weights,
            "time_iter": np.array(self.time_iter),
            "time_opt_acquisition": np.array(self.time_opt_acquisition),
            "time_fit_surrogate": np.array(self.time_fit_surrogate),
            "time_f_eval": np.array(self.time_f_eval),
        }

    def _print_iteration(self):
        if self.verbose:
            print(
                "N = %4d, fbest = %7.4f, f = %7.4f, x = %s"
                % (
                    self.N,
                    self.fbest,
                    self.F[self.N],
                    np.array2string(
                        self.X[self.N, :]
                        if not self.scaling
                        else (self.X[self.N, :] * self.dd + self.d0),
                        precision=4,
                    ),
                )
            )

    def _dump_iteration(self) -> None:
        """
        Dump the iteration result in the csv file
        """
        with open(self.csv_dump_file, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [self.F[self.N]]
                + self.X[self.N, :].tolist()
                # + (
                #     self.X[self.N, :]
                #     if not self.scaling
                #     else (self.X[self.N, :] * self.dd + self.d0)
                # ).tolist()
            )

    def _get_rbf_weights(self, n: int):
        # Solve M*W = F using SVD
        U, dS, V = np.linalg.svd(self.M[np.ix_(range(n), range(n))])
        non_negligeable_singular_values_idx = np.nonzero(dS >= self.epsilon_svd)[0]
        ns = np.max(non_negligeable_singular_values_idx) + 1
        W = np.transpose(V[0:ns, :]).dot(
            np.diag(1 / dS[0:ns].flatten()).dot(U[:, 0:ns].T.dot(self.F[0:n]))
        )

        return W

    def _facquisition(self, x: np.ndarray, N: int, dF: float, W: np.ndarray):
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

            v = self.rbf_function(self.X[:N, :], x)
            fhat = v.ravel().dot(W.ravel())

            s = (
                np.sqrt(np.sum(w * np.square(self.F[0:N] - fhat)) / sw)
                if not np.allclose(w, 0.0)
                else 0.0
            )
            z = 2.0 / np.pi * np.arctan(1.0 / np.sum(1.0 / d))

        f = fhat - self.delta * s - self.alpha * dF * z

        return f
