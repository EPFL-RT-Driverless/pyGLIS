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
    use_rbf: bool
    rbf_function: Callable
    scaling: bool
    epsilon_svd: float
    feasible_sampling: bool
    verbose: bool
    epsilon_DeltaF: float

    # variables used during execution of the solver
    f0: Callable  # specified objective function that will be used either to define the actual objective function as is,
    # or after rescaling.
    g0: Callable  # specified nonlinear constraints that will be used either to define the actual nonlinear constraints
    # as is, or after rescaling.
    isLinConstrained: bool
    isNLConstrained: bool
    X: np.ndarray
    F: np.ndarray
    rbf_weights: Optional[np.ndarray]
    M: Optional[np.ndarray]
    csv_dump_file: str
    time_iter: list[float]
    time_f_eval: list[float]
    time_opt_acquisition: list[float]
    time_fit_surrogate: list[float]
    fbest: float
    xbest: np.ndarray

    def __init__(
        self,
        f: Callable = None,
        nvar: int = 1,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        Aineq: Optional[np.ndarray] = None,
        bineq: Optional[np.ndarray] = None,
        g: Optional[Callable] = None,
        nsamp: Optional[int] = None,
        maxevals: int = 20,
        alpha: float = 1.0,
        delta: float = 0.5,
        epsilon_svd: float = 1e-6,
        epsilon_DeltaF: float = 1.0e-4,
        use_rbf: bool = True,
        rbf_function: Callable = lambda x1, x2: 1
        / (1 + 0.25 * np.sum(np.square(x1 - x2), axis=-1)),
        scaling: bool = True,
        feasible_sampling=True,
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
        if ub is None:
            ub = np.ones(nvar)
        if lb is None:
            lb = -np.ones(nvar)
        if type(ub) == float:
            ub = np.array([ub])
        if type(lb) == float:
            lb = np.array([lb])
        if nsamp is None:
            nsamp = 2 * nvar

        assert (
            epsilon_svd > 0.0
        ), "epsilon_svd must be positive but epsilon_svd = {}".format(epsilon_svd)
        assert alpha > 0.0, "alpha must be positive but alpha = {}".format(alpha)
        assert delta > 0.0, "delta must be positive but delta = {}".format(delta)
        assert (
            maxevals >= nsamp
        ), "Max number of function evaluations is too low. You specified {} maxevals and {} nsamp.".format(
            maxevals, nsamp
        )

        def test_callable(callback, callback_name):
            try:
                callback(np.ones(nvar))
            except Exception as e:
                raise ValueError(
                    "{} does not handle inputs of the appropriate size nvar={}, error message={}".format(
                        callback_name, nvar, e
                    )
                )

        # test_callable(f, "f")
        # if g is not None:
        #     test_callable(g, "g")

        if bineq is not None and Aineq is not None:
            assert len(Aineq.shape) == 2, "Aineq must be a 2D array"
            assert Aineq.shape[1] == nvar, "Aineq must have {} columns".format(nvar)
            assert len(bineq.shape) == 1, "bineq must be a 1D array"
            assert (
                Aineq.shape[0] == bineq.shape[0]
            ), "Aineq and bineq must have the same number of rows"
        elif not (bineq is None and Aineq is None):
            raise ValueError("Both Aineq and bineq must be specified")

        if not os.path.isabs(csv_dump_file):
            csv_dump_file = os.path.join(os.getcwd(), csv_dump_file)

        # store parameters ===========================================================================================
        self.nvar = nvar
        self.f0 = f
        self.lb = lb
        self.ub = ub
        self.maxevals = maxevals
        self.alpha = alpha
        self.delta = delta
        self.nsamp = nsamp
        self.use_rbf = use_rbf
        self.rbf_function = rbf_function
        self.scaling = scaling
        self.epsilon_svd = epsilon_svd
        self.Aineq = Aineq
        self.bineq = bineq
        self.g0 = g
        self.feasible_sampling = feasible_sampling
        self.verbose = verbose
        self.epsilon_DeltaF = epsilon_DeltaF
        self.csv_dump_file = csv_dump_file
        self.isLinConstrained = self.bineq is not None and self.Aineq is not None
        self.isNLConstrained = self.g0 is not None
        if not self.isLinConstrained and not self.isNLConstrained:
            self.feasible_sampling = False

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
            self.f = lambda x: self.f0(x * self.dd + self.d0)

            if self.isLinConstrained:
                self.bineq = self.bineq - self.Aineq.dot(self.d0)
                self.Aineq = self.Aineq.dot(np.diag(self.dd))

            if self.isNLConstrained:
                self.g = lambda x: self.g0(x * self.dd + self.d0)
        else:
            self.f = self.f0
            self.g = self.g0

        self.X = np.zeros((self.maxevals, self.nvar))
        self.F = np.zeros(self.maxevals)

        if load_previous_results:
            with open(self.csv_dump_file) as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= self.maxevals:
                        raise ValueError(
                            "There are more samples in the provided CSV file than the total number of"
                            "evaluation requested"
                        )

                    self.F[i] = float(row[0])
                    self.X[i, :] = np.array([float(x) for x in row[1:]])
                self.N = i + 1
        else:
            self.N = self.nsamp

            sampler = LatinHypercube(d=self.nvar)
            if not self.feasible_sampling:
                # generate the initial samples using regular Latin Hypercube Sampling (generating values in [0,1])
                self.X[: self.nsamp, :] = sampler.random(self.nsamp)
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
                sample_idx = np.ndarray(shape=(0,))
                generated_samples = np.ndarray(shape=(0, self.nvar))
                while nbr_feasible_samples < self.nsamp:
                    generated_samples = sampler.random(tpr_nsamp)
                    generated_samples = (
                        generated_samples
                        * (np.ones((tpr_nsamp, 1)) * (self.ub - self.lb))
                        + np.ones((tpr_nsamp, 1)) * self.lb
                    )

                    # find indices of sample points where all the constraints are satisfied.
                    # TODO: create a function for this
                    sample_idx = np.ones(tpr_nsamp, dtype=bool)
                    for i in range(tpr_nsamp):
                        if self.isLinConstrained:
                            sample_idx[i] = np.all(
                                self.Aineq.dot(generated_samples[i, :]) <= self.bineq
                            )
                        if self.isNLConstrained:
                            sample_idx[i] = sample_idx[i] and np.all(
                                self.g(generated_samples[i, :]) <= 0.0
                            )

                    # check if we have enough feasible points. If not increase the number of samples and try again
                    nbr_feasible_samples = np.sum(sample_idx)
                    if nbr_feasible_samples == 0:
                        tpr_nsamp = 20 * tpr_nsamp
                    elif nbr_feasible_samples < self.nsamp:
                        tpr_nsamp = np.ceil(
                            np.min(20, 1.1 * self.nsamp / nbr_feasible_samples)
                            * tpr_nsamp
                        )

                feasible_samples_idx = np.nonzero(sample_idx)[0]
                self.X[0 : self.nsamp, :] = generated_samples[feasible_samples_idx, :]

            # evaluate the function f at the initial samples
            for i in range(self.nsamp):
                time_fun_eval_start = perf_counter()
                self.F[i] = self.f(self.X[i, :])

                # dump this iteration result in the csv file
                with open(self.csv_dump_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.F[i]] + list(self.X[i, :]))

                time_fun_eval_i = perf_counter() - time_fun_eval_start
                self.time_iter.append(time_fun_eval_i)
                self.time_f_eval.append(time_fun_eval_i)

            # end sampling

        # pre-allocate the Gram matrix for the RBF functions
        if self.use_rbf:
            self.M = np.zeros((self.maxevals, self.maxevals))
            for i in range(self.N):
                for j in range(i, self.N):
                    rbf_kernel_value = self.rbf_function(
                        self.X[i, :],
                        self.X[j, :],
                    )
                    self.M[i, j] = rbf_kernel_value
                    self.M[j, i] = rbf_kernel_value
        else:
            self.M = None

        # get RBF weights fot the initial samples
        if self.use_rbf:
            self.rbf_weights = self._get_rbf_weights(self.N)
        else:
            self.rbf_weights = None

        # find the optimal point among the FEASIBLE initial samples
        self.fbest = np.inf
        self.xbest = np.zeros(self.nvar)
        for i in range(self.N):
            isfeas = True
            if self.isLinConstrained:
                isfeas = isfeas and np.all(self.Aineq.dot(self.X[i, :]) <= self.bineq)
            if self.isNLConstrained:
                isfeas = isfeas and np.all(self.g(self.X[i, :]) <= 0)
            if isfeas and self.fbest > self.F[i]:
                self.xbest = self.X[i, :]
                self.fbest = self.F[i]

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

            if self.use_rbf:
                v = self.rbf_function(self.X[:N, :], x)
                fhat = v.ravel().dot(W.ravel())
            else:
                fhat = np.sum(self.F[:N] * w) / sw if not np.allclose(w, 0.0) else 0.0

            s = (
                np.sqrt(np.sum(w * np.square(self.F[0:N] - fhat)) / sw)
                if not np.allclose(w, 0.0)
                else 0.0
            )
            z = 2.0 / np.pi * np.arctan(1.0 / np.sum(1.0 / d))

        f = fhat - self.delta * s - self.alpha * dF * z

        return f

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
        Fmax = np.max(self.F[0 : self.nsamp])
        Fmin = np.min(self.F[0 : self.nsamp])

        # we iterate to construct new values
        while self.N < self.maxevals:
            time_iter_start = perf_counter()

            dF = np.maximum(
                (np.max(self.F[0 : self.nsamp]) - np.min(self.F[0 : self.nsamp])),
                self.epsilon_DeltaF,
            )

            # define constraints for the scipy minimize method
            constraints = []
            if self.isLinConstrained:
                constraints.append(
                    {"type": "ineq", "fun": lambda x: self.bineq - self.Aineq.dot(x)}
                )
            if self.isNLConstrained:
                constraints.append({"type": "ineq", "fun": lambda x: -self.g(x)})

            # define the acquisition function based on the current surrogate function
            # and sample points
            def acquisition(x):
                return self._facquisition(x, self.N, dF, self.rbf_weights)

            # minimize the acquisition function to get a new sample point x_next
            time_opt_acq_start = perf_counter()
            x_next = self.lb
            # noinspection PyTypeChecker
            res: OptimizeResult = minimize(
                fun=acquisition, x0=x_next, method="COBYLA", constraints=constraints
            )
            if not res.success:
                warnings.warn("Optimization failed with COBYLA")
            x_next = res.x

            self.time_opt_acquisition.append(perf_counter() - time_opt_acq_start)

            # evaluate the objective function f at the new sample point z
            time_fun_eval_start = perf_counter()
            f_x_next = self.f(x_next)
            self.time_f_eval.append(perf_counter() - time_fun_eval_start)

            # update everything
            self.N += 1
            self.X[self.N - 1, :] = x_next
            self.F[self.N - 1] = f_x_next

            Fmax = np.max((Fmax, f_x_next))
            Fmin = np.min((Fmin, f_x_next))

            time_fit_surrogate_start = perf_counter()
            if self.use_rbf:
                # Just update last row and column of M
                for k in range(self.N):
                    mij = self.rbf_function(
                        self.X[k, :],
                        self.X[self.N - 1, :],
                    )
                    self.M[k, self.N - 1] = mij
                    self.M[self.N - 1, k] = mij

                self.rbf_weights = self._get_rbf_weights(self.N)

            self.time_fit_surrogate.append(perf_counter() - time_fit_surrogate_start)

            if self.fbest > f_x_next:
                self.fbest = f_x_next.copy()
                self.xbest = x_next.copy()

            if self.verbose:
                print(
                    "self.N = %4d, cost = %7.4f, best = %7.4f"
                    % (self.N, f_x_next, self.fbest)
                )
                string = ""
                for j in range(self.nvar):
                    aux = self.xbest[j]
                    if self.scaling:
                        aux = aux * self.dd[j] + self.d0[j]

                    string = string + " x" + str(j + 1) + " = " + ("%7.4f" % aux)
                print(string)

            # dump this iteration result in the csv file
            with open(self.csv_dump_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([f_x_next] + x_next.tolist())

            self.time_iter.append(perf_counter() - time_iter_start)

        # end of the sample acquisition

        # find best result and return it
        fopt = self.fbest
        if self.scaling:
            # Scale variables back
            xopt = self.xbest * self.dd + self.d0
            self.X = (
                self.X * (np.ones((self.N, 1)) * self.dd)
                + np.ones((self.N, 1)) * self.d0
            )
        else:
            xopt = self.xbest

        return {
            "xopt": xopt,
            "fopt": fopt,
            "X": self.X,
            "F": self.F,
            "W": self.rbf_weights,
            "time_iter": np.array(self.time_iter),
            "time_opt_acquisition": np.array(self.time_opt_acquisition),
            "time_fit_surrogate": np.array(self.time_fit_surrogate),
            "time_f_eval": np.array(self.time_f_eval),
        }
