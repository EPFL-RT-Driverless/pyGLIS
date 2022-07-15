# Test global optimization based on IDW and RBF on a benchmark problem
#
# (C) 2019 A. Bemporad, June 14, 2019

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time

import GPyOpt as BO
import matplotlib.pyplot as plt
import numpy as np
from pyGLIS import GLIS
from pyswarm import pso

Ntests = 1
run_bayesopt = False

if __name__ == "__main__":
    # rng default for reproducibility
    np.random.seed(0)

    plt.close("all")
    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(14, 7))

    # benchmark_problem="ackley"
    # benchmark_problem="camelsixhumps"
    # benchmark_problem = "hartman6"
    benchmark_problem = "rosenbrock8"

    if benchmark_problem == "camelsixhumps":
        # Camel six-humps function
        nvars = 2
        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        fun = lambda x: (
            (4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2
            + x[0] * x[1]
            + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2
        )
        xopt0 = np.array(
            [[0.0898, -0.0898], [-0.7126, 0.7126]]
        )  # unconstrained optimizers, one per column
        fopt0 = -1.0316  # unconstrained optimum
        maxevals = 25
        use_linear_constraints = False
        use_nl_constraints = False
        if use_linear_constraints or use_nl_constraints:
            run_bayesopt = False  # constraints not supported

    elif benchmark_problem == "hartman6":
        from math import exp

        nvars = 6
        lb = np.zeros(nvars)
        ub = np.ones(nvars)
        alphaH = np.array([1.0, 1.2, 3.0, 3.2])
        AH = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        PH = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        def fun(x):
            # xx = x.flatten("c")
            f = 0
            for j in range(0, 4):
                aux = 0
                for i in range(0, 6):
                    aux = aux + (x[i] - PH[j, i]) ** 2 * AH[j, i]
                f = f - exp(-aux) * alphaH[j]
            return f

        fopt0 = -3.32237  # optimum
        xopt0 = np.array(
            [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
        )  # optimizer
        maxevals = 80
        use_linear_constraints = False
        use_nl_constraints = False

    elif benchmark_problem == "rosenbrock8":
        nvars = 8
        lb = -5 * np.ones(nvars)
        ub = -lb

        def fun(x):
            f = 0
            for j in range(0, nvars - 1):
                f = f + 100.0 * (x[j + 1] - x[j] ** 2) ** 2 + (1.0 - x[j]) ** 2
            return f

        maxevals = 500

        xopt0 = np.ones(nvars)
        fopt0 = 0.0
        use_linear_constraints = False
        use_nl_constraints = False

    elif benchmark_problem == "ackley":
        nvars = 2
        lb = -5.0 * np.ones(nvars)
        ub = -lb

        fun = (
            lambda x: -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
            - np.exp(0.5 * (np.cos(2.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1])))
            + np.exp(1.0)
            + 20.0
        )
        maxevals = 60

        xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200, minfunc=1e-12, maxiter=10000)
        use_linear_constraints = False
        use_nl_constraints = False

    glis_instance = GLIS(
        nvar=nvars,
        f=fun,
        ub=ub,
        lb=lb,
        Aineq=np.array(
            [
                [1.6295, 1],
                [-1, 4.4553],
                [-4.3023, -1],
                [-5.6905, -12.1374],
                [17.6198, 1],
            ]
        )
        if use_linear_constraints and benchmark_problem == "camelsixhumps"
        else None,
        bineq=np.array([[3.0786, 2.7417, -1.4909, 1, 32.5198]])
        if use_linear_constraints and benchmark_problem == "camelsixhumps"
        else None,
        g=(lambda x: np.array([x[0] ** 2 + (x[1] + 0.1) ** 2 - 0.5]))
        if use_nl_constraints and benchmark_problem == "camelsixhumps"
        else None,
        maxevals=maxevals,
        useRBF=True,
        globoptsol=GLIS.SubproblemSolver.pso,
        verbose=True,
        scalevars=True,
        constraint_penalty=1.0e3,
        feasible_sampling=False,
        alpha=1.0,
        delta=0.5,
    )

    print("Running GLIS optimization:\n")

    for i in range(0, Ntests):
        tic = time.perf_counter()
        out = glis_instance.run()
        toc = time.perf_counter()
        print("Test # %2d, elapsed time: %5.4f" % (i + 1, toc - tic))

        xopt1 = out["xopt"]
        fopt1 = out["fopt"]
        F = out["F"]
        X = out["X"]

        minf = np.zeros((maxevals, 1))
        for j in range(maxevals):
            minf[j] = min(F[0 : j + 1])

        plt.plot(np.arange(0, maxevals), minf, color=[0.8500, 0.3250, 0.0980])

    if run_bayesopt:

        print("\nRunning Bayesian optimization:\n")

        def fobj(
            x,
        ):  #  the objective function must have 2d numpy arrays as input and output
            xx = x[0]
            ff = np.zeros((1, 1))
            ff[0, 0] = fun(xx)
            return ff

        space = []
        for i in range(0, nvars):
            space.append(
                {
                    "name": "x" + str(i + 1),
                    "type": "continuous",
                    "domain": (lb[i], ub[i]),
                }
            )

        for i in range(0, Ntests):
            tic = time.perf_counter()
            Bopt = BO.methods.BayesianOptimization(fobj, space)
            Bopt.run_optimization(max_iter=maxevals - 5)
            toc = time.perf_counter()
            print("Test # %2d, elapsed time: %5.4f" % (i + 1, toc - tic))
            # Bopt.plot_acquisition()

            xopt2 = Bopt.x_opt
            fopt2 = Bopt.fx_opt
            XBO = Bopt.X
            FBO = Bopt.Y
            maxevalsBO = FBO.size

            minfBO = np.zeros((maxevalsBO, 1))
            for j in range(maxevalsBO):
                minfBO[j] = min(FBO[0 : j + 1])

            plt.plot(
                np.arange(0, maxevalsBO), minfBO, color=[237 / 256, 177 / 256, 32 / 256]
            )

    plt.grid()
    if Ntests == 1:
        thelegend = ["GLIS"]
        if run_bayesopt:
            thelegend.append("BO")
        if not (use_linear_constraints or use_nl_constraints):
            plt.plot(np.arange(0, maxevals), fopt0 * np.ones(maxevals))
            thelegend.append("optimum")
        plt.legend(thelegend)
    else:
        axes = plt.gca()
        ylim = axes.get_ylim()
        ymax = ylim[1]
        ymin = ylim[0]
        plt.text(5, ymax - (ymax - ymin) / 10, "GLIS", color=[0.8500, 0.3250, 0.0980])
        if run_bayesopt:
            plt.text(
                5,
                ymax - 2 * (ymax - ymin) / 10,
                "BO",
                color=[237 / 256, 177 / 256, 32 / 256],
            )
    plt.show()

    if Ntests == 1 and nvars == 2:

        fig, ax = plt.subplots(figsize=(14, 7))

        [x, y] = np.meshgrid(
            np.arange(lb[0], ub[0], 0.01), np.arange(lb[1], ub[1], 0.01)
        )
        z = np.zeros(x.shape)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                z[i, j] = fun(np.array([x[i, j], y[i, j]]))

        plt.contour(x, y, z, 100, alpha=0.4)
        plt.plot(
            X[:, 0], X[:, 1], "*", color=[237 / 256, 177 / 256, 32 / 256], markersize=11
        )
        plt.plot(
            xopt0[
                0,
            ],
            xopt0[
                1,
            ],
            "o",
            color=[0, 0.4470, 0.7410],
            markersize=15,
        )
        plt.plot(xopt1[0], xopt1[1], "*", color=[0.8500, 0.3250, 0.0980], markersize=15)

        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection

        patches = []
        if use_linear_constraints:
            V = np.array(
                [
                    [0.4104, -0.2748],
                    [0.1934, 0.6588],
                    [1.3286, 0.9136],
                    [1.8412, 0.0783],
                    [1.9009, -0.9736],
                ]
            )

            polygon = mpatches.Polygon(V, True)
            patches.append(polygon)

        if use_nl_constraints:
            from math import sin

            th = np.arange(0, 2 * np.pi, 0.01)
            N = th.size
            V = np.zeros((N, 2))
            for i in range(0, N):
                V[i, 0] = 0 + np.sqrt(0.5) * np.cos(th[i])
                V[i, 1] = -0.1 + np.sqrt(0.5) * np.sin(th[i])
            circle = mpatches.Polygon(V, True)
            patches.append(circle)

        if use_linear_constraints or use_nl_constraints:
            collection = PatchCollection(
                patches, edgecolor=[0, 0, 0], facecolor=[0.5, 0.5, 0.5], alpha=0.6
            )
            ax.add_collection(collection)

        plt.grid()
        plt.show()
