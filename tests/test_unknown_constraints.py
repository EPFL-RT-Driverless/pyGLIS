# Test optimization by preference learning with unknown constraint handling on benchmark problems
# Algorithms:
#   - C-GLISp
#   - C-GLIS
#
# Reference code: 'test_pref_benchmarks' by A. Bemporad, September 21, 2019

# M. Zhu, June 08, 2021

import glis.glisp
import glis.glis
from glis.glisp_function1 import glisp_function1
from glis.glisp_function2 import glisp_function2
from glis.glisp_function3 import glisp_function3

from numpy import array, zeros, ones, logspace, maximum
from numpy import sum as vecsum
from numpy.random import seed
import time  # for tic-toc
from math import pi, sqrt, cos, sin, exp, ceil, atan2, atan

# plotting libraries
from numpy import arange, meshgrid
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

Ntests =1 #number of tests executed on the same problem

runGLIS = 0   #0 = run GLISp, 1 = run GLIS
runGLISp = 1-runGLIS

RBFcalibrate = 1  # recalibrate parameters during optimization
acquisition_method = 1  # acquisition method for RBF-based preference learning


if __name__ == '__main__':

    TIME0 = time.time()

    seed(0)  # rng default for reproducibility
    plt.close('all')
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(14, 7))

    # benchmarks without unknown constraints
    # benchmark='1d'
    # benchmark='camelsixhumps'
    # benchmark='camelsixhumps-constr' #camelsixhumps with known constraints

    # 2D benchmarks used for illustration of pref with unknown constraints
    # benchmark = 'MBC'  # Mishra's Bird function constrained
    benchmark='CHC' #CamelSixHumps function with feasibility constraints
    # benchmark='CHSC' #CamelSixHumps function with feasibility and satisfactory constraints


    if benchmark == "MBC":
        # Mishra's Bird function constrained
        nvars = 2
        lb = array([-10.0, -6.5])
        ub = array([-2, 0.0])

        fun = lambda x: sin(x[1])*exp((1-cos(x[0]))**2) + cos(x[0])*exp((1-sin(x[1]))**2) + (x[0] - x[1])**2
        xopt0 = array([-3.1302468, -1.5821422]) # unconstrained optimizer
        fopt0 = -106.7645367 # unconstrained optimum

        xopt_const = array([-9.3669,-1.62779]) # constrained optimizer
        fopt_const = -48.4060 # constrained optimum

        # For unknown constraints
        isUnknownFeasibilityConstrained = 1  # isUnknownFeasibilityConstrained = 1, if unknown inequality constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0 # isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist, isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained == 1:
            g_unkn_fun = lambda x: (x[0] + 9) ** 2 + (x[1] + 3) ** 2 - 9
        else:
            g_unkn_fun = lambda x: 0

        if isUnknownSatisfactionConstrained: # add the necessary eqns if relavent
            s_unkn_fun = lambda x: 0
        else:
            s_unkn_fun = lambda x: 0

        comparetol = 1e-4
        delta = 1
        maxevals = 50
        nsamp = ceil(maxevals / 4)

        # For known constraints
        use_linear_constraints = 0
        use_nl_constraints =0

        if use_linear_constraints:
            Aineq = array([])  # placeholder, if relavent, add the required constraints
            bineq = array([])
        else:
            Aineq = array([])
            bineq = array([])

        if use_nl_constraints:
            g = array([]) # placeholder, if relavent, add the required constraints
        else:
            g = array([])

    elif benchmark == "CHC":
        # CamelSixHumps function with feasibility constraints
        nvars = 2
        lb = array([-2.0, -1.0])
        ub = array([2.0, 1.0])
        fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                         x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
        xopt0 = array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
        fopt0 = -1.0316  # unconstrained optimum
        xopt_const = array([0.21305, 0.57424])  # constrained optimizers
        fopt_const = -0.58445 #constrained optimum

        # For unknown constraints
        isUnknownFeasibilityConstrained = 1  # isUnknownFeasibilityConstrained = 1, if unknown inequality constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0  # isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist, isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained == 1:
            Aineq_unkn = array([[1.6295, 1],
                                [-1, 4.4553],
                                [-4.3023, -1],
                                [-5.6905, -12.1374],
                                [17.6198, 1]])

            bineq_unkn = array([[3.0786, 2.7417, -1.4909, 1, 32.5198]])
            g_nl_unkn = lambda x: array([x[0] ** 2 + (x[1] + 0.1) ** 2 - .5])

            g_unkn_fun = lambda x: array([sum(maximum((Aineq_unkn.dot(x) - bineq_unkn).flatten("c"), 0.0)) + sum(maximum(g_nl_unkn(x), 0))])
        else:
            g_unkn_fun = lambda x: 0

        if isUnknownSatisfactionConstrained: # add the necessary eqns if relavent
            s_unkn_fun = lambda x: 0
        else:
            s_unkn_fun = lambda x: 0

        comparetol = 1e-4
        maxevals = 100
        delta = 2
        nsamp = round(maxevals / 4)

        # For known constraints
        use_linear_constraints = 0
        use_nl_constraints = 0

        if use_linear_constraints: # placeholder, if relavent, add the required constraints
            Aineq = array([])

            bineq = array([])
        else:
            Aineq = array([])
            bineq = array([])

        if use_nl_constraints:
            g = array([])
        else:
            g = array([])

    elif benchmark == "CHSC":
        # CamelSixHumps function with feasibility and satisfactory constraints
        nvars = 2
        lb = array([-2.0, -1.0])
        ub = array([2.0, 1.0])
        fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                         x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
        xopt0 = array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
        fopt0 = -1.0316  # unconstrained optimum
        xopt_const = array([0.0781, 0.6562])  # constrained optimizers
        fopt_const =  -0.9050 #constrained optimum

        # For unknown constraints
        isUnknownFeasibilityConstrained = 1  # isUnknownFeasibilityConstrained = 1, if unknown inequality constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 1  # isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist, isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained == 1:
            g_nl_unkn = lambda x: array([x[0] ** 2 + (x[1] + 0.04) ** 2 - .8])

            g_unkn_fun = lambda x: array([sum(maximum(g_nl_unkn(x), 0))])
        else:
            g_unkn_fun = lambda x: 0

        if isUnknownSatisfactionConstrained:
            Aineq_unkn = array([[1.6295, 1],
                                [0.5, 3.875],
                                [-4.3023, -4],
                                [-2, 1],
                                [0.5, -1]])

            bineq_unkn = array([[3.0786, 3.324, -1.4909, 0.5, 0.5]])
            s_unkn_fun = lambda x: array([sum(maximum((Aineq_unkn.dot(x) - bineq_unkn).flatten("c"), 0.0))])
        else:
            s_unkn_fun = lambda x: 0

        comparetol = 1e-4
        maxevals = 50
        delta = 1
        nsamp = round(maxevals / 4)

        # For known constraints
        use_linear_constraints = 0
        use_nl_constraints = 0

        if use_linear_constraints: # placeholder, if relavent, add the required constraints
            Aineq = array([])
            bineq = array([])
        else:
            Aineq = array([])
            bineq = array([])

        if use_nl_constraints:
            g = lambda x: array([x[0] ** 2 + (x[1] + 0.1) ** 2 - .5])
        else:
            g = array([])

    ########### benchmarks w/o unknown constraints

    elif benchmark == "1d":
        nvars = 1
        lb = array([-3])
        ub = array([3])
        fun = lambda x: (1+x*sin(2*x)*cos(3*x)/(1+x**2))**2+x**2/12+x/10
        xopt0 = -0.956480816387759  # unconstrained optimizers, one per column
        fopt0 = 0.279546426870577  # unconstrained optimum
        # xopt_const = array([0.21305, 0.57424])  # constrained optimizers
        # fopt_const =  -0.58445 #constrained optimum

        # For unknown constraints
        isUnknownFeasibilityConstrained = 0  # isUnknownFeasibilityConstrained = 1, if unknown inequality constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0  # isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist, isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained == 1:
            g_unkn_fun = lambda x: 0 # placeholder, if relavent, add the required constraints
        else:
            g_unkn_fun = lambda x: 0

        if isUnknownSatisfactionConstrained:
            s_unkn_fun = lambda x: 0
        else:
            s_unkn_fun = lambda x: 0

        comparetol = 1e-4
        maxevals = 25
        delta = 2
        nsamp = 10

        # For known constraints
        use_linear_constraints = 0
        use_nl_constraints = 0

        if use_linear_constraints:
            Aineq = array([]) # placeholder, if relavent, add the required constraints
            bineq = array([])
        else:
            Aineq = array([])
            bineq = array([])

        if use_nl_constraints:
            g = array([])
        else:
            g = array([])

    elif benchmark == "camelsixhumps":
        # CamelSixHumps function
        nvars = 2
        lb = array([-2.0, -1.0])
        ub = array([2.0, 1.0])
        fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                         x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
        xopt0 = array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
        fopt0 = -1.0316  # unconstrained optimum
        # xopt_const = array([0.21305, 0.57424])  # constrained optimizers
        # fopt_const =  -0.58445 #constrained optimum

        # For unknown constraints
        isUnknownFeasibilityConstrained = 0  # isUnknownFeasibilityConstrained = 1, if unknown inequality constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0  # isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist, isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained == 1:
            g_unkn_fun = lambda x: 0 # placeholder, if relavent, add the required constraints
        else:
            g_unkn_fun = lambda x: 0

        if isUnknownSatisfactionConstrained:
            s_unkn_fun = lambda x: 0
        else:
            s_unkn_fun = lambda x: 0

        comparetol = 1e-4
        maxevals = 50
        delta = 1
        nsamp = round(maxevals / 3)

        # For known constraints
        use_linear_constraints = 0
        use_nl_constraints = 0

        if use_linear_constraints:
            Aineq = array([]) # placeholder, if relavent, add the required constraints

            bineq = array([])
        else:
            Aineq = array([])
            bineq = array([])

        if use_nl_constraints:
            g = array([])
        else:
            g = array([])

    elif benchmark == "camelsixhumps-constr":
        # CamelSixHumps function with known constraints
        nvars = 2
        lb = array([-2.0, -1.0])
        ub = array([2.0, 1.0])
        fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                         x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
        xopt0 = array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
        fopt0 = -1.0316  # unconstrained optimum
        xopt_const = array([0.21305, 0.57424])  # constrained optimizers
        fopt_const =  -0.58445 #constrained optimum

        # For unknown constraints
        isUnknownFeasibilityConstrained = 0  # isUnknownFeasibilityConstrained = 1, if unknown inequality constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0  # isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist, isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained == 1:
            g_unkn_fun = lambda x: 0 # placeholder, if relavent, add the required constraints
        else:
            g_unkn_fun = lambda x: 0

        if isUnknownSatisfactionConstrained:
            s_unkn_fun = lambda x: 0
        else:
            s_unkn_fun = lambda x: 0

        comparetol = 1e-4
        maxevals = 50
        delta = 0.3
        nsamp = round(maxevals / 3)

        # For known constraints
        use_linear_constraints = 1
        use_nl_constraints = 1

        if use_linear_constraints:
            Aineq = array([[1.6295, 1],
                           [-1, 4.4553],
                           [-4.3023, -1],
                           [-5.6905, -12.1374],
                           [17.6198, 1]])

            bineq = array([[3.0786, 2.7417, -1.4909, 1, 32.5198]])
        else:
            Aineq = array([])
            bineq = array([])

        if use_nl_constraints:
            g = lambda x: array([x[0] ** 2 + (x[1] + 0.1) ** 2 - .5])
        else:
            g = array([])


    if runGLISp:
        if isUnknownSatisfactionConstrained and isUnknownFeasibilityConstrained:
            pref_fun = glisp_function3(fun, comparetol,g_unkn_fun,s_unkn_fun) # Include query for both unknown feasibility and satisfactory constraints besides the preference query
        elif ~isUnknownSatisfactionConstrained and isUnknownFeasibilityConstrained:
            pref_fun = glisp_function2(fun, comparetol, g_unkn_fun) # Inlude query for only feasibility constraints besides the preference query
        elif isUnknownSatisfactionConstrained and ~isUnknownFeasibilityConstrained:
            pref_fun = glisp_function2(fun, comparetol, g_unkn_fun) #Inlude query for only satisfactory constraints besides the preference query
        else:
            pref_fun = glisp_function1(fun, comparetol, Aineq,bineq,g) #with only preference query

    pref = lambda x, y: pref_fun.eval(x, y)

    problem = glis.glisp.default(nvars)

    problem["Aineq"] = Aineq
    problem["bineq"] = bineq
    problem["g"] = g
    problem["isUnknownFeasibilityConstrained"] = isUnknownFeasibilityConstrained
    problem["isUnknownSatisfactionConstrained"] = isUnknownSatisfactionConstrained
    problem["g_unkn_fun"] = g_unkn_fun
    problem["s_unkn_fun"] = s_unkn_fun

    problem["lb"] = lb
    problem["ub"] = ub
    problem["maxevals"] = maxevals
    problem["sepvalue"] = 1 / maxevals
    problem["pref"] = pref

    epsil = 1
    problem["epsil"] = epsil

    problem["RBFcalibrate"] = RBFcalibrate
    problem["thetas"] = logspace(-1, 1, 10, False)

    problem["delta"] = delta

    problem["nsamp"] = nsamp
    problem["svdtol"] = 1e-6
    # problem["globoptsol"] = "direct"
    problem["globoptsol"] = "pswarm"
    problem["display"] = 1

    problem["scalevars"] = 1
    problem["compare_tol"] = 1e-6

    problem["constraint_penalty"] = 1e5
    problem["feasible_sampling"] = False

    if runGLISp:
        problem["rbf"] = lambda x1, x2, epsil: 1 / (
                    1 + epsil ** 2 * vecsum((x1 - x2) ** 2, axis=-1))  # inverse quadratic
        # problem["rbf"] = lambda x1,x2,epsil: exp(-(epsil**2*vecsum((x1-x2)**2,axis=-1)) # Gaussian RBF
        # problem["rbf"] = lambda x1,x2,epsil: sqrt((1+epsil**2*vecsum((x1-x2)**2,axis=-1)) # multiquadric
        print("Running GLISp optimization:\n")
    if runGLIS:
        problem["useRBF"] = 1
        problem["alpha"] = delta/5
        problem["f"] = fun
        if problem["useRBF"]:
            epsil = .5
            def fun_rbf(x1, x2):
                return 1 / (1 + epsil ** 2 * vecsum((x1 - x2) ** 2, axis=-1))
            problem["rbf"] = fun_rbf
        print("Running GLIS optimization:\n")

    viridis = plt.cm.get_cmap('viridis', Ntests)

    xopt_Ntests = zeros((Ntests, nvars))
    fopt_Ntests = zeros((Ntests, 1))
    feas_unkn_Ntests = zeros((Ntests, 1))
    fesseq_unkn_Ntest = zeros((Ntests, maxevals))
    fes_first_unkn_Ntest = zeros((Ntests, 1))
    satConst_unkn_Ntests = zeros((Ntests, 1))
    satConstseq_unkn_Ntest = zeros((Ntests, maxevals))
    feas_comb_Ntests = zeros((Ntests, 1))
    feascombseq_unkn_Ntest = zeros((Ntests, maxevals))
    ibest_Ntests = zeros((Ntests, 1)).astype(int)
    ibestseq_Ntests = zeros((Ntests,maxevals)).astype(int)

    for i in range(0, Ntests):
        tic = time.perf_counter()
        if runGLISp:
            pref_fun.clear()  # reset preference function
            out = glis.glisp.solve(problem)
        elif runGLIS:
            out = glis.glis.solve(problem)
        toc = time.perf_counter()
        print("Test # %2d, elapsed time: %5.4f" % (i + 1, toc - tic))

        xopt1 = out["xopt"]
        xopt_Ntests[i] = xopt1
        feas_unkn_Ntests[i] = out["fes_opt_unkn"]
        satConst_unkn_Ntests[i] = out["satConst_opt_unkn"]
        feas_comb_Ntests[i] = out["feas_opt_comb"]
        ibest_Ntests[i] = out["ibest"]
        ibestseq_Ntests[i,:] = out["ibestseq"].ravel()
        fesseq_unkn_Ntest[i,:] = out["Feasibility_unkn"].ravel()
        satConstseq_unkn_Ntest[i,:] = out["SatConst_unkn"].ravel()
        feascombseq_unkn_Ntest[i,:] = out["isfeas_seq"].ravel()

        X = out["X"]
        fopt_Ntests[i] = fun(X[ibest_Ntests[i],:].T)

        if runGLISp:
            F = zeros(maxevals)
            minf = zeros(maxevals)
            for j in range(maxevals):
                F[j] = fun(X[j,:])
                # (F[j],_) = pref_fun.value(X[j,:])
                minf[j] = F[ibestseq_Ntests[i,j]]
            # minf[j] = min(F[0:j + 1])
            # if (minf[j] < fopt_Ntests[i]) and (j <ibest_Ntests[i]):
            #     minf[j] = minf[j-1]
            # elif (minf[j] < fopt_Ntests[i]) and (j >ibest_Ntests[i]):
            #     minf[j] = fopt_Ntests[i]
        else: #run GLIS
            F = out["F"]
            minf = zeros((maxevals, 1))
            for j in range(maxevals):
                minf[j] = F[ibestseq_Ntests[i,j]]

        # fopt_Ntests[i] = minf[-1]

        plt.plot(arange(0, maxevals), minf, color=viridis(i))

    plt.xlabel("preference queries")
    plt.title("Best value of latent function")

    print("\nTotal CPU time: %5.1f s\n" % (time.time() - TIME0))

    plt.grid()
    if Ntests == 1:
        thelegend = ["function values"]
        if not (use_linear_constraints or use_nl_constraints or isUnknownFeasibilityConstrained or isUnknownFeasibilityConstrained):
            plt.plot(arange(0, maxevals), fopt0 * ones(maxevals))
        else:
            plt.plot(arange(0, maxevals), fopt_const * ones(maxevals))
        axes = plt.gca()
        ylim = axes.get_ylim()
        ymax = ylim[1]
        ymin = ylim[0]
        thelegend.append("optimum")
        plt.legend(thelegend)
    plt.show()

    if Ntests == 1 and nvars == 2:

        fig, ax = plt.subplots(figsize=(28, 14))

        [x, y] = meshgrid(arange(lb[0], ub[0], .01), arange(lb[1], ub[1], .01))
        z = zeros(x.shape)
        z_g1 = zeros(x.shape)
        z_g2 = zeros(x.shape)

        x_g_fes = []
        y_g_fes = []
        z_g_fes = []


        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                z[i, j] = fun(array([x[i, j], y[i, j]]))

        CS = plt.contour(x, y, z, 100, alpha=.4)
        ax.clabel(CS,fmt='%2.4f', colors='k', fontsize=10)
        plt.colorbar()
        plt.plot(X[:, 0], X[:, 1], "*", color=[237 / 256, 177 / 256, 32 / 256], markersize=11)
        plt.plot(xopt0[0,], xopt0[1,], "o", color=[0, 0.4470, 0.7410], markersize=15)
        if use_linear_constraints or use_nl_constraints or isUnknownFeasibilityConstrained or isUnknownFeasibilityConstrained:
            plt.plot(xopt_const[0,], xopt_const[1,], "s", color=[0, 0.9, 0.1], markersize=15)
        plt.plot(xopt1[0], xopt1[1], "*", color=[0.8500, 0.3250, 0.0980], markersize=15)

        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection

        patches = []

        if (use_nl_constraints or isUnknownFeasibilityConstrained) and benchmark == "MBC":  # plot for benchmark MBC
            th = arange(0, 2 * pi, .01)
            N = th.size
            V = zeros((N, 2))
            for i in range(0, N):
                V[i, 0] = -9 + 3 * cos(th[i])
                V[i, 1] = -3 + 3 * sin(th[i])
            circle = mpatches.Polygon(V, True)
            patches.append(circle)

        if (use_nl_constraints or isUnknownFeasibilityConstrained) and (benchmark == "CHC" or benchmark =="camelsixhumps-constr"):  # plot for benchmark CHC
            th = arange(0, 2 * pi, .01)
            N = th.size
            V = zeros((N, 2))
            for i in range(0, N):
                V[i, 0] = 0 + sqrt(0.5) * cos(th[i])
                V[i, 1] = -0.1 + sqrt(0.5) * sin(th[i])
            circle = mpatches.Polygon(V, True)
            patches.append(circle)

            V = array([[0.4104, -0.2748], [0.1934, 0.6588], [1.3286, 0.9136],
                       [1.8412, 0.0783], [1.9009, -0.9736]])

            polygon = mpatches.Polygon(V, True)
            patches.append(polygon)

        if (use_nl_constraints or isUnknownFeasibilityConstrained) and benchmark == "CHSC":  # plot for benchmark CHSC
            th = arange(0, 2 * pi, .01)
            N = th.size
            V = zeros((N, 2))
            for i in range(0, N):
                V[i, 0] = 0 + sqrt(0.8) * cos(th[i])
                V[i, 1] = -0.04 + sqrt(0.8) * sin(th[i])
            circle = mpatches.Polygon(V, True)
            patches.append(circle)

            V = array([[1.48,0.667], [0.168,0.836], [ -0.041,0.417],
                       [ 0.554, -0.223], [1.68,0.34]])

            polygon = mpatches.Polygon(V, True)
            patches.append(polygon)

        if use_linear_constraints or use_nl_constraints or isUnknownFeasibilityConstrained or isUnknownSatisfactionConstrained:
            collection = PatchCollection(patches, edgecolor=[0, 0, 0], facecolor=[.5, .5, .5], alpha=0.6)
            ax.add_collection(collection)

        plt.grid()
        plt.show()