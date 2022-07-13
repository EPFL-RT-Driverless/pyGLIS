# Test global optimization based on IDW and RBF on a benchmark problem
#
# (C) 2019 A. Bemporad, June 14, 2019

import glis.glis
from numpy import array, zeros, ones
from numpy import sum as vecsum
from numpy.random import seed
import time # for tic-toc

# plotting libraries
from numpy import arange, meshgrid
import matplotlib.pyplot as plt

# for comparison with Bayesian Optimization:
import GPyOpt as BO # sheffieldml.github.io/GPyOpt

Ntests=1
run_bayesopt = False

if __name__ == '__main__':
    seed(0) #rng default for reproducibility
    plt.close('all')
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(14,7))

    #benchmark="ackley"
    #benchmark="camelsixhumps"
    benchmark="hartman6"
    #benchmark="rosenbrock8"

    if benchmark=="camelsixhumps":
        # Camel six-humps function
        nvars = 2
        lb = array([-2.0,-1.0])
        ub = array([2.0,1.0])
        fun = lambda x: ((4.0-2.1*x[0]**2+x[0]**4/3.0)*x[0]**2 +
            x[0]*x[1] + (4.0*x[1]**2-4.0)*x[1]**2)
        xopt0=array([[0.0898, -0.0898],[-0.7126, 0.7126]]) # unconstrained optimizers, one per column
        fopt0=-1.0316 # unconstrained optimum
        maxevals=25
        use_linear_constraints = 0
        use_nl_constraints = 0
        if use_linear_constraints or use_nl_constraints:
            run_bayesopt = 0 # constraints not supported

    elif benchmark=="hartman6":
        from math import exp
        nvars = 6
        lb=zeros((nvars,1)).flatten("c")
        ub=ones((nvars,1)).flatten("c")
        alphaH = array([1.0, 1.2, 3.0, 3.2])
        AH = array([[10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14]])
        PH = 1e-4*array([[1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381]])
        def fun(x):
            xx = x.flatten("c")
            f = 0
            for j in range(0,4):
                aux = 0
                for i in range(0,6):
                    aux = aux + (xx[i]-PH[j,i])**2*AH[j,i]
                f = f - exp(-aux)*alphaH[j]
            return f
        fopt0 = -3.32237 # optimum
        xopt0 = array([.20169,.150011,.476874,.275332,.311652,.6573]) # optimizer
        maxevals=80
        use_linear_constraints = 0
        use_nl_constraints = 0

    elif benchmark=="rosenbrock8":
        nvars = 8
        lb = -30*ones((nvars,1)).flatten("c")
        ub = -lb
        def fun(x):
            xx = x.flatten("c")
            f = 0
            for j in range(0,7):
                f = f+100*(xx[j+1]-xx[j]**2)**2+(1.0-xx[j])**2
            return f
        maxevals = 80
        # compute optimum/optimizer by PSO
        from pyswarm import pso # https://pythonhosted.org/pyswarm/
        xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
                           minfunc=1e-12, maxiter=10000)
        use_linear_constraints = 0
        use_nl_constraints = 0

    elif benchmark=="ackley":
        nvars=2
        lb=-5*ones((nvars,1)).flatten("c")
        ub=-lb
        from math import exp, cos, sqrt, pi
        fun = lambda x: array([-20.0*exp(-.2*sqrt(0.5*(x[0]**2+x[1]**2)))-exp(
                0.5*(cos(2.0*pi*x[0])+cos(2.0*pi*x[1])))+exp(1.0)+20.0])
        maxevals=60
        # compute optimum/optimizer by PSO
        from pyswarm import pso # https://pythonhosted.org/pyswarm/
        xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
                           minfunc=1e-12, maxiter=10000)
        use_linear_constraints = 0
        use_nl_constraints = 0

    problem = glis.glis.default(nvars)

    problem["Aineq"] = []
    problem["bineq"] = []

    if use_linear_constraints and benchmark=="camelsixhumps":
        problem["Aineq"] =array([[1.6295, 1],
                                [-1,    4.4553],
                                [-4.3023,   -1],
                                [-5.6905,  -12.1374],
                                [17.6198,    1]])

        problem["bineq"] =array([[3.0786, 2.7417, -1.4909, 1, 32.5198]])

    if use_nl_constraints and benchmark=="camelsixhumps":
        #problem["g"] = lambda x: array([(x[0]-1)**2+x[1]**2-.25,
        #       (x[0]-0.5)**2+(x[1]-0.5)**2-.25])
        problem["g"] = lambda x: array([x[0]**2+(x[1]+0.1)**2-.5])


    problem["lb"] = lb
    problem["ub"] = ub
    problem["f"] = fun
    problem["maxevals"] = maxevals
    problem["useRBF"] = 1 # use Radial Basis Functions
    #problem["useRBF"] = 0 # Inverse Distance Weighting

    if problem["useRBF"]:
        epsil = .5
        def fun_rbf(x1,x2):
            return  1 / (1 + epsil ** 2 * vecsum((x1 - x2) ** 2, axis=-1))
        problem["rbf"] = fun_rbf

    problem["alpha"] = 1
    problem["delta"] = .5

    problem["nsamp"] = 2*nvars
    problem["svdtol"] = 1e-6
    #problem["globoptsol"] = "direct"
    problem["globoptsol"] = "pswarm"
    problem["display"] = 1
    problem["scalevars"] = 1

    problem["constraint_penalty"] = 1e3
    problem["feasible_sampling"] = False

    print("Running GLIS optimization:\n")

    for i in range(0,Ntests):
        tic = time.perf_counter()
        out=glis.glis.solve(problem)
        toc = time.perf_counter()
        print ("Test # %2d, elapsed time: %5.4f" % (i+1,toc-tic))

        xopt1 = out["xopt"]
        fopt1 = out["fopt"]
        F = out["F"]
        X = out["X"]

        minf=zeros((maxevals,1))
        for j in range(maxevals):
            minf[j] = min(F[0:j+1])

        plt.plot(arange(0,maxevals),minf,color=[0.8500, 0.3250, 0.0980])

    if run_bayesopt:

        print("\nRunning Bayesian optimization:\n")

        def fobj(x): #  the objective function must have 2d numpy arrays as input and output
            xx=x[0]
            ff=zeros((1,1))
            ff[0,0]=fun(xx)
            return ff

        space = []
        for i in range(0,nvars):
            space.append({'name': 'x'+str(i+1), 'type': 'continuous', 'domain': (lb[i],ub[i])})

        for i in range(0,Ntests):
            tic = time.perf_counter()
            Bopt = BO.methods.BayesianOptimization(fobj, space)
            Bopt.run_optimization(max_iter=maxevals-5)
            toc = time.perf_counter()
            print ("Test # %2d, elapsed time: %5.4f" % (i+1,toc-tic))
            #Bopt.plot_acquisition()

            xopt2=Bopt.x_opt
            fopt2=Bopt.fx_opt
            XBO=Bopt.X
            FBO=Bopt.Y
            maxevalsBO=FBO.size

            minfBO=zeros((maxevalsBO,1))
            for j in range(maxevalsBO):
                minfBO[j] = min(FBO[0:j+1])

            plt.plot(arange(0,maxevalsBO),minfBO, color=[237/256,177/256,32/256])

    plt.grid()
    if Ntests==1:
        thelegend=["GLIS"]
        if run_bayesopt:
            thelegend.append("BO")
        if not(use_linear_constraints or use_nl_constraints):
            plt.plot(arange(0,maxevals),fopt0*ones(maxevals))
            thelegend.append("optimum")
        plt.legend(thelegend)
    else:
        axes = plt.gca()
        ylim = axes.get_ylim()
        ymax=ylim[1]
        ymin=ylim[0]
        plt.text(5,ymax-(ymax-ymin)/10,"GLIS", color=[0.8500, 0.3250, 0.0980])
        if run_bayesopt:
            plt.text(5,ymax-2*(ymax-ymin)/10,"BO", color=[237/256,177/256,32/256])
    plt.show()


    if Ntests==1 and nvars==2:

        fig, ax = plt.subplots(figsize=(14,7))

        [x,y]=meshgrid(arange(lb[0],ub[0],.01),arange(lb[1],ub[1],.01))
        z=zeros(x.shape)
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                z[i,j]=fun(array([x[i,j],y[i,j]]))

        plt.contour(x,y,z,100,alpha=.4)
        plt.plot(X[:,0],X[:,1],"*", color=[237/256,177/256,32/256], markersize=11)
        plt.plot(xopt0[0,],xopt0[1,],"o", color=[0, 0.4470, 0.7410], markersize=15)
        plt.plot(xopt1[0],xopt1[1],"*", color=[0.8500, 0.3250, 0.0980], markersize=15)

        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection
        patches = []
        if use_linear_constraints:
            V=array([[0.4104, -0.2748],[0.1934, 0.6588],[1.3286, 0.9136],
                     [1.8412, 0.0783],[1.9009, -0.9736]])

            polygon = mpatches.Polygon(V, True)
            patches.append(polygon)

        if use_nl_constraints:
            from math import sin
            th=arange(0,2*pi,.01)
            N=th.size
            V=zeros((N,2))
            for i in range(0,N):
                V[i,0]=0+sqrt(.5)*cos(th[i])
                V[i,1]=-.1+sqrt(.5)*sin(th[i])
            circle = mpatches.Polygon(V, True)
            patches.append(circle)

        if use_linear_constraints or use_nl_constraints:
            collection = PatchCollection(patches, edgecolor=[0,0,0], facecolor=[.5,.5,.5], alpha=0.6)
            ax.add_collection(collection)

        plt.grid()
        plt.show()
