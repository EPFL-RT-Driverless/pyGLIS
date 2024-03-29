import numpy as np

from pyGLIS import *


def main():
    glis_instance = GLIS(
        nvar=1,
        f=lambda x: (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10,
        lb=-3.0,
        ub=3.0,
        alpha=1.0,
        delta=0.5,
        maxevals=140,
        verbose=True,
        # load_previous_results=True,
    )
    sol = glis_instance.run()
    print(sol["xopt"])


if __name__ == "__main__":
    main()
