import numpy as np
import matplotlib.pyplot as plt


def compare_exponential_factorial(bases, x):

    xs = np.arange(0, x)

    fact = []
    for x in xs:
        fact.append(np.math.factorial(x))

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.semilogy(xs, fact, label="$f_1 = x!$", color="r")

    exps = []
    for i, b in enumerate(bases):
        exp = np.power(b, xs)
        exps.append(exp)

        ax.plot(xs, exp, label=f"$f_{i + 2} = {b}^x$", color="b", alpha=(i + 1) / len(bases))

    ax.set_xlabel("x")
    ax.set_title("Epic race between Exponentials and Factorial functions", fontsize=14)
    ax.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    compare_exponential_factorial(bases=np.array([2, 3, 4, 5, 8, 10, 15, 50, 85], dtype=np.float), x=150)

