import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norminvgauss, uniform, cauchy, poisson

# Instructions for library scipy at https://docs.scipy.org/doc/scipy/reference/stats.html


def normalDistribution(selections, a, b):
    for sz in selections:
        fig, ax = plt.subplots(1, 1);
        x = np.linspace(norminvgauss(b, a).ppf(0.01), norminvgauss(b, a).ppf(0.99), 100);
        ax.hist(norminvgauss.rvs(b, a, size=sz), histtype='stepfilled', alpha=0.5, color='red', density=True);
        ax.plot(x, norminvgauss(b, a).pdf(x), 'b-', lw=2);
        ax.set_title("normal distribution n = " + str(sz));
        ax.set_xlabel("distribution");
        ax.set_ylabel("density");
        plt.grid();
        plt.show();
    return;


def uniformDistribution(selections, a, b):
    for sz in selections:
        fig, ax = plt.subplots(1, 1);
        x = np.linspace(uniform(loc=a, scale=2*b).ppf(0.01), uniform(loc=a, scale=2*b).ppf(0.99), 100);
        ax.hist(uniform.rvs(size=sz, loc=a, scale=2*b), histtype='stepfilled', alpha=0.5, color='red', density=True);
        ax.plot(x, uniform(loc=a, scale=2*b).pdf(x), 'b-', lw=2);
        ax.set_title("uniform distribution n = " + str(sz));
        ax.set_xlabel("distribution");
        ax.set_ylabel("density");
        plt.grid();
        plt.show();
    return;


def cauchyDistribution(selections):
    for sz in selections:
        fig, ax = plt.subplots(1, 1);
        x = np.linspace(cauchy().ppf(0.01), cauchy().ppf(0.99), 100);
        ax.hist(cauchy().rvs(size=sz), histtype='stepfilled', alpha=0.5, color='red', density=True);
        ax.plot(x, cauchy().pdf(x), 'b-', lw=2);
        ax.set_title("cauchy distribution n = " + str(sz));
        ax.set_xlabel("distribution");
        ax.set_ylabel("density");
        plt.grid();
        plt.show();
    return;


def poissonDistribution(selections, k):
    for sz in selections:
        fig, ax = plt.subplots(1, 1);
        x = np.arange(poisson.ppf(0.01, k), poisson.ppf(0.99, k));
        ax.hist(poisson.rvs(k, size=sz), histtype='stepfilled', alpha=0.5, color='red', density=True);
        ax.plot(x, poisson(k).pmf(x), 'b-', lw=2);
        ax.set_title("poisson distribution n = " + str(sz));
        ax.set_xlabel("distribution");
        ax.set_ylabel("density");
        plt.grid();
        plt.show();
    return;


def main():
    # Создадим массив c размерностью выборки
    selections = [10, 50, 1000];
    print('Lab 1');

    normalDistribution(selections, 0, 1);
    uniformDistribution(selections, -math.sqrt(3), math.sqrt(3));
    cauchyDistribution(selections);
    poissonDistribution(selections, 10);


if __name__ == '__main__':
    main()


