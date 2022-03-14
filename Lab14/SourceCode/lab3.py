import numpy as np
import matplotlib.pyplot as plt
import math
from tabulate import tabulate


def getDistribution(type, size):
    if type == 0:
        return np.random.normal(0, 1, size)
    elif type == 1:
        return np.random.uniform(-math.sqrt(3), math.sqrt(3), size)
    elif type == 2:
        return np.random.standard_cauchy(size)
    elif type == 3:
        return np.random.poisson(10, size)


def tChance(distrib):
    X1 = np.quantile(distrib, 0.25) - 1.5 * (np.quantile(distrib, 0.75) - np.quantile(distrib, 0.25))
    X2 = np.quantile(distrib, 0.25) + 1.5 * (np.quantile(distrib, 0.75) - np.quantile(distrib, 0.25))

    return X1, X2


def emissions(distrib):
    cEmissions = 0
    X1, X2 = tChance(distrib)
    for i in distrib:
        if i < X1 or i > X2:
            cEmissions += 1

    return cEmissions


def generateBoxplot(types):
    # Генерируем боксплот
    for type in range(4):
        plt.boxplot((getDistribution(type, 20), getDistribution(type, 100)), patch_artist=True,
                    boxprops=dict(facecolor='red'), labels=[20, 100])

        plt.title(types[type], fontweight='bold')
        plt.xlabel('n')
        plt.ylabel('x')
        plt.show()


def generateTC(types):
    rows = []
    for type in range(4):
        lowCE = 0
        highCE = 0
        for i in range(1000):
            lowCE += emissions(getDistribution(type, 20))
            highCE += emissions(getDistribution(type, 100))

        rows.append([types[type] + ' n = 20', np.around(lowCE / 1000 / 20, decimals=2)])
        rows.append([types[type] + ' n = 100', np.around(highCE / 1000 / 100, decimals=2)])

    print(tabulate(rows))


def main():
    # do something
    types = ['normal', 'uniform', 'cauchy', 'poisson']
    generateBoxplot(types)
    generateTC(types)


if __name__ == '__main__':
    main()