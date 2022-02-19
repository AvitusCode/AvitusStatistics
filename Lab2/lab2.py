import scipy.stats as ss
import numpy as np
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


def makeRowE(name, mid, medx, z_R, z_Q, z_tr, sz):
    row = [name + " E(z) with n = " + str(sz),
           np.around(np.mean(mid), decimals=6),
           np.around(np.mean(medx), decimals=6),
           np.around(np.mean(z_R), decimals=6),
           np.around(np.mean(z_Q), decimals=6),
           np.around(np.mean(z_tr), decimals=6)];
    return row


def makeRowD(name, mid, medx, z_R, z_Q, z_tr, sz):
    row = [name + " D(z) with n = " + str(sz),
           np.around(np.std(mid) * np.std(mid), decimals=6),
           np.around(np.std(medx) * np.std(medx), decimals=6),
           np.around(np.std(z_R) * np.std(z_R), decimals=6),
           np.around(np.std(z_Q) * np.std(z_Q), decimals=6),
           np.around(np.std(z_tr) * np.std(z_tr), decimals=6)];
    return row


def makeEstimation():
    selections = [10, 100, 1000]
    fields = ["Distribution", "mid", "medx", "z_R", "z_Q", "z_tr"]
    distributions = ["normal", "uniform", "cauchy", "poisson"]
    calculations = 1000

    for type in range(4):
        table_rows = []
        for sz in selections:
            mid, medx, z_R, z_Q, z_tr = [], [], [], [], []
            # Make 1000 estimates
            for calc in range(calculations):
                distrib = getDistribution(type, sz)
                sDistrib = np.sort(distrib)
                mid.append(np.mean(distrib))
                medx.append(np.median(distrib))
                z_R.append((sDistrib[0] + sDistrib[-1]) / 2)
                z_Q.append((np.percentile(distrib, 100 * 1/4) + np.percentile(distrib, 100 * 3/4)) / 2)
                z_tr.append(ss.trim_mean(distrib, 0.25))

            table_rows.append(makeRowE(distributions[type], mid, medx, z_R, z_Q, z_tr, sz))
            table_rows.append(makeRowD(distributions[type], mid, medx, z_R, z_Q, z_tr, sz))

        # TODO: Может сделать удобней, дабы не копировать значения вручную?
        print(tabulate(table_rows, headers=fields, tablefmt="pipe"));


def main():
    makeEstimation();


if __name__ == '__main__':
    main()
