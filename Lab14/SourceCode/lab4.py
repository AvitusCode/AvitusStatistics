import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import seaborn as sb


class StepFunction(object):
    def __init__(self, x, y, ival=0., sorted=False, side='left'):

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):
        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


class EmpiricalDistributionFunction(StepFunction):
    def __init__(self, x, side='right'):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1./nobs, 1, nobs)
        super(EmpiricalDistributionFunction, self).__init__(x, y, side=side, sorted=True)


def getDistribution(type, size):
    if type == 0:
        return np.random.normal(0, 1, size)
    elif type == 1:
        return np.random.uniform(-math.sqrt(3), math.sqrt(3), size)
    elif type == 2:
        return np.random.standard_cauchy(size)
    elif type == 3:
        return np.random.poisson(10, size)


def getCumulativeDistributionFunction(type, x):
    if type == 0:
        return stats.norm.cdf(x)
    if type == 1:
        return stats.uniform.cdf(x)
    if type == 2:
        return stats.cauchy.cdf(x)
    if type == 3:
        return stats.poisson.cdf(x, 10)


def getProbDensityFunction(type, x):
    if type == 0:
        return stats.norm.pdf(x, 0, 1)
    if type == 1:
        return stats.uniform.pdf(x, -math.sqrt(3), 2*math.sqrt(3))
    if type == 2:
        return stats.cauchy.pdf(x)
    if type == 3:
        return stats.poisson.pmf(10, x)


def makeProcess(estimates):
    typeName = ["normal", "uniform", "cauchy", "poisson"]

    for type in range(4):
        a, b, step = (6, 14, 1) if type == 3 else (-4, 4, 0.01)

        xLen = np.arange(a, b, step)
        container = []

        for sz in estimates:
            container.append([num for num in getDistribution(type, sz) if num >= a or num <= b])

        # plot EF
        i = 1
        for nums in container:
            plt.subplot(1, 3, i)
            plt.title(typeName[type] + " n = " + str(estimates[i - 1]))

            if type | 1:
                plt.step(xLen, getCumulativeDistributionFunction(type, xLen), color='red')
            else:
                plt.plot(xLen, getCumulativeDistributionFunction(type, xLen), color='red')

            xRange = np.linspace(a, b)
            edf = EmpiricalDistributionFunction(nums)
            yRange = edf(xRange)
            plt.step(xRange, yRange, color='black')
            plt.xlabel('x')
            plt.ylabel('F(x)')
            plt.subplots_adjust(wspace=0.8)
            i += 1

        # plt.savefig(typeName[type] + '_edf.png', format='png')
        plt.show()

        # plot KF
        i = 1
        h_names = [r'$h = h_n/2$', r'$h = h_n$', r'$h = 2*h_n$']
        for nums in container:
            fig, ax = plt.subplots(1, 3)
            plt.subplots_adjust(wspace=0.5)
            j = 0
            for hfactor in [0.5, 1, 2]:
                kde = stats.gaussian_kde(nums, bw_method='silverman')
                h_n = kde.factor
                fig.suptitle(typeName[type] + " n = " + str(estimates[i - 1]))
                ax[j].plot(xLen, getProbDensityFunction(type, xLen), color='black', alpha=0.5, label='pdf')
                ax[j].set_title(h_names[j])

                sb.kdeplot(nums, ax=ax[j], bw_method=h_n * hfactor, label='kde', color='red')
                ax[j].set_xlabel('x')
                ax[j].set_ylabel('f(x)')
                ax[j].set_xlim([a, b])
                ax[j].set_ylim([0, 1])
                j += 1

            # plt.savefig(typeName[type] + "_kde" + str(estimates[i - 1]) + '.png', format='png')
            plt.show()
            i += 1


def main():
    estimates = [20, 60, 100]
    makeProcess(estimates)


if __name__ == '__main__':
    main()
