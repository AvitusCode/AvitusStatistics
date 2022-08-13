import math

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate

# для двух случаев меняем промежутки и размер
A, B = -1.5, 1.5
SIZE = 100
ALPHA = 0.05
GAMMA = 0.95
K = 5


def chi2_quantile(mu, sigma, sample):
    # hyp = lambda x: stats.laplace.cdf(x, loc=mu, scale=sigma/np.sqrt(2))
    hyp = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)
    borders = np.linspace(A, B, num=K-1)
    prob  = np.array(hyp(A))
    quants = np.array(len(sample[sample < A]))

    for i in range(K-2):
        p_i = hyp(borders[i+1]) - hyp(borders[i])
        prob = np.append(prob, p_i)
        n_i = len(sample[(sample < borders[i + 1]) & (sample >= borders[i])])
        quants = np.append(quants, n_i)

    prob = np.append(prob, 1 - hyp(B))
    quants = np.append(quants, len(sample[sample >= B]))

    chi2 = np.divide(np.multiply(quants - SIZE * prob, quants - SIZE * prob), SIZE * prob)

    # quantile = stats.chi2.ppf(GAMMA, K-1)

    return chi2, borders, prob, quants


def ms(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)

    return mu, sigma


def make_table(chi2, borders, prob, quants):
    head = ['i',  'border', 'n_i', 'p_i', 'np_i', 'n_i - np_i', '(n_i - np_i)^2/np_i']
    rows = []

    for i in range(0, len(quants)):
        if i == 0:
            limits = ['-inf', np.around(borders[0], decimals=2)]
        elif i == len(quants) - 1:
            limits = [np.around(borders[-1], decimals=2), 'inf']
        else:
            limits = [np.around(borders[i-1], decimals=2), np.around(borders[i], decimals=2)]

        rows.append([i + 1,
                    limits,
                    quants[i],
                    np.around(prob[i], decimals=4),
                    np.around(SIZE*prob[i], decimals=2),
                    np.around(quants[i] - SIZE*prob[i], decimals=2),
                    np.around(chi2[i], decimals=2)]
                    )

    rows.append(['sum',
                 '---',
                 np.sum(quants),
                 np.around(np.sum(prob), decimals=4),
                 np.around(np.sum(SIZE*prob), decimals=2),
                 -np.around(np.sum(quants - SIZE * prob), decimals=2),
                 np.around(np.sum(chi2), decimals=2)
                 ])

    return tabulate(rows, head, tablefmt='latex_raw')


# For lab 7
def do_something(sample):
    mu, sigma = ms(sample)
    chi2, borders, prob, quant = chi2_quantile(mu, sigma, sample)

    print('mu = ', np.around(mu, decimals=2), " sigma = ", np.around(sigma, decimals=2))
    print(make_table(chi2, borders, prob, quant))


# For interval estimates (lab 8)
def interval_mean(sample):
    x_mean = np.mean(sample)
    std = np.std(sample)
    st = stats.t.ppf(1 - ALPHA / 2, SIZE - 1)

    return [x_mean - std * st / (SIZE - 1) ** 0.5, x_mean + std * st / (SIZE - 1) ** 0.5]


def interval_sigma(sample):
    std = np.std(sample)
    chi_left = stats.chi2.ppf(1 - ALPHA / 2, SIZE - 1)
    chi_right = stats.chi2.ppf(ALPHA / 2, SIZE - 1)

    return [std * (SIZE ** 0.5) / (chi_left ** 0.5), std * (SIZE ** 0.5) / (chi_right ** 0.5)]


def interval_asy_mean(sample):
    x_mean = np.mean(sample)
    std = np.std(sample)
    u = stats.norm.ppf(1 - ALPHA / 2)

    return [x_mean - std * u / (SIZE ** 0.5), x_mean + std * u / (SIZE ** 0.5)]


def interval_asy_sigma(sample):
    std = np.std(sample)
    u = stats.norm.ppf(1 - ALPHA / 2)
    e = stats.moment(sample, moment=4) / std ** 4 - 3
    ua = u * (((e + 2) / SIZE) ** 0.5)

    return [std * (1 - 0.5 * ua), std * (1 + 0.5 * ua)]


def interval_check(sample, mean, std):
    print(" n = " + str(SIZE))
    print("mean = " + str(mean))
    print("std = " + str(std) + "\n")

    plt.hist(sample, density=True, alpha=0.2)
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.title('n = ' + str(SIZE))

    plt.vlines(mean[0], 0, 1, color='black', label='minE')
    plt.vlines(mean[1], 0, 1, color='black', label='maxE')
    plt.vlines(mean[0] - std[1], 0, 1, color='red', label='minE - maxD')
    plt.vlines(mean[1] + std[1], 0, 1, color='red', label='maxE + maxD')

    plt.legend()
    plt.show()


def main():
    sample = np.random.normal(0, 1, size=SIZE)
    interval_check(sample, interval_mean(sample), interval_sigma(sample))
    interval_check(sample, interval_asy_mean(sample), interval_asy_sigma(sample))
    # do_something(sample)


if __name__ == '__main__':
    main()
