import math

import numpy as np
import scipy.stats as stats
from tabulate import tabulate

# для двух случаев меняем промежутки и размер
A, B = -1.5, 1.5
SIZE = 20
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


# For interval estimates
def interval_check():
    sz = 100
    sample = np.random.normal(0, 1, size=sz)
    x_m = np.mean(sample)
    x_d = np.std(sample)

    x_moment = stats.moment(sample, moment=4)
    e = x_moment / (x_d**4) - 3
    U = np.quantile(sample, 1-ALPHA/2) * np.sqrt((e + 2) / sz)
    a1s = x_d / np.sqrt(1 + U)
    b1s = x_d / np.sqrt(1 - U)
    a1m = x_m - x_d * np.quantile(sample, 1-ALPHA/2) / np.sqrt(sz)
    b1m = x_m + x_d * np.quantile(sample, 1-ALPHA/2) / np.sqrt(sz)

    print('Asymptotic interval for m ', np.around(a1m, decimals=2), ' ',
          np.around(b1m, decimals=2))
    print('Asymptotic interval for sigma ', np.around(a1s, decimals=2), ' ',
          np.around(b1s, decimals=2))

    temp = stats.norm.interval(alpha=1-ALPHA, loc=x_m, scale=x_d / np.sqrt(sz))
    print('Teor for m ', temp)

    a2s = x_d * np.sqrt(sz) / np.sqrt(stats.chi2.ppf(1-ALPHA/2, sz - 1))
    b2s = x_d * np.sqrt(sz) / np.sqrt(stats.chi2.ppf(ALPHA/2, sz - 1))
    print('Teor for sigma ', a2s, ' ', b2s)


def main():
    interval_check()
    sample = np.random.normal(0, 1, size=SIZE)
    mu, sigma = ms(sample)
    chi2, borders, prob, quant = chi2_quantile(mu, sigma, sample)

    print('mu = ', np.around(mu, decimals=2), " sigma = ", np.around(sigma, decimals=2))
    print(make_table(chi2, borders, prob, quant))


if __name__ == '__main__':
    main()
