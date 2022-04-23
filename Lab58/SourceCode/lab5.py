import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy.stats as stats
import tabulate
import statistics
from matplotlib.patches import Ellipse


# GLOBAL CONSTS
SAMPLE = [20, 60, 100]
RHOS   = [0, 0.5, 0.9]
MEAN   = [0, 0]


def get_rvs(num, cov, mixed):
    return stats.multivariate_normal.rvs(MEAN, cov, num) if not mixed \
        else 0.9 * stats.multivariate_normal.rvs(MEAN, [[1, 0.9], [0.9, 1]], num) \
             + 0.1 * stats.multivariate_normal.rvs(MEAN, [[10, -0.9], [-0.9, 10]], num)


def quad_coef(x, y):
    med_x = np.median(x)
    med_y = np.median(y)
    return np.mean(np.sign(x - med_x) * np.sign(y - med_y))


def accumulate_coefs(num, cov, mixed):
    pearson_coefs = []
    spearman_coefs = []
    quad_coefs = []

    for _ in range(1000):
        sample = get_rvs(num, cov, mixed)
        x, y = sample[:, 0], sample[:, 1]
        pearson_coefs.append(stats.pearsonr(x, y)[0])
        spearman_coefs.append(stats.spearmanr(x, y)[0])
        quad_coefs.append(quad_coef(x, y))

    return pearson_coefs, spearman_coefs, quad_coefs


def make_table(num, cov, mixed):
    pearson, spearman, quad = accumulate_coefs(num, cov, mixed)
    p = np.median(pearson)
    s = np.median(spearman)
    q = np.median(quad)
    rows = []

    rows.append(['$E(z)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = np.median([pow(pearson[k], 2) for k in range(1000)])
    s = np.median([pow(spearman[k], 2) for k in range(1000)])
    q = np.median([pow(quad[k], 2) for k in range(1000)])
    rows.append(['$E(z^2)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = statistics.variance(pearson)
    s = statistics.variance(spearman)
    q = statistics.variance(quad)
    rows.append(['$D(z)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    return rows


def make_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    t = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(t + ax.transData)

    return ax.add_patch(ellipse)


def plot_ellipses(samples):
    num = len(samples[0])
    fig, ax = plt.subplots(1, len(samples))
    fig.suptitle("n = " + str(num))
    titles = ['$ \\rho = 0$', '$\\rho = 0.5 $', '$ \\rho = 0.9$']
    i = 0

    for sample in samples:
        x = sample[:, 0]
        y = sample[:, 1]
        ax[i].scatter(x, y, c='blue', s=3)
        make_ellipse(x, y, ax[i], edgecolor='red')
        ax[i].scatter(np.mean(x), np.mean(y), c='black', s=3)
        ax[i].set_title(titles[i])
        i += 1

    plt.savefig("Ellipse n = " + str(num) + ".png", format='png')


def main():
    for num in SAMPLE:
        table = []
        for rho in RHOS:
            cov = [[1.0, rho], [rho, 1.0]]
            extension_table = make_table(num, cov, False)
            title_row = ["$\\rho$ = " + str(rho), '$r$', '$r_S$', '$r_Q$']
            table.append(title_row)
            table.extend(extension_table)
        print(tabulate.tabulate(table, headers=[], tablefmt="latex_raw"))

    table_mixed = []

    for num in SAMPLE:
        extension_table_mixed = make_table(num, None, True)
        title_row = ["$n = " + str(num) + "$", '$r$', '$r_S$', '$r_Q$']
        table_mixed.append(title_row)
        table_mixed.extend(extension_table_mixed)

    print(tabulate.tabulate(table_mixed, headers=[], tablefmt="latex_raw"))

    samples = []
    for num in SAMPLE:
        for rho in RHOS:
            samples.append(get_rvs(num, [[1.0, rho], [rho, 1.0]], False))
        plot_ellipses(samples)
        samples = []


if __name__ == '__main__':
    main()
