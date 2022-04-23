import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import numpy as np


def least_squares_method1(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)

    return beta_0, beta_1


def least_squares_method2(x, y, initial_guess):
    ftm = lambda beta: np.sum(np.abs(y - beta[0] - beta[1] * x))
    result = opt.minimize(ftm, initial_guess)
    beta_0R = result['x'][0]
    beta_1R = result['x'][1]

    return beta_0R, beta_1R


def coefficient_estimates(x, y):
    beta_0, beta_1 = least_squares_method1(x, y)
    beta_0R, beta_1R = least_squares_method2(x, y, np.array([beta_0, beta_1]))

    return beta_0, beta_1, beta_0R, beta_1R


def graph_regression(x, y, type, estimates):
    alpha_ls, beta_ls, alpha_lm, beta_lm = estimates
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Model', color='red')
    plt.plot(x, x * (beta_ls * np.ones(len(x))) + alpha_ls * np.ones(len(x)), label='lsk', color='black')
    plt.plot(x, x * (beta_lm * np.ones(len(x))) + alpha_lm * np.ones(len(x)), label='lsm', color='blue')
    plt.scatter(x, y, label="Selection", facecolors='none', edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.savefig(type + '.png', format='png')
    plt.close()


def criteria_comparison(x, estimates):
    alpha_ls, beta_ls, alpha_lm, beta_lm = estimates
    model = lambda x: 2 + 2 * x
    lsc = lambda x: alpha_ls + beta_ls * x
    lmc = lambda x: alpha_lm + beta_lm * x
    sum_ls, sum_lm = 0, 0
    for point in x:
        y_ls = lsc(point)
        y_lm = lmc(point)
        y_model = model(point)
        sum_ls += pow(y_model - y_ls, 2)
        sum_lm += pow(y_model - y_lm, 2)
    print('sum_ls =', sum_ls, " < ", 'sum_lm =', sum_lm) if sum_ls < sum_lm \
        else print('sum_lm =', sum_lm, " < ", 'sum_ls =', sum_ls)


def main():
    x = np.linspace(-1.8, 2, 20)
    y = 2 + 2 * x + stats.norm(0, 1).rvs(20)
    for type in ['Without perturbations', 'With perturbations']:
        estimates = coefficient_estimates(x, y)
        alpha_ls, beta_ls, alpha_lm, beta_lm = estimates
        print(type)
        print("lsk:")
        print('alpha_ls = ' + str(np.around(alpha_ls, decimals=2)))
        print('beta_ls = ' + str(np.around(beta_ls, decimals=2)))
        print("lsm:")
        print('alpha_lm = ' + str(np.around(alpha_lm, decimals=2)))
        print('beta_lm = ' + str(np.around(beta_lm, decimals=2)))
        graph_regression(x, y, type, estimates)
        criteria_comparison(x, estimates)
        y[0] += 10
        y[-1] -= 10


if __name__ == "__main__":
    main()
