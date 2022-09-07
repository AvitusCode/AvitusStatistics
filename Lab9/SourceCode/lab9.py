import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import csv


# Regression class; least modules method
class Regression:
    def __init__(self):
        self.B0 = 0
        self.B1 = 0

    def solve(self, x, y):
        self.B1 = self.RQ(x, y) * ((np.quantile(y, 3 / 4) - np.quantile(y, 1 / 4)) / (len(y) * 0.5)
                                       / ((np.quantile(x, 3 / 4) - np.quantile(x, 1 / 4)) / (len(x) * 0.5)))
        self.B0 = np.median(y) - self.B1 * np.median(x)

    def RQ(self, x, y):
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0

        x_med = np.median(x)
        y_med = np.median(y)

        for i in range(len(x)):
            if (x[i] >= x_med) and (y[i] >= y_med):
                n1 += 1
            if (x[i] < x_med) and (y[i] >= y_med):
                n2 += 1
            if (x[i] < x_med) and (y[i] < y_med):
                n3 += 1
            if (x[i] >= x_med) and (y[i] < y_med):
                n4 += 1
        return ((n1 + n3) - (n2 + n4)) / len(x)

    def get_y(self, x):
        return self.B0 + self.B1 * x

    def get_B0(self):
        return self.B0

    def get_B1(self):
        return self.B1


def read_data(file):
    data = [[], []]

    with open(file, newline='') as fl:
        reader = csv.reader(fl, delimiter=';')
        for i in reader:
            try:
                data[0].append(float(i[0]))
                data[1].append(float(i[1]))
            except ValueError:
                pass

    return np.array(data)


# Первая часть задания (выводим данные на экран)
def plot_data(data1, data2):
    n = np.array([i for i in range(len(data1[0]))])

    ax = plt.gca()
    ax.plot(n, data1[0], color="blue", label='ch-1.csv')
    ax.plot(n, data2[0], color="red", label='ch-2.csv')
    ax.set_title('Data from experiment')
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    ax.grid()
    ax.legend()
    plt.show()


# Вторая часть, отображаем интервалы
def plot_intervals(data, title='', cl=''):
    eps = 0.0001
    ax = plt.gca()
    ax.set_ylabel('mV')
    ax.set_xlabel('n')
    ax.set_title(title)
    ax.grid()

    for i in range(len(data)):
        ax.vlines(i, data[i] - eps, data[i] + eps, color=cl)

    plt.show()


# зададим функцию W(n)
def W_FUNC(n):
    regression = Regression()
    x = np.array([i for i in range(len(n))])
    regression.solve(x, n)

    Lin_n = regression.get_B0() + x * regression.get_B1()  # Формула (10)
    b1 = Lin_n - n
    b2 = -Lin_n + n
    c = [1 for i in range(len(Lin_n))]

    A = (-1.0) * np.eye(len(n)) * 0.0001
    A = np.concatenate((A, A), axis=0)
    B = np.concatenate((b1, b2), axis=0)

    result = opt.linprog(method='simplex', c=c, A_ub=A, b_ub=B, bounds=[1, None])

    return result.x, regression.get_B0(), regression.get_B1()


def plot_regression(data, w, B0, B1, title, cl):
    eps = 0.0001
    ax = plt.gca()
    ax.set_ylabel('mV')
    ax.set_xlabel('n')
    ax.set_title(title + ' linear regression')
    ax.grid()

    for i in range(len(data)):
        ax.vlines(i, data[i] - w[i]*eps, data[i] + w[i]*eps, color=cl)

    n = np.array([i for i in range(len(data))])
    y = B0 + n * B1

    ax.plot(n, y, color='green', label='Lin(n)')
    ax.legend()
    plt.show()

    # without drifting
    ax = plt.gca()
    ax.set_ylabel('mV')
    ax.set_xlabel('n')
    ax.set_title(title + ' without linear drifting')
    ax.grid()

    for i in range(len(data)):
        ax.vlines(i, data[i] - w[i]*eps - B1*i, data[i] + w[i]*eps - B1*i, color=cl)

    y = [B0 for i in range(len(n))]

    ax.plot(n, y, color='green', label='line y = ' + str(B0))
    ax.legend()
    plt.show()

    print("A = " + str(B0))
    print("B = " + str(B1))


def plot_hists(data, w, B0, B1, title, cl):
    ax = plt.gca()
    ax.set_ylabel('w')
    ax.set_xlabel('n')
    ax.set_title(title + ' histogram')
    ax.grid()
    ax.hist(w, color=cl)
    plt.show()

    # correct hist
    n = np.array([i for i in range(len(data))])
    ax = plt.gca()
    ax.set_ylabel('w')
    ax.set_xlabel('n')
    ax.set_title(title + ' correct histogram')
    ax.grid()
    ax.hist(data - B1*n, color=cl)
    plt.show()


def Jaccard(data1, data2, w1, A1, B1, w2, A2, B2):
    eps = 0.0001
    n = np.array([i for i in range(len(data1))])
    dt1c = data1 - n * B1
    dt2c = data2 - n * B2

    jaccars = []
    interval = [0.001 * i + 1 for i in range(400)]

    # Действуем по формуле (13) и (14)
    for R in interval:
        first = [[(dt1c[j] - w1[j]*eps) * R, (dt1c[j] + w1[j]*eps) * R] for j in range(len(n))]
        second = [[(dt2c[j] - w2[j]*eps), (dt2c[j] + w2[j]*eps)] for j in range(len(n))]
        summ = first + second

        intersection = [None, None]
        union = [None, None]

        for dt in summ:
            if intersection[0] is None:
                intersection = dt.copy()
                union = dt.copy()
            else:
                intersection[0] = max(intersection[0], dt[0])
                intersection[1] = min(intersection[1], dt[1])
                union[0] = min(union[0], dt[0])
                union[1] = max(union[1], dt[1])

        jaccars.append((intersection[1] - intersection[0]) / (union[1] - union[0]))

    print(jaccars.index(max(jaccars)) * 0.001 + 0.5)

    opt_r = jaccars.index(max(jaccars)) * 0.001 + 1

    ax = plt.gca()
    ax.set_ylabel('jaccard(r)')
    ax.set_xlabel('R_{21}')
    ax.set_title('Jaccard vs R')
    ax.grid()

    ax.plot(interval, jaccars)
    ax.plot(opt_r, max(jaccars), 'bo', label='optimal point at R = ' + str(round(opt_r, 2)))
    ax.legend()
    ax.set_xlim([1.0, 1.10])
    plt.show()

    print('opt_r = ' + str(opt_r))
    print('jaccars max = ' + str(max(jaccars)))

    ax = plt.gca()
    ax.set_title('Histogram of combined data with optimal R21')
    ax.grid()
    ax.hist(list(dt1c * opt_r) + list(dt2c))
    plt.show()



def main():
    chanal1 = read_data('Канал 1_500nm_2mm.csv')
    chanal2 = read_data('Канал 2_500nm_2mm.csv')

    plot_data(chanal1, chanal2)

    plot_intervals(chanal1[0], 'Ch1 data with intervals', 'blue')
    plot_intervals(chanal2[0], 'Ch2 data with intervals', 'red')

    w1, A1, B1 = W_FUNC(chanal1[0])
    plot_regression(chanal1[0], w1, A1, B1, 'Ch1', 'blue')
    plot_hists(chanal1[0], w1, A1, B1, 'Ch1', 'blue')
    w2, A2, B2 = W_FUNC(chanal2[0])
    plot_regression(chanal2[0], w2, A2, B2, 'Ch2', 'red')
    plot_hists(chanal2[0], w2, A2, B2, 'Ch2', 'red')

    Jaccard(chanal1[0], chanal2[0], w1, A1, B1, w2, A2, B2)


if __name__ == '__main__':
    main()
