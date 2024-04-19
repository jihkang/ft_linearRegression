import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """ train with linear regression model """
    def __init__(self):
        self.theta0 = self.theta1 = 0
        self.historyValue = False

    def cost_calculate(self, data):
        """
         cost calculate function
         will return cost value
         mse
        """
        total_cost = 0
        ret_x = 0
        for (x, y) in data:
            predicted_y = self.predict(x)
            cost = (predicted_y - y) ** 2
            total_cost += cost
            ret_x += x
        return total_cost / (2 * len(data))

    def gradient_descent(self, data, learning_rate):
        """
            gradient descent is used to updated theta
            estimate: theta0 * x0(1) + theta1 * x1
        """
        w = 0
        b = 0
        n = len(data)
        for (x, y) in data:
            predicted_y = self.predict(x) - y
            w += predicted_y * x
            b += predicted_y
        self.theta0 -= (b * learning_rate) / n
        self.theta1 -= (w * learning_rate) / n

    def fit(self, data, learning_rate=0.001, iterations=10):
        """
            :param data:
            :param learning_rate:
            :param iterations:
        """
        for _ in range(iterations):
            self.gradient_descent(data, learning_rate)
            yield self.theta0, self.theta1
        return self.precision(data)

    def predict(self, mileage):
        """ calculate the price of a car for a given mileage"""
        return self.theta1 * mileage + self.theta0

    def precision(self, data):
        """ not correct precision """
        ret = [0 for _ in data]
        for i, line in enumerate(data):
            x, y = line
            ret[i] = (self.predict(x) - y)
            return ret

    def get_state(self):
        return [self.theta0, self.theta1]

    def history(self, flag):
        self.historyValue = flag


class Plotter:
    minmax = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], 'r-')
    ax.set_xlabel('Mileage')
    ax.set_ylabel('Price')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    @staticmethod
    def set_range(x, y):
        Plotter.ax.set_xlim(0, x)
        Plotter.ax.set_ylim(0, y)

    @staticmethod
    def set(x, y):
        Plotter.line.set_data(x, y)  # 라인 데이터 업데이트

    @staticmethod
    def show():
        plt.draw()
        plt.pause(0.01)


def minmax_scaler(data) -> list[[[float, float]]]:
    f_min_value = [float('inf'), float('inf')]
    f_max_value = [float('-inf'), float('-inf')]
    for x, y in data:
        f_min_value[0] = min(f_min_value[0], x)
        f_max_value[0] = max(f_max_value[0], x)
        f_min_value[1] = min(f_min_value[1], y)
        f_max_value[1] = max(f_max_value[1], y)
    return [[f_min_value, f_max_value], [
        [
            (x - f_min_value[0]) / (f_max_value[0] - f_min_value[0]),
            (y - f_min_value[1]) / (f_max_value[1] - f_min_value[1]),
        ] for x, y in data
    ]]


def convert_minmax(minmax, estimate) -> float:
    return (estimate - minmax[0][0]) / (minmax[1][0] - minmax[0][0])


def revert_minmax(minmax, data):
    return [
        [
            x * (minmax[1][0] - minmax[0][0]) + minmax[0][0],
            y * (minmax[1][1] - minmax[0][1]) + minmax[0][1]
        ]
        for (x, y) in data
    ]
