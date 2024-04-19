
"""
The first program will be used to predict the price of a car for a given mileage.
When you launch the program, it should prompt you for a mileage, and then give
you back the estimated price for that mileage. The program will use the following
hypothesis to predict the price :
estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
Before the run of the training program, theta0 and theta1 will be set to 0.
"""
import matplotlib.pyplot as plt
import pandas as pd
from linear import LinearRegression
from linear import minmax_scaler
from linear import Plotter
from linear import convert_minmax
from linear import revert_minmax
from estimate import estimate_price
from IPython import display


def plot_scatter(x, y):
    plt.scatter(x, y)


def plot_price(data, model, minmax, estimate=None):
    if estimate:
        res = model.predict(estimate) * (minmax[1][1] - minmax[0][1]) + minmax[0][1]
        plt.scatter(estimate, res)
    for line in data:
        plt.scatter(line[0], line[1])


def plot_line(minmax, model):
    x = [i for i in range(0, minmax[1][0])]
    y = [
        model.predict(
            convert_minmax(minmax, x_)
        ) * (minmax[1][1] - minmax[0][1]) + minmax[0][1] for x_ in x
    ]
    print(y[:10])
    Plotter.set(x, y)
    Plotter.show()


class App:
    """ Main Application class """
    def __init__(self):
        """ init data"""
        self.data = self.org = pd.read_csv('data.csv').values
        self.linear = LinearRegression()
        self.linear2 = LinearRegression()
        self.minmax = [[0, 0], [0, 0]]

    def run(self):
        """ run program"""
        try:
            [minmax, data] = minmax_scaler(self.data)
            Plotter.set_range(minmax[1][0], minmax[1][1])
            self.linear.history(True)
            plt.ion()
            for i in self.linear.fit(data, learning_rate=0.1, iterations=1000):
                result = i
                plot_line(minmax, self.linear)
            plt.ioff()  # 대화형 모드 비활성화
            plt.show()  # 최종 플롯 보여주기
        except KeyboardInterrupt:
            print(self.linear.get_state())
        plt.show()
        theta = self.linear.get_state()
        theta[1] = theta[1] * (minmax[1][1] - minmax[0][1]) / (minmax[1][0] - minmax[0][0])
        theta[0] = (theta[0] * (minmax[1][1] - minmax[0][1]) + minmax[0][1]
                    - theta[1] * minmax[0][0])
        output = [theta, result]
        with open('output.csv', 'w') as file:
            for x in output:
                file.write(str(x) + '\n')
                print(x)



def main():
    """ main function """
    app = App()
    app.run()


if __name__ == '__main__':
    main()
