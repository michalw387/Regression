import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Regression:

    title: str = None

    def __init__(self, degree=1, h=0.00001, iterations=300, xrange=[-3, 3]) -> None:
        self.h = h
        self.iterations = iterations
        self.xrange = xrange
        self.degree = degree

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.args = None
        self.args_pred = None

        if isinstance(degree, int):
            if degree < 1:
                raise Exception("Wrong regression degree")
            elif degree <= 3:
                regression_names = {
                    1: "Linear",
                    2: "Quadratic",
                    3: "Cubic",
                }
                self.title = regression_names[degree]
            else:
                self.title = f"{degree} degree polynomial"

    def process(self, x, *args, degree=None):
        sum = 0
        if degree is None:
            degree = self.degree + 1
        for i, arg in enumerate(args):
            sum += arg * x ** (degree - i - 1)
        return sum

    def make_dataset(self, *args, n_samples=100, sigma=20):
        self.args = args
        x_data = np.random.uniform(*self.xrange, size=n_samples)
        noise = np.random.normal(0, scale=sigma, size=n_samples)
        y_data = self.process(x_data, *args, degree=len(args)) + noise
        self.split_data(x_data, y_data)

    def split_data(self, x_data, y_data):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_data, y_data
        )

    def get_updated_par(self, y_pred, args):
        outputs = [0] * len(args)
        for i, arg in enumerate(args):
            grad = -np.sum((self.y_train - y_pred) * self.x_train ** (self.degree - i))
            outputs[i] = arg - grad * self.h
        return outputs

    def mse(self, x1, x2):
        return np.sum((x1 - x2) ** 2) / 2

    def fit(self):
        mse_errors = {}
        parameters = [0] * (self.degree + 1)
        best_args = best_error = None
        for _ in range(self.iterations):

            y_pred = self.process(self.x_train, *parameters)

            parameters = self.get_updated_par(y_pred, parameters)

            mse_errors[tuple(parameters)] = self.mse(self.y_train, y_pred)
            error = mse_errors[tuple(parameters)]
            if best_error is None or error < best_error:
                best_error = error
                best_args = parameters.copy()

        self.args_pred = best_args

    def plot(self):
        print(f"Real model parameters: {self.args}")
        print(f"Estimated model parameters: {self.args_pred}")

        x_test = np.linspace(*self.xrange, num=300)
        y_test = self.process(x_test, *self.args_pred)

        y_process = self.process(x_test, *self.args, degree=len(self.args))

        plt.figure(figsize=(8, 6))
        plt.title(self.title + " Regression")
        plt.scatter(self.x_train, self.y_train, s=5.0, label="Data")
        plt.plot(
            x_test, y_process, color="orange", linewidth=3.5, label="Process", alpha=0.5
        )
        plt.plot(
            x_test,
            y_test,
            color="red",
            linewidth=2.5,
            alpha=0.8,
            label=self.title + " model",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(shadow=True)
        plt.grid()
        plt.show()


model_lin = Regression(degree=2, h=0.00001, iterations=300, xrange=[-3, 3])

model_lin.make_dataset(-31, -1, 15, n_samples=500)
model_lin.fit()
model_lin.plot()
