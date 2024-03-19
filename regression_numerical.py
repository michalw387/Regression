import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, h=0.0001, iterations=1000, xrange=[-3, 3]) -> None:
        self.h = h
        self.iterations = iterations
        self.xrange = xrange

        self.x_data = None
        self.y_data = None
        self.a = None
        self.b = None
        self.a_pred = None
        self.b_pred = None

    def process_lin(self, x, a=None, b=None):
        if a is None or b is None:
            raise ValueError()
        return a * x + b

    def make_dataset_lin(self, a, b, n_samples=100, sigma=8):
        self.a = a
        self.b = b
        self.x_data = np.random.uniform(*self.xrange, size=n_samples)
        noise = np.random.normal(0, scale=sigma, size=n_samples)
        self.y_data = self.process_lin(self.x_data, a, b) + noise

    def mse(self, x1, x2):
        return np.sum((x1 - x2) ** 2) / 2

    def get_updated_par(self, y_pred, a, b):
        da = -np.sum(self.x_data * (self.y_data - y_pred))
        db = -np.sum(self.y_data - y_pred)
        return (a - da * self.h, b - db * self.h)

    def fit(self):
        mse_errors = {}
        a = b = 0
        best_par = best_error = None
        for _ in range(self.iterations):
            y_pred = self.process_lin(self.x_data, a, b)

            a, b = self.get_updated_par(y_pred, a, b)

            mse_errors[(a, b)] = self.mse(self.y_data, y_pred)

            if best_error is None or mse_errors[(a, b)] < best_error:
                best_error = mse_errors[(a, b)]
                best_par = (a, b)
        self.a_pred = best_par[0]
        self.b_pred = best_par[1]

    def plot(self):
        x_test = np.linspace(*self.xrange, num=300)
        y_test = self.process_lin(x_test, self.a_pred, self.b_pred)

        y_process = self.process_lin(x_test, self.a, self.b)

        plt.figure(figsize=(8, 6))
        plt.title("Proces liniowy")
        plt.scatter(self.x_data, self.y_data, s=5.0, label="dane")
        plt.plot(
            x_test, y_process, color="orange", linewidth=3.5, label="proces", alpha=0.5
        )
        plt.plot(
            x_test,
            y_test,
            color="red",
            linewidth=2.5,
            alpha=0.8,
            label="model liniowy",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(shadow=True)
        plt.grid()
        plt.show()


model = LinearRegression()

model.make_dataset_lin(5, -1)
model.fit()
model.plot()
