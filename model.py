import numpy as np


class LogisticRegressionFromScratch:
    """Custom Logistic Regression via Gradient Descent."""

    def __init__(self, learning_rate=0.05, iterations=3000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            y_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            cost = (-1 / n_samples) * np.sum(
                y * np.log(y_clipped) + (1 - y) * np.log(1 - y_clipped)
            )
            if i % 100 == 0:
                self.cost_history.append((i, cost))
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias   -= self.lr * db

        print(f"Training Complete. Final Cost: {cost:.4f}")

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
