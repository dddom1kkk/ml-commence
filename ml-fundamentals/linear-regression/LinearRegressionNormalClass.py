class LinearRegressionNormal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def predict(self, X):
        return [self.w * x + self.b for x in X]
    
    def fit(self, X, y):
        x_mean = sum(X) / len(X)
        y_mean = sum(y) / len(y)

        num = sum((x - x_mean) * (y_ - y_mean) for x, y_ in zip(X, y))
        den = sum((x - x_mean) ** 2 for x in X)
        self.w = num / den
        self.b = (sum(y) - self.w * sum(X)) / len(X)

        return self
    
    def r_squared(self, X, y):
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((i - j) ** 2 for i, j in zip(y_pred, y))
        ss_tot = sum((i - y_mean) ** 2 for i in y)
        return 1 - (ss_res / ss_tot)