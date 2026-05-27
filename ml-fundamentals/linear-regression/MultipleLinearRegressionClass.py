class MultipleLinearRegression:
    def __init__(self, n_features, lr=0.01):
        self.b = 0.0
        self.weights = [0.0] * n_features
        self.lr = lr
        self.cost_history = []

    def predict(self, X):
        ans = []
        for x in X:
            ans.append(sum(w * xi for w, xi in zip(self.weights, x)) + self.b)
        return ans
    
    def cost_compute(self, X, y):
        y_pred = self.predict(X)
        n = len(y)
        cost = sum((i - j) ** 2 for i, j in zip(y_pred, y)) / n
        return cost
    
    def grad_compute(self, X, y):
        y_pred = self.predict(X)
        n = len(y)
        dw = []
        db = (2/n) * sum((i - j) for i, j in zip(y_pred, y))
        for i in range(len(X[0])):
            xi = []
            for x in X:
                xi.append(x[i])
            grad = (2/n) * sum((j - k) * xii for xii, j, k in zip(xi, y_pred, y))
            dw.append(grad)
        return dw, db
    
    def fit(self, X, y, epochs=1000, print_epoch=200):
        for i in range(epochs):
            dw, db = self.grad_compute(X, y)
            self.weights = [weight - self.lr * dwi for weight, dwi in zip(self.weights, dw)]
            self.b -= self.lr * db
            cost = self.cost_compute(X, y)
            self.cost_history.append(cost)
            if i % print_epoch == 0:
                print(f"Cost {cost:.4f} at epoch {i:4d}")
        return self
    
    def r_squared(self, X, y):
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((i - j) ** 2 for i, j in zip(y_pred, y))
        ss_tot = sum((i - y_mean) ** 2 for i in y)
        return 1 - (ss_res / ss_tot)