class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_hist = []

    def predict(self, X):
        return [self.w * x + self.b for x in X]
    
    def cost_compute(self, X, y):
        y_pred = self.predict(X)
        tot = sum((i - j) ** 2 for i, j in zip(y_pred, y)) / len(y)
        return tot
    
    def grad_compute(self, X, y):
        y_pred = self.predict(X)
        n = len(y)
        dw = (2/n) * sum((i - j) * x for i, j, x in zip(y_pred, y, X)) 
        db = (2/n) * sum((i - j) for i, j in zip(y_pred, y))
        return dw, db
    
    def fit(self, X, y, epochs=1000, print_epoch=200):
        for i in range(epochs):
            dw, db = self.grad_compute(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.cost_compute(X, y)
            self.cost_hist.append(cost)
            if i % print_epoch == 0:
                print(print(f"  Epoch {i:4d} | Cost: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}"))
        return self
    
    def r_squared(self, X, y):
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((i - j) ** 2 for i, j in zip(y_pred, y))
        ss_tot = sum((i - y_mean) ** 2 for i in y)
        return 1 - (ss_res / ss_tot)