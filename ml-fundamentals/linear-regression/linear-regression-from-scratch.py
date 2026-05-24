import random
import math
from linearRegressionClass import LinearRegression

random.seed(42)

WEIGHT = 3.0
BIAS = 7.0
N_SAMPLES = 100

X = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
y = [WEIGHT * x + BIAS + random.gauss(0, 2) for x in X] 

# print(X)
# print(y)

model = LinearRegression()
model.fit(X, y)
# y_pred = model.predict(X)
print(f"{model.w:.4f} and {model.b:.4f}")