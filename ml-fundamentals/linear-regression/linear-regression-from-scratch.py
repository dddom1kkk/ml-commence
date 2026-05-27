import random
import math
from LinearRegressionClass import LinearRegression
from LinearRegressionNormalClass import LinearRegressionNormal
from MultipleLinearRegressionClass import MultipleLinearRegression

random.seed(90)

WEIGHT = 3.0
BIAS = 7.0
N_SAMPLES = 100

WEIGHT_MULT = [50, 10000, -1000]
BIAS_MULT = 5000

X = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
y = [WEIGHT * x + BIAS + random.gauss(0, 2) for x in X] 

# print(X)
# print(y)

model = LinearRegression()
model.fit(X, y)
# y_pred = model.predict(X)
print(f"1D Linear Regression {model.w:.4f} and {model.b:.4f}")
print(f"{model.r_squared(X, y):.4f}")

model_normal = LinearRegressionNormal()
model_normal.fit(X, y)

print(f"Normal Equation Linear Regression {model_normal.w:.4f} and {model_normal.b:.4f}")
print(f"{model_normal.r_squared(X, y):.4f}")

X_mult = []
y_mult = []
for _ in range(N_SAMPLES):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1, 5)
    age = random.uniform(0, 50)
    price = WEIGHT_MULT[0] * size + WEIGHT_MULT[1] * bedrooms + WEIGHT_MULT[2] * age + BIAS_MULT + random.gauss(0, 20000)
    X_mult.append([size, bedrooms, price])
    y_mult.append(price)

def standardize(X):
    for i in range(len(X[0])):
        xi = []
        for x in X:
            xi.append(x[i])
        xi_mean = sum(xi) / len(xi)
        xi_variance = sum((xii - xi_mean) ** 2 for xii in xi) / len(xi)
        
    pass

model_mult = MultipleLinearRegression(n_features=3)
model_mult.fit(X_mult, y_mult)

print(model_mult.weights)
print(model_mult.b)