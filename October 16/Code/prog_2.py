import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(0)
x = 6 * np.random.rand(100, 1) - 2
y = .5 * x**2 + x + 2 + np.random.randn(100, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

degree = 2
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
x_poly_train = poly_features.fit_transform(x_train)
x_poly_test = poly_features.transform(x_test)

model = LinearRegression()
model.fit(x_poly_train, y_train)

y_pred = model.predict(x_poly_test)

mae = mean_absolute_error(y_test, y_pred)
cal_r2_error = r2_score(y_test, y_pred)

print(mae)
print(cal_r2_error)

plt.figure(figsize=(10, 6))

plt.scatter(x, y, color='blue', label='Original Data', s=30)


x_plot = np.linspace(-3, 3, 100).reshape(100, 1)
x_poly_plot = poly_features.transform(x_plot)
y_plot = model.predict(x_poly_plot)
plt.plot(x_plot, y_plot, color='black', linewidth=2, label='Polynomial Regression Curve')

y_train_pred = model.predict(x_poly_train)
plt.scatter(x_train, y_train, color='green', edgecolor='k', label='Training Data')
plt.scatter(x_train, y_train_pred, color='red', edgecolor='k', label='Training Predictions', alpha=.5)

plt.scatter(x_test, y_test, color='orange', edgecolor='k', label='Test Data')
plt.scatter(x_test, y_pred, color='purple', edgecolor='k', label='Test Predictions', alpha=.5)

plt.show()
