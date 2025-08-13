import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

np.random.seed(0)
x = 2 * np.random.rand(100, 5)
true_coefficients = [4, 3, 0, 0, 0]
y = x.dot(true_coefficients) + np.random.randn(100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
linear_reg_model = LinearRegression()
linear_reg_model.fit(x_train, y_train)

linear_reg_pred = linear_reg_model.predict(x_test)
linear_mae = mean_absolute_error(y_test, linear_reg_pred)
print(linear_mae)

ridge_reg_model = Ridge(alpha=2.0)
ridge_reg_model.fit(x_train, y_train)
ridge_reg_pred = ridge_reg_model.predict(x_test)
ridge_mae = mean_absolute_error(y_test, ridge_reg_pred)
print(ridge_mae)

lasso_reg_model = Lasso(alpha=1.0)
lasso_reg_model.fit(x_train, y_train)
lasso_reg_pred = lasso_reg_model.predict(x_test)
lasso_mae = mean_absolute_error(y_test, lasso_reg_pred)
print(lasso_mae)

elasticnet_reg_model = ElasticNet(alpha=1.0, l1_ratio=.9)
elasticnet_reg_model.fit(x_train, y_train)
elasticnet_reg_pred = elasticnet_reg_model.predict(x_test)
elasticnet_mae = mean_absolute_error(y_test, elasticnet_reg_pred)
print(elasticnet_mae)

plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, color='black', label='Actual Values', alpha=.6)
plt.plot(range(len(y_test)), linear_reg_pred, color='blue', label='Linear Regression', linewidth=2)
plt.plot(range(len(y_test)), ridge_reg_pred, color='red', label='Ridge Regression', linewidth=2)
plt.plot(range(len(y_test)), lasso_reg_pred, color='green', label='Lasso Regression', linewidth=2)
plt.plot(range(len(y_test)), elasticnet_reg_pred, color='purple', label='Elastic Net Regression', linewidth=2)
plt.xlabel('Sample Index')
plt.ylabel('y')
plt.legend()
plt.title('Comparison of regularization Techniques')
plt.show()

# print(x.size, y.size)