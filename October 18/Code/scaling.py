import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = '../Data/housing.csv'
df = pd.read_csv(file_path)

min_max_scalar = MinMaxScaler()
x_scaled = min_max_scalar.transform(df['total_rooms'])
print(min_max_scalar)