import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

file_path = '../Data/housing.csv'
df = pd.read_csv(file_path)

print(df.head())

df.dropna(inplace=True)
df.drop(columns= 'ocean_proximity', inplace=True)
target = df['median_house_value']
x = df.drop('median_house_value', axis=1)
y=target

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=.2, random_state=42)

linear_model=LinearRegression()

linear_model.fit(x_train, y_train)
linear_pred=linear_model.predict(x_test)
linear_mae=mean_absolute_error(y_test, linear_pred)
print(linear_mae)

# print(df.head())
# print(df.describe())
# print(df.info())