import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

file_path = '../Data/Titanic-Dataset.csv'
df = pd.read_csv(file_path)

df.dropna(inplace=True)

target = df['Fare']
y=target

x=df.drop(columns=['Name', 'Sex', 'Embarked', 'Ticket', 'Fare', 'Cabin'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

scalar = StandardScaler()

x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled=scalar.fit_transform(x_test)

model=LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred=model.predict(x_test_scaled)
mae=mean_absolute_error(y_test, y_pred)
cal_r2_score=r2_score(y_test, y_pred)
print(mae)
print(cal_r2_score)