import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

file_path = '../Data/Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# data pre-processing
# print(df.info())
# print(df.isnull().sum())


# encoding
# label_encoder = LabelEncoder()
# df['Sex'] = label_encoder.fit_transform(df['Sex'])
# df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
# df['Ticket'] = label_encoder.fit_transform(df['Ticket'])
# df['Cabin'] = label_encoder.fit_transform(df['Cabin'])
# df['Name'] = label_encoder.fit_transform(df['Name'])
# feature selection

target = df['Fare']

df['Age'] = df['Age'].fillna(df['Age'].mean())
df.drop(columns= ['Name', 'Sex', 'Embarked', 'Ticket', 'Fare', 'Cabin'], inplace=True)
X=df
print(X)
Y=target


# Now we will create Training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
scalar=StandardScaler()


X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled=scalar.fit_transform(X_test)

#model initialisation
model=LinearRegression()
model.fit(X_test_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)
mae=mean_absolute_error(Y_test, Y_pred)