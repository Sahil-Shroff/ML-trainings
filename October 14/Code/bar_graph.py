import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = '../Data/Titanic-Dataset.csv'

df = pd.read_csv(file_path)
# survival_by_class = df.groupby('Pclass')['Survived'].mean()
# plt.figure(figsize=(8, 5))

# survival_by_class.plot(kind='bar', color='lightgreen', edgecolor="black")
# plt.title("Survival rate by Passenger Class")
# plt.xlabel("Passenger Class")
# plt.ylabel("Survival Rate")

# # scatter
# plt.figure(figsize=(10, 6))
# plt.scatter(df['Age'], df['Fare'], alpha=.5, c='blue')
# plt.title('Fare vs Age')
# plt.xlabel('Age')
# plt.ylabel('Fare')

#line plot: Fare Over Passenger ID

# print(df.head())
plt.figure(figsize=(10, 6))
plt.plot(df['PassengerId'], df['Fare'], color='purple', linestyle='-', marker='o')
plt.title('Fare Over Passenger IDs')
plt.xlabel('Pasenger ID')
plt.ylabel('Fare')

plt.grid(True)
plt.show()