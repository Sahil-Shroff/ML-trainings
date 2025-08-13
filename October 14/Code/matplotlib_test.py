import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = '../Data/Titanic-Dataset.csv'

df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
# print(df.head())

plt.hist(df["Age"].dropna(), bins=20, color='skyblue', edgecolor="black")
plt.title("")
plt.grid(True)

plt.show()