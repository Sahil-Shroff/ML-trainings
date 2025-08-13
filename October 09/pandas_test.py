import pandas as pd

data = {
    'Name': ['Rahul', 'Pawan', 'Sachin', 'Piyush', 'Sahil'],
    'Age': [10, 20, 30, 40, 50],
    'ID': [1, 2, 3, 4, 5],
    'City': ['Delhi', 'Mumbai', 'Chennai', 'Kolkota', 'Chennai']
}


df = pd.DataFrame(data)
print(df.head())