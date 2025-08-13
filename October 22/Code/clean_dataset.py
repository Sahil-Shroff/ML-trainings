import pandas as pd

file_path = '../Data/large_dataset_with_issues.csv'

df = pd.read_csv(file_path)

# print(df.info())
# print(df.describe())

df_copy = df.copy()

df['age'] = df['age'].fillna(df['age'].mean())
# fill na value for string, also replace invalid strings
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
df['salary'] = df['salary'].fillna(df['salary'].median())
# change data type of age to int
df['age'] = df['age'].astype('int')

print(df.info())
print(df.describe())
print(df.head())

print(df[['name', 'joining_date']].duplicated().sum())

# drop duplicates
df.drop_duplicates(subset=['name', 'joining_date'], inplace=True)

df['employee_status'] = df['employee_status'].replace({
    'active': 1,
    'terminated': 0
})

print(df.head())

df.to_csv('../Data/new_dataset.csv', index=False)