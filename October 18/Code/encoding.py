import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = '../Data/housing.csv'
df = pd.read_csv(file_path)

df_copy = df.copy()

print(df['ocean_proximity'].unique())

# label_encoder = LabelEncoder()
# df_copy['ocean_proximity_encoded'] = label_encoder.fit_transform(df['ocean_proximity'] )

ohe = OneHotEncoder(sparse_output=False)
one_hot_encoded = ohe.fit_transform(df_copy[['ocean_proximity', 'longitude']])

print(one_hot_encoded)

#  Ordinal not working
# oe = OrdinalEncoder(categories=[['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND']], handle_unknown='use_encoded_value', unknown_value=-1)
# df_copy['ocean_proximity_encoded'] = oe.fit_transform(df_copy[['ocean_proximity']])

# print(df_copy.head())


# df_copy.drop('ocean_proximity', axis=1, inplace=True)

# df_copy.dropna(inplace=True)

# x = df_copy.drop('median_house_value', axis=1)
# target = df_copy['median_house_value']
# y = target

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

# linear_reg_model = LinearRegression()
# linear_reg_model.fit(x_train, x_test)
