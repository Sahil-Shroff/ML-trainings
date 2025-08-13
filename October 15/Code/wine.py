import pandas as pd
from sklearn.datasets import load_wine
# import seaborn as sns

wine_data = load_wine()

wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

wine_df["target"] = wine_data.target

wine_df.head()

from sklearn.preprocessing import StandardScaler

X=wine_df[wine_data.feature_names].copy()
Y=wine_df["target"].copy()

scalar=StandardScaler()
scalar.fit(X)

X_scaled = scalar.fit_transform(X.values)

from sklearn.model_selection import train_test_split

X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X_scaled, Y, train_size=.7, random_state=25)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logistic_regression = LogisticRegression()
svm = SVC()
tree=DecisionTreeClassifier()

logistic_regression.fit(X_train_scaled, Y_train)
svm.fit(X_train_scaled, Y_train)
tree.fit(X_train_scaled, Y_train)

log_pred = logistic_regression.predict(X_test_scaled)
svm_pred = svm.predict(X_test_scaled)
tree_pred=tree.predict(X_test_scaled)

print(log_pred)
print(svm_pred)
print(tree_pred)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

log_accu = accuracy_score(log_pred, Y_test)
svm_accu = accuracy_score(svm_pred, Y_test)
tree_accu = accuracy_score(tree_pred, Y_test)

print(log_accu)
print(svm_accu)
print(tree_accu)

import matplotlib.pyplot as plt
import numpy as np

Xpoints=np.array(['Log', 'SVM', 'Tree'])
Ypoints=np.array([log_accu, svm_accu, tree_accu])
plt.plot(Xpoints, Ypoints, 'h', ms='20', mfc='#4CAF50', linestyle=':', color='r')

plt.xlabel('Model')
plt.title('Model Accuracy', loc='left')
plt.grid()
# plt.scatter(Xpoints, Ypoints)
plt.bar(Xpoints, Ypoints)
plt.show()