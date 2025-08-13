from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y=iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)

clf=DecisionTreeClassifier(max_depth=3, random_state=42)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# accuracy = clf.score(x_test, y_pred)
# print('Model accuracy: {accuracy: .2f}')

accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy: {accuracy: .2f}")

plt.figure(figsize= (10, 6))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Decision Tree for Iris')

plt.show()