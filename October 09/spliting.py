import numpy as np

vector_one = np.arange(20) + 1
print(np.split(vector_one, 2))

vector_two = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector_three = vector_two.flatten()
print(vector_three.reshape((3, 3)))
