import numpy as np

vector_one = np.array([1, 2, 3])
vector_two = np.array([4, 5, 6])

# z=np.array([[1, 2, 3], [4, 5, 6]])
# print(np.dot(vector_one, vector_two))
# print(np.transpose(z))

# print(np.concatenate((vector_one, vector_two), axis=0))

# print(np.hstack((vector_one, vector_two)))
# print(np.vstack((vector_one, vector_two)))

# print(vector_one*2)
# print(vector_one**2)
# print(vector_one+10)

# vector_2xd = np.array([[1, 2, 3], [4, 5, 6]])
# print(vector_2xd*2)
# print(vector_2xd**2)
# print(vector_2xd+10)

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([10, 20, 30])
result = arr2 + arr1
print(result)