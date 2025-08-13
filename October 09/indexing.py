import numpy as np

'''
vector_one = np.arange(20)
print(vector_one)

exe_one = vector_one[0]
print(exe_one)

exe_two = vector_one[-1]
print(exe_two)

exe_four = vector_one[1:4]
print(exe_four)

exe_five = vector_one[::2]
print(exe_five)

exe_six = vector_one[::-1]
print(exe_six)

exe_seven = vector_one[-3:]
print(exe_seven)

'''

'''
vector_one = np.arange(9) + 1
print(vector_one[1:10:-1])
print(vector_one[10:1:-1])
print(vector_one[-1:-10:-1])
print(vector_one[-1:-10:1])
'''

vector_two = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(vector_two[:2, :2])