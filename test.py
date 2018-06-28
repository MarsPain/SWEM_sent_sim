import numpy as np
# a = np.array((1,2,3,4,5,6))
# print(a, type(a))
# a = a.reshape((3, 2))
# print(a)

values = np.array([0, 0, 1, 0, 1, 1])
n_values = np.max(values) + 1
print(np.eye(n_values)[values])