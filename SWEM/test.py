import numpy as np
a = [1, 2, 3]
a = np.asarray(a)
print(a)
a = a.reshape((len(a), 3))
print(a)