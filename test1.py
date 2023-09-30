import numpy as np
T = np.array([[3, 3, 1], [4, 3, 1], [1, 1, -1]])
print(np.insert(T[0][0:T.shape[1] - 1],0,1))
# print(T[0])

