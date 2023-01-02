import random
n = 512
A = [list() for i in range(n)]
for i in range(n):
    for j in range(n):
        A[i].append(random.randint(10, 100))
print(A)


# import math
# a = -math.inf
# print(a<-10000000)

# import numpy as np
# mul = np.zeros((4,9))
# # print(mul)
# print(3*4)