import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
from tqdm import *

xdata = np.array(
        [[1,0.2,0.7],
         [1,0.3,0.3],
         [1,0.4,0.5],
         [1,0.6,0.5],
         [1,0.1,0.4],
         [1,0.4,0.6],
         [1,0.6,0.2],
         [1,0.7,0.4],
         [1,0.8,0.6],
         [1,0.7,0.5]])

ydata =np.array([[1],[1],[1],[1],[1],[-1],[-1],[-1],[-1],[-1]])
w = np.ones((3,1))
n = 0.0001

for i in tqdm(range(10000000)):
    L = (2/xdata.shape[0]) * (np.dot(np.dot(xdata.T, xdata),w)-np.dot(xdata.T, ydata))
    w = w - n * L
print(w)

for i in range(xdata.shape[0]):
    if ydata[i][0] > 0:
        ax.scatter(xdata[i][1], xdata[i][2], marker="o", color="blue")
    else:
        ax.scatter(xdata[i][1], xdata[i][2], marker="x", color="red")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Original Data")
x = np.linspace(0, 1, 50)
y = (w[1] * x + w[0]) * ((-1) / w[2])
# 方程y，2x+1
plt.plot(x, y)

plt.show()