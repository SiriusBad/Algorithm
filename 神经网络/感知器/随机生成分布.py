import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
import scipy.linalg as sl

# 生成训练集xdata
xdata = np.ones((100,2))
for i in range(100):
    xdata[i,1] = 10 * np.random.randn()
# print(xdata)
# 生成训练集ydata
ydata = np.zeros((100,1))
for k in range(100):
    random1 = 3 * np.random.randn()
    ydata[k,0] = 2 * xdata[k,1]+3+random1
# print(ydata)
w = np.dot(sl.pinv(xdata) , ydata)
# print(w[0][0])

for i in range(100):
    ax.scatter(xdata[i][1], ydata[i], marker="x", color="red")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Data")

x = np.linspace(-40, 40, 50)
y = w[1][0] * x + w[0][0]

plt.plot(x, y)

plt.show()


