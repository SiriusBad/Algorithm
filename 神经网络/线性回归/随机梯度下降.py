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

def stochastic_gradient_descent(X, y, learning_rate=0.0001, num_epochs=1000000, batch_size=2):
    num_instances, num_features = X.shape
    num_batches = num_instances // batch_size    #batch个数

    # 初始化模型参数
    w = np.ones((3,1))

    for epoch in tqdm(range(num_epochs)):

        for start in range(0,num_batches,batch_size):
            # 随机选择一个小批量样本
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            # 计算梯度
            L = (2 / X_batch.shape[0]) * (np.dot(np.dot(X_batch.T, X_batch), w) - np.dot(X_batch.T, y_batch))

            # 更新模型参数
            w = w - learning_rate * L

        # 打乱数据集顺序
        indices = np.random.permutation(num_instances)
        X = X[indices]
        y = y[indices]

    return w

if __name__ == '__main__':
    w = stochastic_gradient_descent(xdata,ydata)
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