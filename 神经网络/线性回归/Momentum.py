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

def Momentum(X, y, learning_rate=0.0001, num_epochs=1000000, batch_size=2):
    num_instances, num_features = X.shape
    num_batches = num_instances // batch_size    #batch个数

    # 初始化模型参数
    w = np.zeros((3,1))
    G_w =np.zeros((3,1))
    decay = 0.9

    for epoch in tqdm(range(num_epochs)):

        for start in range(0,num_batches,batch_size):
            # 随机选择一个小批量样本
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            gradient_w = (2 / X_batch.shape[0]) * (np.dot(np.dot(X_batch.T, X_batch), w) - np.dot(X_batch.T, y_batch))

            if epoch == 0 and start == 0:
                G_w = G_w
            else:
                G_w = decay * G_w - learning_rate * gradient_w

            w += G_w

        # 打乱数据集顺序
        indices = np.random.permutation(num_instances)
        X = X[indices]
        y = y[indices]

    return w

if __name__ == '__main__':
    w = Momentum(xdata,ydata)
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