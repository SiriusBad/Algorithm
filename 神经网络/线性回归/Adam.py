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

def Adam(X, y, learning_rate=0.0001, num_epochs=1000000, batch_size=2):
    global m_hat, v_hat
    num_instances, num_features = X.shape
    num_batches = num_instances // batch_size    #batch个数

    # 初始化模型参数
    w = np.zeros((3,1))
    m =np.zeros((3,1))
    v =np.zeros((3,1))
    m_hat = np.zeros((3,1))
    v_hat = np.zeros((3,1))
    epsilon = 1e-60
    beta1 = 0.9
    beta2 = 0.999
    t = 0

    for epoch in tqdm(range(num_epochs)):

        for start in range(0,num_batches,batch_size):
            t = t + 1
            # 随机选择一个小批量样本
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            gradient_w = (2 / X_batch.shape[0]) * (np.dot(np.dot(X_batch.T, X_batch), w) - np.dot(X_batch.T, y_batch))

            if epoch == 0 and start == 0:
                m = m
                v = v
            else:
                m = beta1 * m + (1 - beta1) * gradient_w
                v = beta2 * v + (1 - beta2) * (gradient_w ** 2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

            w -= (learning_rate / (np.sqrt(v_hat) + epsilon)) * m_hat

        # 打乱数据集顺序
        indices = np.random.permutation(num_instances)
        X = X[indices]
        y = y[indices]

    return w

if __name__ == '__main__':
    w = Adam(xdata,ydata)
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