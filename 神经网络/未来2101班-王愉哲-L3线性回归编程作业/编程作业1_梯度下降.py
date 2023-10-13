import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
from tqdm import *



def gradient_descent(X, y, learning_rate=0.001, num_epochs=10000):
    w = np.ones((X.shape[1], 1))
    for i in tqdm(range(num_epochs)):
        l = (2 / xdata.shape[0]) * (np.dot(np.dot(xdata.T, xdata), w) - np.dot(xdata.T, ydata))
        w = w - learning_rate * l
    L = (1 / X.shape[0]) * ((np.linalg.norm(np.dot(X, w) - y)) ** 2)
    return w, L

def stochastic_gradient_descent(X, y, learning_rate=0.001, num_epochs=10000, batch_size=2):
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
    L = (1 / X.shape[0]) * ((np.linalg.norm(np.dot(X, w) - y)) ** 2)
    return w,L

def Adagrad(X, y, learning_rate=0.005, num_epochs=100000, batch_size=2):
    num_instances, num_features = X.shape
    num_batches = num_instances // batch_size    #batch个数
    # 初始化模型参数
    w = np.zeros((3,1))
    G_w =np.zeros((3,1))
    epsilon = 1e-60
    for epoch in tqdm(range(num_epochs)):
        for start in range(0,num_batches,batch_size):
            # 随机选择一个小批量样本
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            gradient_w = (2 / X_batch.shape[0]) * (np.dot(np.dot(X_batch.T, X_batch), w) - np.dot(X_batch.T, y_batch))

            G_w += gradient_w ** 2
            w -= (learning_rate / (np.sqrt(G_w) + epsilon)) * gradient_w
        # 打乱数据集顺序
        indices = np.random.permutation(num_instances)
        X = X[indices]
        y = y[indices]

    L = (1 / X.shape[0]) * ((np.linalg.norm(np.dot(X, w) - y)) ** 2)
    return w,L

def RMSProp(X, y, learning_rate=0.001, num_epochs=10000, batch_size=2):
    num_instances, num_features = X.shape
    num_batches = num_instances // batch_size    #batch个数

    # 初始化模型参数
    w = np.zeros((3,1))
    G_w =np.zeros((3,1))
    epsilon = 1e-60
    decay = 0.9

    for epoch in tqdm(range(num_epochs)):
        for start in range(0,num_batches,batch_size):
            # 随机选择一个小批量样本
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            gradient_w = (2 / X_batch.shape[0]) * (np.dot(np.dot(X_batch.T, X_batch), w) - np.dot(X_batch.T, y_batch))

            if epoch == 0 and start == 0:
                G_w = gradient_w ** 2
            else:
                G_w = decay * G_w + (1 - decay) * (gradient_w ** 2)

            w -= (learning_rate / (np.sqrt(G_w) + epsilon)) * gradient_w

        # 打乱数据集顺序
        indices = np.random.permutation(num_instances)
        X = X[indices]
        y = y[indices]

    L = (1 / X.shape[0]) * ((np.linalg.norm(np.dot(X, w) - y)) ** 2)
    return w,L


def Momentum(X, y, learning_rate=0.001, num_epochs=10000, batch_size=2):
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

    L = (1 / X.shape[0]) * ((np.linalg.norm(np.dot(X, w) - y)) ** 2)
    return w,L

def Adam(X, y, learning_rate=0.001, num_epochs=10000, batch_size=2):
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

    L = (1 / X.shape[0]) * ((np.linalg.norm(np.dot(X, w) - y)) ** 2)
    return w,L

if __name__ == '__main__':
    xdata = np.array(
        [[1, 0.2, 0.7],
         [1, 0.3, 0.3],
         [1, 0.4, 0.5],
         [1, 0.6, 0.5],
         [1, 0.1, 0.4],
         [1, 0.4, 0.6],
         [1, 0.6, 0.2],
         [1, 0.7, 0.4],
         [1, 0.8, 0.6],
         [1, 0.7, 0.5]])

    ydata = np.array([[1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1]])

    #标准梯度下降
    w,L = gradient_descent(xdata,ydata)
    print('标准梯度下降最佳解：', w)
    print('标准梯度下降最小误差平方：', L)

    #随机梯度下降
    # w,L = stochastic_gradient_descent(xdata,ydata)
    # print('随机梯度下降最佳解：', w)
    # print('随机梯度下降最小误差平方：', L)

    #Adagrad
    # w,L = Adagrad(xdata,ydata)
    # print('Adagrad最佳解：', w)
    # print('Adagrad最小误差平方：', L)

    #RMSProp
    # w,L = RMSProp(xdata,ydata)
    # print('RMSProp最佳解：', w)
    # print('RMSProp最小误差平方：', L)

    #Momentum
    # w,L = Momentum(xdata,ydata)
    # print('Momentum最佳解：', w)
    # print('Momentum最小误差平方：', L)

    #Adam
    # w, L = Adam(xdata, ydata)
    # print('Adam最佳解：', w)
    # print('Adam最小误差平方：', L)

    for i in range(xdata.shape[0]):
        if ydata[i][0] > 0:
            ax.scatter(xdata[i][1], xdata[i][2], marker="o", color="blue")
        else:
            ax.scatter(xdata[i][1], xdata[i][2], marker="x", color="red")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("LR")
    x = np.linspace(0, 1, 50)
    y = (w[1] * x + w[0]) * ((-1) / w[2])
    # 方程y，2x+1
    plt.plot(x, y)

    plt.show()
