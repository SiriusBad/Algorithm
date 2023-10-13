import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
from tqdm import *

def gradient_descent(X, y, learning_rate=0.001, num_epochs=10000):
    w = np.ones((X.shape[1], 1))
    L = []
    Lin = (1 / X.shape[0]) * ((np.linalg.norm(np.dot(X, w) - y)) ** 2)
    L.append(Lin)
    for i in tqdm(range(num_epochs)):
        l = (2 / X.shape[0]) * (np.dot(np.dot(X.T, X), w) - np.dot(X.T, y))
        w = w - learning_rate * l
        Lin = (1 / X.shape[0]) * ((np.linalg.norm(np.dot(X, w) - y)) ** 2)
        L.append(Lin)
    return w, L

def creat_data(num):
    tot_num = num
    mean_1 = [-5, 0]
    mean_2 = [0, 5]
    matrix_1 = [[1, 0], [0, 1]]
    matrix_2 = [[1, 0], [0, 1]]
    train_data_1 = np.random.multivariate_normal(mean_1, matrix_1, int(tot_num * 0.8))
    train_data_2 = np.random.multivariate_normal(mean_2, matrix_2, int(tot_num * 0.8))
    test_data_1 = np.random.multivariate_normal(mean_1, matrix_1, int(tot_num * 0.2))
    test_data_2 = np.random.multivariate_normal(mean_2, matrix_2, int(tot_num * 0.2))

    data_train = np.vstack((np.hstack((train_data_2, -np.ones(len(train_data_2)).reshape(-1, 1))),
                            np.hstack((train_data_1, np.ones(len(train_data_1)).reshape(-1, 1)))))
    np.random.shuffle(data_train)
    data_train = np.hstack((np.ones(len(data_train)).reshape(-1, 1), data_train))
    data_test = np.vstack((np.hstack((test_data_2, -np.ones(len(test_data_2)).reshape(-1, 1))),
                           np.hstack((test_data_1, np.ones(len(test_data_1)).reshape(-1, 1)))))
    np.random.shuffle(data_test)
    data_test = np.hstack((np.ones(len(data_test)).reshape(-1, 1), data_test))

    return data_train, data_test

if __name__ == '__main__':
    data_train, data_test = creat_data(200)
    xdata_train, ydata_train = data_train[:, :3],data_train[:, 3:]
    xdata_test, ydata_test = data_test[:, :3],data_test[:, 3:]

    #标准梯度下降
    w, L = gradient_descent(xdata_train, ydata_train)
    print('标准梯度下降最佳解：', w)
    print('标准梯度下降最小误差平方：', L[-1])

    num = 0
    Y_pre = np.dot(xdata_train,w)
    for i in range(Y_pre.shape[0]):
        if Y_pre[i][0] * ydata_train[i][0] > 0:
            num += 1
    print(f'标准梯度下降训练集正确率： {num / Y_pre.shape[0]}')

    num = 0
    Y_pre = np.dot(xdata_test,w)
    for i in range(Y_pre.shape[0]):
        if Y_pre[i][0] * ydata_test[i][0] > 0:
            num += 1
    print(f'标准梯度下降测试集正确率： {num / Y_pre.shape[0]}')

    #训练集
    for i in range(xdata_train.shape[0]):
        if ydata_train[i][0] > 0:
            ax.scatter(xdata_train[i][1], xdata_train[i][2], marker="o", color="blue")
        else:
            ax.scatter(xdata_train[i][1], xdata_train[i][2], marker="x", color="red")

    #测试集
    for i in range(xdata_test.shape[0]):
        if ydata_test[i][0] > 0:
            ax.scatter(xdata_test[i][1], xdata_test[i][2], marker="v", color="limegreen")
        else:
            ax.scatter(xdata_test[i][1], xdata_test[i][2], marker="^", color="yellow")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("LR")
    x = np.linspace(-5, 5, 50)
    y = (w[1] * x + w[0]) * ((-1) / w[2])
    # 方程y，2x+1
    plt.plot(x, y)
    plt.show()

    #损失函数随 epoch 增加的变化曲线
    plt.figure(2)
    x2 = []
    for i in range(len(L)):
        x2.append(i)
    plt.plot(x2[:200], L[:200], 'b-')

    # 添加标题和坐标轴标签
    plt.title('L - epochs')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示第二张图
    plt.show()


