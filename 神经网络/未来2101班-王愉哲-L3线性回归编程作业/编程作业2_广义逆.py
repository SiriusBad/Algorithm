import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
from tqdm import *
import scipy.linalg as sl

def LR(X,y):
    w = np.dot(sl.pinv(X),y)
    L = (1/X.shape[0]) * ((np.linalg.norm(np.dot(X,w) - y)) ** 2)
    return w,L

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

    #广义逆
    w, L = LR(xdata_train, ydata_train)
    print('广义逆最佳解：', w)
    print('广义逆最小误差平方：', L)

    num = 0
    Y_pre = np.dot(xdata_train,w)
    for i in range(Y_pre.shape[0]):
        if Y_pre[i][0] * ydata_train[i][0] > 0:
            num += 1
    print(f'广义逆训练集正确率： {num / Y_pre.shape[0]}')

    num = 0
    Y_pre = np.dot(xdata_test,w)
    for i in range(Y_pre.shape[0]):
        if Y_pre[i][0] * ydata_test[i][0] > 0:
            num += 1
    print(f'广义逆测试集正确率： {num / Y_pre.shape[0]}')


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
