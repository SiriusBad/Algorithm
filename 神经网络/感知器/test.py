import numpy as np
import random
import matplotlib.pyplot as plt
fig,ax=plt.subplots()

class Perceptron(object):
    def __init__(self):
        self.study_step = 1  # 学习步长即学习率
        self.study_total = 100  # 学习次数即训练次数
        self.w_total = 1  # w更新次数

    # 对数据集进行训练
    def train(self, T):
        w = np.zeros(T.shape[1])  # 初始化权重向量为0 [权重都从0开始]
        # print(w)
        # b = 0  # 初始化阈值为0

        # 训练study_total次
        for study in range(self.study_total):
            w_before = w  # 训练前的w值
            # print('w_before:', w_before)
            # b_before = b  # 训练前的b值
            # 训练
            for t in range(T.shape[0]):
                # 计算实际的y值，其期望值为T[0][2]
                X = np.insert(T[t][0:T.shape[1] - 1],0,1)  # X的值
                # print('X:',X)
                Y = T[t][T.shape[1] - 1]  # 期望值
                # print('Y:',Y)
                distin = Y * self.input_X(X, w)
                # 判断X是否是误分类点
                if distin <= 0:
                    w = w + self.study_step * Y * X

                    self.w_total = self.w_total + 1

            # 经过训练后w、b都不在变化，说明训练集中已没有误分类点，那么跳出循环
            if w_before is w :

                break
        return w

    # 得出w*x+b的值
    def input_X(self, X, w):
        return np.dot(X, w)  # wwww**
    #
    # 由X去预测Y值
    def prediction(self, X, w):
        Y = self.input_X(X, w)
        return np.where(Y >= 0, 1, -1)


def creat_data(num):
    tot_num = num
    mean_1 = [1, 0]
    mean_2 = [0, 1]
    matrix_1 = [[1, 0], [0, 1]]
    matrix_2 = [[1, 0], [0, 1]]
    train_data_1 = np.random.multivariate_normal(mean_1, matrix_1, int(tot_num * 0.8))
    train_data_2 = np.random.multivariate_normal(mean_2, matrix_2, int(tot_num * 0.8))
    test_data_1 = np.random.multivariate_normal(mean_1, matrix_1, int(tot_num * 0.2))
    test_data_2 = np.random.multivariate_normal(mean_2, matrix_2, int(tot_num * 0.2))

    data_train = np.vstack((np.hstack((train_data_2, -np.ones(len(train_data_2)).reshape(-1, 1))),
                            np.hstack((train_data_1, np.ones(len(train_data_1)).reshape(-1, 1)))))
    np.random.shuffle(data_train)
    data_test = np.vstack((np.hstack((test_data_2, -np.ones(len(test_data_2)).reshape(-1, 1))),
                           np.hstack((test_data_1, np.ones(len(test_data_1)).reshape(-1, 1)))))
    np.random.shuffle(data_test)

    return data_train, data_test

T ,X= creat_data(200)
# print(X)
per = Perceptron()
w = per.train(T)  # 经过训练得到w
print("Training:", w)

Y = []
Y_pre = []
for x in range(X.shape[0]):
    x1 = np.insert(X[x][0:T.shape[1] - 1], 0, 1)  # X的值
    # print('X:',X)
    x2 = X[x][T.shape[1] - 1]  # 期望值
    Y.append(x2) #
    x2_pre = per.prediction(x1, w)  # 得到X的预测值
    Y_pre.append(x2_pre.tolist()) #

print('预测得到：', Y_pre)
print('实际得到：', Y)
num = 0
for i in range(len(Y_pre)):

    if Y_pre[i] != Y[i]:
        num += 1
print(len(Y_pre))
print(num)

for i in range(T.shape[0]):
    if T[i][2] > 0:
        ax.scatter(T[i][0], T[i][1], marker="o", color="blue")
    else:
        ax.scatter(T[i][0], T[i][1], marker="x", color="red")

for i in range(X.shape[0]):
    if X[i][2] > 0:
        ax.scatter(X[i][0], X[i][1], marker="o", color="green")
    else:
        ax.scatter(X[i][0], X[i][1], marker="x", color="purple")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Original Data")
x = np.linspace(-10, 10, 50)
y = (w[1] * x + w[0]) * ((-1) / w[2])
# 方程y，2x+1
plt.plot(x, y)

plt.show()

