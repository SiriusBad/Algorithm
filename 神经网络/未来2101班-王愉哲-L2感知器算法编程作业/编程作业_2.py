import numpy as np
import random
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
import time


class Perceptron(object):
    def __init__(self):
        self.study_step = 0.1  # 学习步长即学习率
        self.study_total = 20  # 学习次数即训练次数
        self.w_total = 1  # w更新次数

    # 对数据集进行训练
    def Pocket_train(self, T):
        w = np.zeros(T.shape[1])  # 初始化权重向量为0 [权重都从0开始]
        w_poc = w
        least_err_amount = T.shape[0]
        # 训练study_total次
        for study in range(self.study_total):
            # 训练
            for t in range(T.shape[0]):
                # 计算实际的y值，其期望值为T[0][2]
                X = np.insert(T[t][0:T.shape[1] - 1],0,1)  # X的值
                Y = T[t][T.shape[1] - 1]  # 期望值
                distin = Y * np.dot(X, w)
                # 判断X是否是误分类点
                if distin <= 0:
                    w = w + self.study_step * Y * X
                    count, choice = self.count_error(T, w)
                    if count < least_err_amount:
                        w_poc = w
                        least_err_amount = count
                        if choice >= 0:
                            x_choices = np.insert(T[choice][0:T.shape[1] - 1], 0, 1)
                            y_choices = T[choice][T.shape[1] - 1]
                            w = w + x_choices * y_choices
                        else:
                            w = w
            print(f'第{study+1}次训练： 错误点有{least_err_amount}个  w = {w}')

        return w_poc,least_err_amount

    def PLA_train(self, T):
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
                distin = Y * np.dot(X, w)
                # 判断X是否是误分类点
                if distin <= 0:
                    w = w + self.study_step * Y * X

                    self.w_total = self.w_total + 1

            # 经过训练后w、b都不在变化，说明训练集中已没有误分类点，那么跳出循环
            if w_before is w :

                break
        return w

    # 得出w*x+b的值

    def count_error(self, T, w):
        errset = []
        count = 0
        for t in range(T.shape[0]):
            X = np.insert(T[t][0:T.shape[1] - 1], 0, 1)  # X的值
            Y = T[t][T.shape[1] - 1]  # 期望值
            distin = Y * np.dot(X, w)
            if distin <= 0:
                errset.append(t)
                count = count + 1
        if len(errset) > 0:
            rand_choice = random.choice(errset)
        else:
            rand_choice = -1
        return count, rand_choice



    # 由X去预测Y值
    def prediction(self, X, w):
        Y = np.dot(X, w)
        return np.where(Y >= 0, 1, -1)

    def creat_data(self,num):
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
        data_test = np.vstack((np.hstack((test_data_2, -np.ones(len(test_data_2)).reshape(-1, 1))),
                               np.hstack((test_data_1, np.ones(len(test_data_1)).reshape(-1, 1)))))
        np.random.shuffle(data_test)

        return data_train, data_test


if __name__ == '__main__':
    per = Perceptron()
    data_train, data_test = per.creat_data(200)
    #Pocket
    t = time.time()
    w1,num1= per.Pocket_train(data_train)  # 经过训练得到w
    print(f'Pocket_coast:{time.time() - t:.4f}s')
    print(f'Pocket_训练集正确率： {1-(num1/data_train.shape[0])}')
    num1_test,t = per.count_error(data_test,w1)
    print(f'Pocket_测试集正确率： {1 - (num1_test / data_train.shape[0])}')

    #PLA
    t = time.time()
    w2= per.PLA_train(data_train)  # 经过训练得到w
    print(f'PLA_coast:{time.time() - t:.4f}s')
    Y = []
    Y_pre = []
    for x in range(data_train.shape[0]):
        x1 = np.insert(data_train[x][0:data_train.shape[1] - 1], 0, 1)  # X的值
        # print('X:',X)
        x2 = data_train[x][data_train.shape[1] - 1]  # 期望值
        Y.append(x2)  #
        x2_pre = per.prediction(x1, w2)  # 得到X的预测值
        Y_pre.append(x2_pre.tolist())  #
    num2 = 0
    for i in range(len(Y_pre)):

        if Y_pre[i] != Y[i]:
            num2 += 1

    print(f'PLA_训练集正确率： {1 - (num2 / len(Y_pre))}')
    Y = []
    Y_pre = []
    for x in range(data_test.shape[0]):
        x1 = np.insert(data_test[x][0:data_train.shape[1] - 1], 0, 1)  # X的值
        # print('X:',X)
        x2 = data_test[x][data_train.shape[1] - 1]  # 期望值
        Y.append(x2)  #
        x2_pre = per.prediction(x1, w2)  # 得到X的预测值
        Y_pre.append(x2_pre.tolist())  #
    num2 = 0
    for i in range(len(Y_pre)):

        if Y_pre[i] != Y[i]:
            num2 += 1

    print(f'PLA_测试集正确率： {1 - (num2 / len(Y_pre))}')

    for i in range(data_train.shape[0]):
        if data_train[i][2] > 0:
            ax.scatter(data_train[i][0], data_train[i][1], marker="o",color="blue")
        else:
            ax.scatter(data_train[i][0], data_train[i][1], marker="x",color="red")
    for i in range(data_test.shape[0]):
        if data_test[i][2] > 0:
            ax.scatter(data_test[i][0], data_test[i][1], marker="o",color="yellow")
        else:
            ax.scatter(data_test[i][0], data_test[i][1], marker="x",color="purple")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Original Data")
    x1 = np.linspace(-10, 10, 50)
    y1 = (w1[1] * x1 + w1[0])*((-1)/w1[2])
    # 方程y，2x+1
    plt.plot(x1, y1)
    y2 = (w2[1] * x1 + w2[0])*((-1)/w2[2])
    # 方程y，2x+1
    plt.plot(x1, y2)

    plt.show()