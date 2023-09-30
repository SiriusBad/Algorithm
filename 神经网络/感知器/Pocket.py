import numpy as np
import random
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
import time


class Perceptron(object):
    def __init__(self):
        self.study_step = 0.1  # 学习步长即学习率
        self.study_total = 200  # 学习次数即训练次数
        self.w_total = 1  # w更新次数

    # 对数据集进行训练
    def train(self, T):
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
                distin = Y * self.input_X(X, w)
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
            print(f'第{study}次训练： 错误点有{least_err_amount}个  w = {w}')

        return w_poc

    # 得出w*x+b的值
    def input_X(self, X, w):
        return np.dot(X, w)  # wwww**

    def count_error(self,T, w):
        errset = []
        count = 0
        for t in range(T.shape[0]):
            X = np.insert(T[t][0:T.shape[1] - 1],0,1)  # X的值
            Y = T[t][T.shape[1] - 1]  # 期望值
            distin = Y * self.input_X(X, w)
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
        Y = self.input_X(X, w)
        return np.where(Y >= 0, 1, -1)


if __name__ == '__main__':
    tot_num = 200
    mean_1 = [1, 0]
    mean_2 = [0, 1]
    matrix_1 = [[1, 0], [0, 1]]
    matrix_2 = [[1, 0], [0, 1]]
    train_data_1 = np.random.multivariate_normal(mean_1, matrix_1, int(tot_num * 0.4))
    train_data_2 = np.random.multivariate_normal(mean_2, matrix_2, int(tot_num * 0.4))
    test_data_1 = np.random.multivariate_normal(mean_1, matrix_1, int(tot_num * 0.1))
    test_data_2 = np.random.multivariate_normal(mean_2, matrix_2, int(tot_num * 0.1))

    data_train = np.vstack((np.hstack((train_data_2, -np.ones(len(train_data_2)).reshape(-1, 1))),
                            np.hstack((train_data_1, np.ones(len(train_data_1)).reshape(-1, 1)))))
    np.random.shuffle(data_train)
    data_test = np.vstack((np.hstack((test_data_2, -np.ones(len(test_data_2)).reshape(-1, 1))),
                           np.hstack((test_data_1, np.ones(len(test_data_1)).reshape(-1, 1)))))
    np.random.shuffle(data_test)

    per = Perceptron()
    T = np.array([[0.200, 0.700, 1], [0.300, 0.300, 1], [0.400, 0.500, 1],[0.600,0.500,1],[0.100,0.400,1],[0.400,0.600,-1],[0.600,0.200,-1],[0.700,0.400,-1],[0.800,0.600,-1],[0.700,0.500,-1]])  # 进行训练的数据集
    # T = data_train
    t = time.time()

    w = per.train(T)  # 经过训练得到w
    print(f'coast:{time.time() - t:.4f}s')
    for i in range(T.shape[0]):
        if T[i][2] > 0:
            ax.scatter(T[i][0], T[i][1], marker="o",color="blue")
        else:
            ax.scatter(T[i][0], T[i][1], marker="x",color="red")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Original Data")
    x = np.linspace(-5, 5, 50)
    y = (w[1] * x + w[0])*((-1)/w[2])
    # 方程y，2x+1
    plt.plot(x, y)
    #
    plt.show()