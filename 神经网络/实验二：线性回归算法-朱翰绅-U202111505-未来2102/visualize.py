import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from numpy import  ndarray

class Plotter:
    def __init__(self, X:Optional[ndarray]=None, title="Scatter Plot", xlabel="X-axis", ylabel="Y-axis"):
        if isinstance(X,ndarray):
            self.X = X[:,0]
            self.y = X[:,1]
            self.title = title
            self.xlabel = xlabel
            self.ylabel = ylabel

    def plot_scatter(self):
        plt.figure(figsize=(8, 6))
        colors = ['b' if label == 1 else 'r' for label in self.y]
        plt.scatter(self.X, self.y, c=colors)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()

    @classmethod
    def plot_decision_boundary(cls,model,X):
        plt.figure(figsize=(8, 6))
        colors = ['b' if label == 1 else 'r' for label in X[:,2]]
        plt.scatter(X[:, 0], X[:, 1], c=colors)

        # 绘制分类面
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                             np.linspace(ylim[0], ylim[1], 50))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, colors=['r', 'b'], alpha=0.2)

        plt.title("Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.savefig("F:/plt.png")
        plt.show()


    @classmethod
    def plot_loss_curve(cls,loss_history):
        # 创建一个图形窗口
        plt.figure()

        # 绘制损失曲线
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')

        # 添加标题和标签
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # 显示图例
        plt.legend(['Loss'], loc='upper right')
        plt.savefig("F:/loss.png")


        # 显示损失曲线图
        plt.show()


#
# x=np.array([x for x in range(0,40,1)])
# alpha=0.9
# y=alpha**x+(1-alpha)**(x-1)
#
# # 创建一个新的图形
# plt.figure()
#
# # 绘制连线图并设置数据点样式
# plt.plot(x, y, label='数据线', color='r', marker='o', markersize=6, linestyle='-')
#
# # 设置图形标题和坐标轴标签
# plt.title('Adagrad_weight')
# plt.xlabel('iter_time')
# plt.ylabel('kt')
#
# # 添加图例
# plt.legend()
#
# # 显示图形
# plt.show()
