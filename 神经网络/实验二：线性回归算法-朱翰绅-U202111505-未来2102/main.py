import  numpy as np

from metrics import Metric
from LR import LinearRegression
from data_generator import pla_dataset,homework_dataset
from visualize import Plotter

def prepare_data():
    train1,test1=pla_dataset(mean=[-5,0],size=200,split=True,class_label=1)
    train0,test0=pla_dataset(mean=[0,5],size=200,split=True,class_label=-1)
    train_data,test_data=np.concatenate((train1,train0),axis=0),np.concatenate((test1,test0),axis=0)
    return train_data,test_data

if __name__ == "__main__":

    #prepare training data

    #training
    train_data,test_data=prepare_data()
    # plotter = Plotter(train_data)
    # plotter.plot_scatter()


    # 创建线性回归对象
    LR = LinearRegression(learning_rate=0.01,max_epochs=100,batch_size=32)
    # 使用梯度下降法训练模型，设置 batch_size 和 shuffle
    LR.gradient_descent_train(train_data,optimizer='Adagrad')
    # LR.generalized_inverse(train_data)
    # print(LR.loss)
    pl=Plotter()
    Plotter.plot_loss_curve(LR.loss)

    # 输出模型参数
    print("最终权重 w (梯度下降法):", LR.w)
    print("最终偏置 b (梯度下降法):", LR.b)

    # 进行预测
    Y_pred = LR.predict(test_data)
    # Y_pred = LR.predict(train_data)
    # print("预测结果:", Y_pred)



    #
    #Metric counting
    metric = Metric(Y_pred, test_data[:,2])
    # metric = Metric(Y_pred, train_data[:, 2])
    # metric = Metric(prediction, train_data[:,2])
    print("Accuracy:", metric.accuracy())
    print("F1 Score:", metric.f1_score())
    print("Precision:", metric.precision())
    print("Recall:", metric.recall())
    print("Confusion Matrix:", metric.confusion_matrix())



    #
    # 绘制分类决策面
    plotter=Plotter()
    plotter.plot_decision_boundary(LR,test_data)
