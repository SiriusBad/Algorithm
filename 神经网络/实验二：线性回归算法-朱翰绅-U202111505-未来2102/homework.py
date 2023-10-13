import matplotlib.pyplot as plt
import  numpy as np


from metrics import Metric
from LR import LinearRegression
from data_generator import pla_dataset,homework_dataset
from visualize import Plotter

homework_data=homework_dataset()
LR = LinearRegression(learning_rate=0.1, max_epochs=100, batch_size=10)
# LR.gradient_descent_train(homework_data,optimizer='original')
LR.generalized_inverse(homework_data)

# 打印训练后的权重和截距项
print("Weights:", LR.w)
print("Intercept:", LR.b)
prediction = LR.predict(homework_data)
metric=Metric(prediction,homework_data[:,2])
print("Accuracy:", metric.accuracy())
print("F1 Score:", metric.f1_score())
print("Precision:", metric.precision())
print("Recall:", metric.recall())
print("Confusion Matrix:", metric.confusion_matrix())

# 绘制分类决策面
plotter = Plotter()
plotter.plot_decision_boundary(LR, homework_data)
