import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# 定义函数f(x)
def f(x):
    return x * np.cos(0.25 * np.pi * x)

# 梯度函数
def gradient(x):
    return np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)

# 初始化参数
x_init = 2
learning_rate = 0.5
epsilon = 1e-6
iterations = 50
decay=0.8
beta1=0.9
beta2=0.999
def plot_curve(title,x_history,f_history):
    # 绘制函数曲线
    x_range = np.linspace(-6, 6, 100)
    y_range = f(x_range)
    plt.plot(x_range, y_range, label='f(x)')

    # 标注迭代点，前面的用蓝色三角形，最后一个用红色×
    marker_size = 40  # 标记点的大小
    plt.scatter(x_history[:-1], f_history[:-1], c='purple', marker='x', label='Iterations', s=marker_size / 2)
    plt.scatter(x_history[-1], f_history[-1], c='red', marker='^', label='Last Iteration', s=marker_size)

    # 添加标签和图例
    plt.title(f'{title}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.savefig('F:/curve.png')

    # 显示图形
    plt.show()

# 选择不同的优化算法
def gradient_descent(x_init, learning_rate, iterations):
    x_history = []
    f_history = []
    x = x_init
    for i in range(iterations):
        x_history.append(x)
        f_history.append(f(x))
        gradient_x = gradient(x)
        x -= learning_rate * gradient_x
    plot_curve('gradient_descent',x_history, f_history)

def SGD(x_init, learning_rate, iterations):#高斯噪声版
    x_history = []
    f_history = []
    x = x_init
    for i in range(iterations):
        x_history.append(x)
        f_history.append(f(x))
        gradient_x = gradient(x)+np.random.normal(scale=0.1)
        x -= learning_rate * gradient_x
    plot_curve('stochastic_gradient_descent',x_history, f_history)

def Adagrad(x_init, learning_rate, iterations):
    x_history = []
    f_history = []
    epsilon = 1e-6
    gradient_history=0
    x = x_init
    for i in range(iterations):
        x_history.append(x)
        f_history.append(f(x))
        gradient_x = gradient(x)
        gradient_history += gradient_x**2
        x -= (learning_rate / np.sqrt(gradient_history)+epsilon)*gradient_x
    plot_curve('Adagrad',x_history, f_history)

def RMSProp(x_init, learning_rate, iterations, decay):
    x_history = []
    f_history = []
    x = x_init
    epsilon = 1e-6
    G_x=0
    for i in range(iterations):
        x_history.append(x)
        f_history.append(f(x))
        gradient_x=gradient(x)
        G_x = decay * G_x ** 2 + (1 - decay) * gradient_x ** 2
        x -= (learning_rate / (np.sqrt(G_x) + epsilon)) * gradient_x
    plot_curve('RMSProp',x_history, f_history)


def Adam(x_init, learning_rate, iterations,beta1,beta2):
    x_history = []
    f_history = []
    x = x_init
    epsilon = 1e-6
    m_x=0
    v_x=0
    for i in range(iterations):
        t=i+1
        x_history.append(x)
        f_history.append(f(x))
        gradient_x = gradient(x)

        x -= learning_rate * gradient_x
        m_x = beta1 * m_x + (1 - beta1) * gradient_x
        v_x = beta2 * v_x + (1 - beta2) * gradient_x ** 2
        m_x_hat = m_x / (1 - beta1 ** t)
        v_x_hat = v_x/ (1 - beta2 ** t)
        x-= (learning_rate / (np.sqrt(v_x_hat) + epsilon)) * m_x_hat
    plot_curve('Adam',x_history, f_history)






# gradient_descent(x_init, learning_rate, iterations)
# SGD(x_init, learning_rate, iterations)
# Adagrad(x_init, learning_rate, iterations)
# RMSProp(x_init, learning_rate, iterations,decay)
Adam(x_init, learning_rate, iterations,beta1,beta2)



