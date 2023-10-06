import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()

X = np.array(
        [[5,37],
         [7,30],
         [10,35],
         [11.5,40],
         [14,38],
         [12,31],
         [35,21.5],
         [39,21.7],
         [34,16],
         [37,17]])
y =np.array([1,1,1,1,1,1,-1,-1,-1,-1])

def fisher_discriminant(X, y):
    # 提取不同类别的样本
    classes = np.unique(y)
    num_classes = len(classes)
    num_features = X.shape[1]

    # 计算每个类别的均值向量
    mean_vectors = []
    for c in classes:
        mean_vectors.append(np.mean(X[y == c], axis=0).tolist())
    mean_vectors = np.array(mean_vectors)

    # 计算类内散度矩阵Sw
    Sw = np.zeros((num_features, num_features))
    for c in classes:
        class_samples = X[y == c]
        class_mean = mean_vectors[classes == c]
        for x in class_samples:
            x_minus_mean = x - class_mean
            Sw += np.outer(x_minus_mean, x_minus_mean)

    # 计算类间散度矩阵Sb
    u = mean_vectors[1:2, :] - mean_vectors[0:1, :]  # u(1) - u(-1)
    Sb = (u).T @ (u)

    w = np.linalg.inv(Sw) @ u.T
    S = (w.T @ (mean_vectors[1:2, :] + mean_vectors[0:1, :]).T) / 2
    # print(w)
    # print(S)

    return w,S

if __name__ == "__main__":
    w,S = fisher_discriminant(X, y)
    print(w)
    # print(S)
    # print(w[0][0])
    # print(w[1][0])
    for i in range(X.shape[0]):
        if y[i] > 0:
            ax.scatter(X[i][0], X[i][1], marker="o", color="blue")
        else:
            ax.scatter(X[i][0], X[i][1], marker="x", color="red")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Original Data")
    x = np.linspace(0, 40, 50)

    y = (-w[0][0]/w[1][0]) * x
    # 方程y，2x+1
    plt.plot(x, y)

    plt.show()