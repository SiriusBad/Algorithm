import numpy as np
from numpy import  ndarray
from timehook import  TimeHook
from metrics import  Metric
class Perceptron:
    def __init__(self, learning_rate=0.1, max_iterations=5,max_epochs=100):
        self.learning_rate = learning_rate
        self.max_iter=max_iterations
        self.max_epochs=max_epochs

    def iter_pocket_train(self, X, learning_rate=None,
                          max_iterations=None,
                          indices:ndarray=None,
                          need_indices:bool=True,
                          shuffle: bool = True):
        # pocket避免了在每一步都接受不稳定的更新，只有在更新确实改善了性能时才进行权重的更改
        # 初始化权重向量，包括截距项
        #为了第五题作业的可视化效果，我们把每一步的best weight 储存起来
        self.best_weights_list=[]
        if learning_rate is None:
            learning_rate=self.learning_rate
        if max_iterations is None:
            max_iterations=self.max_iter
        data, label = X[:, :2], X[:, 2]
        n_samples, n_features = data.shape
        # assert n_samples > max_iterations, "iters must be less than data size,or you can use epoch_base training rules"
        if indices is  None:
            np.random.seed()
            indices = np.random.randint(0, n_samples, max_iterations) if shuffle else max_iterations
        if need_indices:
            print(f'{indices}indices')
        if isinstance(indices, int):
            data_pack = iter(data[:indices])
            label_pack = iter(label[:indices])
        elif isinstance(indices, ndarray):
            data_pack = iter(data[indices])
            label_pack = iter(label[indices])

        self.weights = np.zeros(n_features + 1)  # 初始化有截距的权重
        self.best_weights = None
        self.best_accuracy = 0.0
        self.hook = TimeHook()
        self.hook.start()
        for iteration in range(max_iterations):
            xi, target = next(data_pack), next(label_pack)
            xi_with_bias = np.concatenate(([1], xi))  # 添加截距项
            activation = np.dot(self.weights, xi_with_bias)
            if target * activation <= 0:
                # self.best_weights_list.append([iteration])
                self.weights += target * xi_with_bias * learning_rate
                pocket_pred = self._predict(data)
                metric = Metric(pocket_pred, label)
                accuracy=metric.accuracy()
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_weights = self.weights.copy()
                    # self.best_weights_list.append(self.best_weights)
                    # print(self.best_accuracy)
                else:
                    self.weights=self.best_weights.copy()
                if self.best_accuracy == 1.0:
                    self.hook.end()
                    print(f"Converged after {iteration+1} iterations.")
                    execution_time = self.hook.execution_time()
                    print("pocket Algorithm execution time:", execution_time, "seconds")
                    break
            if iteration == max_iterations -1 :
                self.weights=self.best_weights
                self.hook.end()
                execution_time = self.hook.execution_time()
                print(f"finish at {iteration+1} iterations.")
                print("pocket Algorithm execution time:", execution_time, "seconds")
    def pla_train(self,X, learning_rate=None, max_epochs=None,shuffle:bool=True):
        # 初始化权重向量，包括截距项
        if learning_rate is None:
            learning_rate=self.learning_rate
        if max_epochs is None:
            max_epochs=self.max_epochs
        data, label = X[:, :2], X[:, 2]
        n_samples, n_features = data.shape
        self.weights = np.zeros(n_features + 1)
        # self.errors=[] #如果要打出来每个epoch的错误数量可以打开
        self.hook = TimeHook()
        self.hook.start()

        for epoch in range(max_epochs):

            if shuffle:
                # 打乱数据的顺序
                random_order = np.random.permutation(n_samples)
                data = data[random_order]
                label = label[random_order]
            misclassified = 0
            for xi, target in zip(data,label):
                xi_with_bias = np.concatenate(([1], xi))  # 添加截距项
                activation = np.dot(self.weights, xi_with_bias)
                if target * activation <= 0:
                    self.weights += target * xi_with_bias * learning_rate
                    misclassified += 1
            # self.errors.append(misclassified) #同上
            if misclassified == 0:
                self.hook.end()
                print(f"Converged after {epoch} epochs.")
                execution_time = self.hook.execution_time()
                print("train Algorithm execution time:", execution_time, "seconds")
                break
            if epoch == (max_epochs-1):
                self.hook.end()
                execution_time = self.hook.execution_time()
                print("train Algorithm execution time:", execution_time, "seconds")


    def iter_pla_train(self,X, learning_rate=None, max_iterations=None,
                       need_indices: bool = False,
                       shuffle:bool=True,indices:ndarray=None,):
        # 初始化权重向量，包括截距项
        if learning_rate is None:
            learning_rate=self.learning_rate
        if max_iterations is None:
            max_iterations=self.max_iter
        data, label = X[:, :2], X[:, 2]
        n_samples, n_features = data.shape
        # assert n_samples > max_iterations, "iters must be less than data size,or you can use epoch_base training rules"
        if indices is None:
            np.random.seed()
            indices = np.random.randint(0, n_samples, max_iterations) if shuffle else max_iterations
        if need_indices:
            print(f'{indices}indices')
        if isinstance(indices,int) :
            data_pack=iter(data[:indices])
            label_pack=iter(label[:indices])
        elif isinstance(indices,ndarray):
            data_pack=iter(data[indices])
            label_pack=iter(label[indices])

        self.weights = np.zeros(n_features + 1) #初始化有截距的权重
        self.hook = TimeHook()
        self.hook.start()
        for iteration in range(max_iterations):
            xi,target=next(data_pack),next(label_pack)
            xi_with_bias = np.concatenate(([1], xi))  # 添加截距项
            activation=np.dot(self.weights, xi_with_bias)
            if target*activation <=0:
                self.weights += target*xi_with_bias*learning_rate
        self.hook.end()
        execution_time = self.hook.execution_time()
        print("Algorithm execution time:", execution_time, "seconds")


    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        self.hook = TimeHook()
        self.hook.start()
        data = X[:, :2]
        self.hook.end()
        execution_time = self.hook.execution_time()
        print("test Algorithm execution time:", execution_time, "seconds")
        return np.where(self.net_input(data) >= 0.0, 1, -1)
    def _predict(self, X):
        data = X[:, :2]
        return np.where(self.net_input(data) >= 0.0, 1, -1)


