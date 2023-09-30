import numpy as np

class LinearRegression:
    def __init__(self):
        self.learning_rate = 0.1
        self.batch_size=1
        self.max_epochs=100
        self.loss=[]

    def generalized_inverse(self, X):
        X, Y = X[:, :2], X[:, 2]
        num_samples, num_features = X.shape
        X_design = np.column_stack((X, np.ones(num_samples)))
        weights = np.linalg.pinv(X_design).dot(Y)
        self.w = weights[:-1]  # 最后一个元素是偏置
        self.b = weights[-1]
    def gradient_descent_train(self, X, learning_rate=None, max_epochs=None,
                               optimizer:str='orignial', decay=0.9,
                               beta1=0.9,beta2=0.999,
                               batch_size=None, shuffle=True):

        if isinstance(optimizer, str):
            optimizer = [optimizer]
        allowed_optimizers = ['original', 'Adagrad', 'RMSProp','Adam']
        if not set(optimizer).issubset(set(allowed_optimizers)):
            raise KeyError(f'metrics {optimizer} is not supported')
        self.optimizer = optimizer[0]
        if learning_rate is None:
            learning_rate=self.learning_rate
        if max_epochs is None:
            max_epochs=self.max_epochs
        if batch_size is None:
             batch_size=self.batch_size
        self.loss=[]

        data, label = X[:, :2], X[:, 2]
        num_samples, n_features = data.shape
        self.w   = np.zeros(n_features)
        self.b   = 0
        if self.optimizer in ['Adagrad', 'RMSProp']:
            self.G_w = np.zeros(n_features)
            self.G_b = 0
        elif self.optimizer == 'Adam':
            self.m_w = None  # 存储参数w的一阶动量
            self.v_w = None  # 存储参数w的二阶动量
            self.m_b = None  # 存储参数b的一阶动量
            self.v_b = None  # 存储参数b的二阶动量
            self.beta1 = beta1  # 一阶动量的衰减率
            self.beta2 = beta2  # 二阶动量的衰减率
            self.epsilon = 1e-6  # 避免除零错误
            self.t = 0  # 迭代次数

        indices = np.arange(num_samples)

        for epoch in range(max_epochs):
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]
                X_batch = data[batch_indices]
                Y_batch = label[batch_indices]
                y_pred = np.dot(X_batch, self.w)+self.b
                if self.optimizer=='original':
                    gradient_w = (-2 / len(X_batch)) * np.dot(X_batch.T, Y_batch - y_pred)
                    gradient_b = (-2 / len(X_batch)) * np.sum(Y_batch - y_pred)
                    self.w -= learning_rate * gradient_w
                    self.b -= learning_rate * gradient_b
                elif self.optimizer=='Adagrad':
                    epsilon=1e-6
                    gradient_w = (-2 / len(X_batch)) * np.dot(X_batch.T, Y_batch - y_pred)
                    gradient_b = (-2 / len(X_batch)) * np.sum(Y_batch - y_pred)
                    self.G_w += gradient_w ** 2
                    self.G_b += gradient_b ** 2
                    self.w -= (learning_rate / (np.sqrt(self.G_w) + epsilon)) * gradient_w
                    self.b -= (learning_rate / (np.sqrt(self.G_b) + epsilon)) * gradient_b
                elif self.optimizer=='RMSProp':
                    epsilon=1e-6
                    gradient_w = (-2 / len(X_batch)) * np.dot(X_batch.T, Y_batch - y_pred)
                    gradient_b = (-2 / len(X_batch)) * np.sum(Y_batch - y_pred)
                    self.G_w = decay*self.G_w**2 +(1-decay)*gradient_w**2
                    self.G_b = decay*self.G_b**2 +(1-decay)*gradient_b**2
                    self.w -= (learning_rate / (np.sqrt(self.G_w) + epsilon)) * gradient_w
                    self.b -= (learning_rate / (np.sqrt(self.G_b) + epsilon)) * gradient_b
                else:#adam
                    self.t += 1
                    gradient_w = (-2 / len(X_batch)) * np.dot(X_batch.T, Y_batch - y_pred)
                    gradient_b = (-2 / len(X_batch)) * np.sum(Y_batch - y_pred)

                    if self.m_w is None:
                        self.m_w = np.zeros_like(self.w)
                        self.v_w = np.zeros_like(self.w)
                        self.m_b = 0
                        self.v_b = 0

                    # 更新一阶动量和二阶动量
                    self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * gradient_w
                    self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * gradient_w ** 2
                    self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * gradient_b
                    self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * gradient_b ** 2

                    # 纠正一阶动量和二阶动量的偏差
                    m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
                    v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
                    m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
                    v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

                    # 更新参数
                    self.w -= (learning_rate / (np.sqrt(v_w_hat) + self.epsilon)) * m_w_hat
                    self.b -= (learning_rate / (np.sqrt(v_b_hat) + self.epsilon)) * m_b_hat



            loss = np.mean((Y_batch - y_pred) ** 2)
            self.loss.append(loss)


    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("模型尚未训练，请先训练模型")
        X = X[:, :2]
        return np.sign(np.dot(X,self.w)+ self.b)




