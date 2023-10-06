import numpy as np

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
num_features = X.shape[1]
classes = np.unique(y)
num_classes = len(classes)

mean_vectors = []
for c in classes:
    mean_vectors.append(np.mean(X[y == c], axis=0).tolist())
mean_vectors = np.array(mean_vectors)

Sw = np.zeros((num_features, num_features))
for c in classes:
    class_samples = X[y == c]
    class_mean = mean_vectors[classes == c]
    for x in class_samples:
        x_minus_mean = x - class_mean
        Sw += np.outer(x_minus_mean, x_minus_mean)

u = mean_vectors[1:2,:]-mean_vectors[0:1,:]   #u(1) - u(-1)
Sb = (u).T @ (u)
w = np.linalg.inv(Sw) @ u.T
S = (w.T @ (mean_vectors[1:2,:] + mean_vectors[0:1,:]).T)/2
print(w)
print(S)