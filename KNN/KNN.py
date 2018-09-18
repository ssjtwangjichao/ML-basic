# coding=utf-8
# 算法解析： KNN分类算法
'''
算法步骤：
1、计算已知类别数据集中的点与当前点之间的距离；
2、按照距离递增次序排序；
3、选取与当前点距离最小的k个点；
4、确定k个点所在类别的出现频率；
（K用于选择最近邻的数目，K的选择非常敏感。K值越小意味着模型复杂度越高，从而容易产生过拟合；K值越大则 意味着整体的模型变得简单，学习的近似误差会增大，在实际的应用中，一般采用一个比较小的K值，用交叉验证的 方法，选取一个最优的K值。）
5、返回前k个点出现频率最高的类别作为当前点的预测分类；
'''

from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 生成数据
"""
生成60个样本，这60个样本分布在centers中心点周围，cluster_std指明生成点分布的松散程度，x是样本集，，y是类别
"""
centers = [[-2, 2], [2, 2], [0, 4]]
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.6)

# print(X)
# print(y)
plt.figure(figsize=(16, 10), dpi=144)
c = np.array(centers)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange')

# plt.show()
# 训练KNN模型
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# X_sample=[0,2]
# y_sample = clf.predict(X_sample)
# 报错误如下：Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# 方法1
X_sample = [[0, 2]]
y_sample = clf.predict(X_sample)
print('y_sample = ', y_sample)
# 方法二
X_sample2 = [0, 2]
temp = np.array(X_sample2).reshape((1, -1))
print(type(temp))
y_sample2 = clf.predict(temp)
print('y_sample2 = ', y_sample2)
print(type(y_sample2))
# y_sample = clf.predict(np.array(X_sample).reshape((1,-1)))
neighbors = clf.kneighbors(temp, return_distance=False)
print(type(neighbors))

plt.scatter(X_sample2[0], X_sample2[1], marker='x', c=y_sample2[0], s=100, cmap='cool')
print('neightbors num ', neighbors[0])

for i in neighbors[0]:
    plt.plot([X[i][0], X_sample2[0]], [X[i][1], X_sample2[1]], 'k--', linewidth=0.6)

plt.show()

print('done')