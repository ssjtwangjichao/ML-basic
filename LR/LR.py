# -*- coding: utf-8 -*-
# 算法解析： 逻辑回归算法是用来解决分类问题的，其实质却是一种常用的分类模型，主要被用于二分类问题，它将特征空间映射成一种可能性，在LR中，y是一个定性变量{0,1}，
# LR方法主要用于研究某些事发生的概率。
# （回归与分类的区别在于：回归所预测的目标量的取值是连续的（例如房屋的价格）；而分类所预测的目标变量的取值是离散的（例如判断邮件是否为垃圾邮件））
'''
算法步骤（sklearn步骤）：

1.加载数据集，将数据集拆分为训练集和测试集
2.调用模型，训练
'''

import numpy as np
import matplotlib.pyplot as plt

# 使用交叉验证的方法，把数据集分为训练集合测试集
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

#加载数据集，将数据集拆分为训练集和测试集
def load_data():
     data = datasets.load_iris()
     X_train,X_test,y_train,y_test = train_test_split(data.data,data.target, test_size=0.30, random_state=0)

     return X_train,X_test,y_train,y_test

#构造LR函数
def test_logisticRegression(X_train,X_test,y_train,y_test):

    cls = LogisticRegression()
    cls.fit(X_train,y_train)

    print("Coefficients:%s, intercept %s" % (cls.coef_, cls.intercept_))
    print("Residual sum of squares: %.2f" % np.mean((cls.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % cls.score(X_test, y_test))

#执行主函数

if __name__ == '__main__':
    X_train,X_test,y_train,y_test=load_data()
    test_logisticRegression(X_train,X_test,y_train,y_test)

