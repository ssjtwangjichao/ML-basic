# coding=utf-8
#算法解析： K-mean算法聚类算法，将数据集根据距离分为K个数据类
'''
算法步骤：
1. 选定 K 个中心 \mu_k 的初值。这个过程通常是针对具体的问题有一些启发式的选取方法，或者大多数情况下采用随机选取的办法。因为前面说过 k-means 并不能保证全局最优，而是否能收敛到全局最优解其实和初值的选取有很大的关系，所以有时候我们会多次选取初值跑 k-means ，并取其中最好的一次结果。
2. 将每个数据点归类到离它最近的那个中心点所代表的 cluster 中。
3. 用公式 \mu_k = \frac{1}{N_k}\sum_{j\in\text{cluster}_k}x_j 计算出每个 cluster 的新的中心点。
4. 重复第二步，一直到迭代了最大的步数或者前后的 J 的值相差小于一个阈值为止。
'''

from numpy import *

#公共变量定义

# 加载数据
def loadDataSet(fileName):
    print('----data begin to load----')
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        #print(line)
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    print('----data end to load----')
    return 0


# 计算欧几里得距离
def distEclud(vecA, vecB):
     return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
     n = shape(dataSet)[1]
     centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
     for j in range(n):
         minJ = min(dataSet[:,j])
         maxJ = max(dataSet[:,j])
         rangeJ = float(maxJ - minJ)
         centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
     return centroids



def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
     m = shape(dataSet)[0]
     clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
     # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
     centroids = createCent(dataSet, k)
     clusterChanged = True   # 用来判断聚类是否已经收敛
     while clusterChanged:
         clusterChanged = False;
         for i in range(m):  # 把每一个数据点划分到离它最近的中心点
             minDist = inf; minIndex = -1;
             for j in range(k):
                 distJI = distMeans(centroids[j,:], dataSet[i,:])
                 if distJI < minDist:
                     minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
             if clusterAssment[i,0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
             clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
         print(centroids)
         for cent in range(k):   # 重新计算中心点
             ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
             centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
     return centroids, clusterAssment


if __name__ == '__main__':
    ##unittest.main()
    datMat = mat(loadDataSet('TestData.txt'))
    myCentroids, clustAssing = kMeans(datMat, 4)
    print(myCentroids)
    print(clustAssing)
