import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 只使用前两个特征
y = iris.target

# 创建分类器对象
classifiers = [LogisticRegression(), KNeighborsClassifier(), GaussianNB()]
names = ['Logistic Regression', 'KNN', 'Naive Bayes']

# 计算分类概率
h = .02  # 步长
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, h),
                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, h))

plt.figure(figsize=(12, 6))
for i, classifier in enumerate(classifiers):
    ax = plt.subplot(1, 3, i + 1)
    classifier.fit(X, y)
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest', origin='lower', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Reds, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.colorbar()
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title(names[i])

plt.tight_layout()
plt.show()