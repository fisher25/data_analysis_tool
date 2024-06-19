import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

plt.rcParams['font.family'] = 'SimHei'  # 将字体设置为SimHei或其他中文字体

# 功能选择函数
def feature_selection(dataset):
    for ds_cnt, ds in enumerate(dataset):
        X = ds.data
        y = ds.target

        X_indices = np.arange(X.shape[1])

        st.title("检测项辅助诊断工具")
        st.write("## 数据展示")
        st.write("单行数据为某发动机检测装配过程测量数据")
        st.write(pd.DataFrame(X).head())
        st.write(f"标签分布: {np.unique(y)}")

        # 定义分类器字典
        models = {
            "线性SVM": SVC(kernel="linear", C=0.025, random_state=42),
            "决策树": DecisionTreeClassifier(max_depth=5, random_state=42),
            "随机森林": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
            "梯度提升": GradientBoostingClassifier(random_state=42)
        }

        # 使用循环调用分类器并绘制特征重要性
        for model_name, model in models.items():
            clf = Pipeline([('preprocessing', MinMaxScaler()), ('classifier', model)])
            clf.fit(X, y)

            # 获取特征重要性
            if hasattr(clf.named_steps['classifier'], 'coef_'):
                feature_importances = np.abs(clf.named_steps['classifier'].coef_).sum(axis=0)
            elif hasattr(clf.named_steps['classifier'], 'feature_importances_'):
                feature_importances = clf.named_steps['classifier'].feature_importances_
            else:
                feature_importances = np.zeros(X.shape[1])

            feature_importances /= feature_importances.sum()

            st.write(f"## {model_name}")
            st.write(f"-  基于{model_name}模型的特征重要性")

            plt.figure()
            plt.bar(X_indices, feature_importances)
            plt.xlabel('特征索引 [-]')
            plt.ylabel(f'{model_name} 权重')
            plt.title(f'{model_name} 特征重要性')
            st.pyplot(plt)

        return

# 示例数据集
class Dataset:
    def __init__(self, data, target, description):
        self.data = data
        self.target = target
        self.description = description

# 创建一个示例数据集
np.random.seed(0)
data = np.random.rand(100, 10)
target = (data[:, 0] + data[:, 1] > 1).astype(int)

dataset = Dataset(data, target, "示例数据集")

# 运行特征选择函数
feature_selection([dataset])