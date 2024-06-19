import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# 功能选择函数
def feature_selection(dataset):
    for ds_cnt, ds in enumerate(dataset):
        X = ds.data
        y = ds.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        X_indices = np.arange(X.shape[1])

        # 单变量特征选择
        selector = SelectKBest(f_classif, k=4)
        selector.fit(X_train, y_train)
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()

        plt.figure(figsize=(10, 6))
        plt.bar(X_indices - 0.45, scores, width=0.2, label=r'Univariate score ($-Log(p_{value})$)')
        plt.title("Univariate Feature Scores")
        plt.xlabel('Feature Index')
        plt.ylabel('Score')
        plt.legend(loc='upper right')
        st.pyplot(plt)
        plt.clf()

        # SVM 权重
        clf = make_pipeline(MinMaxScaler(), LinearSVC())
        clf.fit(X_train, y_train)
        print('Classification accuracy without selecting features: {:.3f}'.format(clf.score(X_test, y_test)))

        svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
        svm_weights /= svm_weights.sum()

        plt.bar(X_indices - 0.25, svm_weights, width=0.2, label='SVM weight')
        plt.title("SVM Feature Weights")
        plt.xlabel('Feature Index')
        plt.ylabel('Weight')
        plt.legend(loc='upper right')
        st.pyplot(plt)
        plt.clf()

        # 选择特征后的 SVM 权重
        clf_selected = make_pipeline(SelectKBest(f_classif, k=4), MinMaxScaler(), LinearSVC())
        clf_selected.fit(X_train, y_train)
        print('Classification accuracy after univariate feature selection: {:.3f}'.format(clf_selected.score(X_test, y_test)))

        svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
        svm_weights_selected /= svm_weights_selected.sum()

        selected_indices = X_indices[selector.get_support()]

        plt.bar(selected_indices - 0.05, svm_weights_selected, width=0.2, label='SVM weights after selection')
        plt.title("SVM Feature Weights After Selection")
        plt.xlabel('Feature Index')
        plt.ylabel('Weight')
        plt.legend(loc='upper right')
        st.pyplot(plt)
        plt.clf()

        # 决策树特征重要性
        clf_dtree = Pipeline([('preprocessing', MinMaxScaler()), ('classifier', DecisionTreeClassifier())])
        clf_dtree.fit(X_train, y_train)
        feature_weights = clf_dtree.named_steps['classifier'].feature_importances_

        plt.bar(X_indices, feature_weights, width=0.2, label='Decision Tree weights')
        plt.title("Decision Tree Feature Weights")
        plt.xlabel('Feature Index')
        plt.ylabel('Weight')
        plt.legend(loc='upper right')
        st.pyplot(plt)
        plt.clf()

        # 使用 Plotly 显示条形图
        stfig_rate = make_subplots(rows=1, cols=1)
        stfig_rate.add_trace(go.Bar(x=X_indices, y=feature_weights, name='Decision Tree weights'))
        stfig_rate.update_xaxes(title_text='Feature Index', tickfont=dict(size=14))
        stfig_rate.update_yaxes(title_text='Weight', tickfont=dict(size=14))
        stfig_rate.update_layout(title_text=f'{ds.description} Feature Importance Analysis')
        
        st.plotly_chart(stfig_rate)

        print('The suspicious features for defective products are:')
        return stfig_rate, feature_weights

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

# 运行分类器投票预测
feature_selection([dataset])