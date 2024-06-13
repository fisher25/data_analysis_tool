import os
from datetime import date,datetime, timedelta
import numpy as np
import pandas as pd
import xlrd
import csv
import struct
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import random

from scipy.stats import weibull_min

from stat import FILE_ATTRIBUTE_ARCHIVE
# from bokeh.io import output_file, show
# from bokeh.models import ColumnDataSource
# from bokeh.models.widgets import DataTable, StringEditor, NumberEditor, TableColumn
# from bokeh.layouts import column, row
# from bokeh.plotting import figure


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import datasets
from sklearn import model_selection
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, max_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# import seaborn as sns
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# from temp_sklearn import FONT_SIZE

plt.rcParams['font.family'] = 'SimHei'  # 将字体设置为SimHei或其他中文字体

# 定义全局常量文件路径等          
FILE_2022_BENCH_TEST = './data/2022_benchtest.xlsx'
FILE_2022_MAINTEANCE = './data/2022mainteance_report.xlsx'
FILE_FAKE_BENCH_CAN = './data/fake_bench_CANdata.xlsx'

FILE_2022_BENCH_TEST_FORMAT = './data/2022_benchtest_format.xlsx'
FILE_FAKE_MAINTEANCE = './data/fake_mainteance_data.xlsx'
FILE_PART_LIFE =  './data/rul_pump_data_from_kaggle.xlsx'  #               './data/fakedata_waterpump_life.xlsx'
FILE_WATERPUMP_MT = './data/fakedata_waterpump_mainteance.xlsx'



class CustomDataset():
    # 输入数据定义接口, numpy转对象
    def __init__(self, data, target, feature_names=None, target_names=None, description=None,file_path=None):
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_names = target_names
        self.description = description
        self.file_path = file_path
        
        self.create_dataframe()
        self.save_dataframe_to_excel()


    def create_dataframe(self):

        data_df = pd.DataFrame(self.data, columns=self.feature_names)
        target_df = pd.DataFrame(self.target, columns= [self.target_names])
        self.dataframe = pd.concat([data_df, target_df], axis=1)

    def save_dataframe_to_excel(self):

        if not os.path.exists(self.file_path):
            self.dataframe.to_excel(self.file_path, index=False)
            print(f'DataFrame saved to {self.file_path}')
        else:
            print(f'File {self.file_path} already exists. Skipping save.')
            
    
class LoadDataset():
    # def __init__(self, file_path, description=None):

        # df = pd.read_excel(file_path)
        # # 最后一列是目标列，其他是特征列
        # self.data = df.iloc[:, :-1].values
        # self.target = df.iloc[:, -1].values
        # self.feature_names = df.columns[:-1].tolist()
        # self.target_names = df.columns[-1]
        # self.description = f'Data loaded from {file_path}'
        
    def __init__(self, file_path=None, description=None):
        self.data = None
        self.target = None
        self.feature_names = None
        self.target_names = None
        self.description = description
        
        if file_path is not None:
            self.load_from_file(file_path)
            
    def load_from_file(self, file_path):
        df = pd.read_excel(file_path)
        # self.data = df.iloc[:, :-1].values
        # self.target = df.iloc[:, -1].values
        random_list = [random.randint(1, 150000) for _ in range(100)]
        self.data = df.iloc[random_list, :-1].values
        self.target = df.iloc[random_list, -1].values
        self.feature_names = df.columns[:-1].tolist()
        self.target_names = df.columns[-1]
        self.description = f'从文件路径 {file_path} 加载的数据'
        
    def load_from_dataframe(self, dataframe):
        self.data = dataframe.iloc[:, :-1].values
        self.target = dataframe.iloc[:, -1].values
        self.feature_names = dataframe.columns[:-1].tolist()
        self.target_names = dataframe.columns[-1]
        self.description = '从DataFrame加载的数据'


class Data_Input():
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def read_xls_2022_mainteance():
        # 读取准能2022维修数据excel，
        # 输入excel，
        # 输出data对象；输出标准excel用来导入数据库
        
        filename =FILE_2022_MAINTEANCE
        
        # 获取当前工作目录
        current_dir = os.getcwd()
        print("当前工作目录:", current_dir)
        
        data = pd.read_excel(filename)
        xls = pd.ExcelFile(filename)
        sheet_names = xls.sheet_names
        print(sheet_names)
        
        if len(sheet_names) >= 1:
            raw_data = {}
            test_rets = np.zeros((1,1))
            features = np.zeros((1,16+96+64+2)) 
            for sheetname in sheet_names:
                data = []
                engine_data = []
                data = pd.read_excel(filename,sheet_name=sheetname,header=None)
                
                engine_SN = np.array(data.iloc[1,7]) # 0,1:9发动机序列号数据
                rod_diameter = np.array(data.iloc[14:16,3:11].values).reshape(1,16) # 连杆直径数据
                feature1_name = ['连杆直径数据' + str(i) for i in range(len(rod_diameter))]
                
                cylinder_round = np.array(data.iloc[20:32,5:13].values).reshape(1,96) # 缸套圆度数据
                feature2_name = ['缸套圆度数据' + str(i) for i in range(len(cylinder_round))]
                
                cylinder_height = np.array(data.iloc[35:43,4:12].values).reshape(1,64) # 缸套高度数据
                feature3_name = ['缸套高度数据' + str(i) for i in range(len(cylinder_height))]
                
                flywheel_vet = np.array(data.iloc[46,3]).reshape(1,1) # 飞轮径向总指示跳动
                feature4_name = ['飞轮径向总指示跳动']
                flywheel_data = np.array(data.iloc[46,6]).reshape(1,1) # 飞轮端面总指示跳动
                feature5_name = ['飞轮径向总指示跳动']
                
                test_ret = np.array(data.iloc[1,10]).reshape(1,1) # 热试验结果
                feature = np.concatenate((rod_diameter,cylinder_round,cylinder_height,flywheel_vet,flywheel_data),axis=1) # 数组和数组拼接,按列增长方向拼接(水平拼接) 
                test_rets = np.vstack((test_rets,test_ret))
                features = np.vstack((features, feature))
                feature_names = feature1_name + feature2_name + feature3_name + feature4_name + feature5_name
        
        # 去掉首0，缺省值补充高频值
        test_rets = np.delete(test_rets, 0, axis=0)
        features = np.delete(features, 0, axis=0)   
        filler = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        filler.fit(features)
        features = filler.transform(features)
        filler.fit(test_rets)
        test_rets = filler.transform(test_rets)
        
        # 创建数据集对象
        target_names = ['合格']
        description = '准能2022年维修数据'
        mainteance_2022_dataset = CustomDataset(data=features, target=test_rets, feature_names=feature_names,
                                    target_names=target_names, description=description)
        return [mainteance_2022_dataset]

    @staticmethod
    def read_xls_2022_benchtest():
        # 输入：历史数据 2022年热试试验数据
        # 输出：标准格式 dataset 
    
        filename =FILE_2022_BENCH_TEST
        # 获取当前工作目录
        current_dir = os.getcwd()
        print("当前工作目录:", current_dir)
        
        data = pd.read_excel(filename)
        xls = pd.ExcelFile(filename)
        sheet_names = xls.sheet_names
        print(sheet_names)
        if len(sheet_names) >= 1:

            test_rets = np.zeros((1,1))
            features = np.zeros((1,91)) 
            for sheetname in sheet_names:
                data = []
 
                data = pd.read_excel(filename,sheet_name=sheetname,header=None)
                feature = np.array(data.iloc[5:12,3:16].values).reshape(1,91) # 连杆直径数据
                test_ret = np.array(data.iloc[1,5]).reshape(1,1) # 热试验结果
                test_rets = np.vstack((test_rets,test_ret))
                features = np.vstack((features, feature))
        
        data = pd.read_excel(filename,sheet_name='Sheet9',header=None) 
        print('column',data.columns)
        
        test_point = data.iloc[5:12, 2].values
        test_item = data.iloc[3, 3:16].values
        feature_names = [f"{a}\n{b}" for a in test_point for b in test_item]
        feature_names = [s.replace(" ", "") for s in feature_names]

        test_rets = np.delete(test_rets, 0, axis=0)
        features = np.delete(features, 0, axis=0)   
        filler = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        filler.fit(features)
        features = filler.transform(features)
        filler.fit(test_rets)
        test_rets = filler.transform(test_rets)
        
        # 创建数据集对象
        target_names = ['合格']
        description = '准能2022热试数据'
        mainteance_2022_dataset = CustomDataset(data=features, target=test_rets, feature_names=feature_names,
                                    target_names=target_names, description=description, file_path=FILE_2022_BENCH_TEST_FORMAT)
        return [mainteance_2022_dataset]

    @staticmethod
    def read_xls_life_data():
        # 输入：寿命excel数据
        # 输出：life_dataset
        
        df = pd.read_excel(FILE_PART_LIFE,sheet_name='Sheet1',header=0)
        
        features = df.iloc[:,0:2].values
        life_rets = df.iloc[:,-1].values
        feature_names = df.columns[0:2]
        # 创建数据集对象
        target_names = df.columns[-1]
        description = '水泵流量测试数据和寿命'
        dataset = CustomDataset(data=features, target=life_rets, feature_names=feature_names,
                                    target_names=target_names, description=description)
        print(vars(dataset))
        
        return [dataset]
    
    @staticmethod
    def read_xls_waterpump_mt():
        
        df = pd.read_excel(FILE_WATERPUMP_MT,sheet_name='Sheet1',header=0)
        features = df.iloc[1:,1:3].values
        test_rets = df.iloc[1:,3].values
        feature_names = df.columns[1:3]
        # 创建数据集对象
        target_names = ['合格', '不合格']
        description = '水泵流量测试数据'
        dataset = CustomDataset(data=features, target=test_rets, feature_names=feature_names,
                                    target_names=target_names, description=description)
        print(vars(dataset))
        
        return [dataset]
    
    
class Data_visual():
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def plot_data(datasets):
        dataset = datasets[0]
        M, N = dataset.data.shape
        N = min(N,5)
        data = dataset.data
        names = dataset.feature_names[0:N]

        fig, axs = plt.subplots(3, N, figsize=(10, 6))
        # fig.subplots_adjust(hspace=0.4)
        
        colors = ['#D81B60', '#0188FF', '#FFC107',
            '#B7A2FF', '#000000', '#2EC5AC']

        # 分布图
        for i in range(N):
            axs[0, i].hist(data[:, i], color=colors[1])
            axs[0, i].set_title(f'分布图 {names[i]}', fontsize=10)

        # 散点图
        for i in range(N):
            axs[1, i].scatter(np.arange(M), data[:, i], color=colors[2])
            axs[1, i].set_title(f'散点图 {names[i]}', fontsize=10)

        # 箱体图
        for i in range(N):
            axs[2, i].boxplot(data[:, i], vert=False, labels=[''], patch_artist=True)
            axs[2, i].set_title(f'箱体图 {names[i]}', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        plt.suptitle(dataset.description, fontsize=14, y=0.98)
        plt.show()
        
        return [fig ,N]

    def stats_analysis(self, dataset):
        # 绘制数据集分布图并展示统计信息
        np.random.seed(0)
        # 示例用法
        n = 100  # 样本数
        m = 5    # 特征数
        data = np.random.randn(n, m)
        for ds_cnt, ds in enumerate(dataset):
            # dataset, labels = ds
            dataset = ds.data
            labels = ds.target
            
            # dataset.head()
            # dataset.describe()
            # dataset.info()
            # dataset.nunique()
            # dataset.skew()
            # dataset.kurtosis()
        
        fig, axs = plt.subplots(3, dataset.shape[1])
        # 输出特征的描述统计信息
        for i in range(dataset.shape[1]):
            feature = dataset[:, i]
            print(f'特征 {i+1}:')
            print(f'最大值: {np.max(feature)}')
            print(f'最小值: {np.min(feature)}')
            print(f'中位数: {np.median(feature)}')
            print(f'平均数: {np.mean(feature)}')
            
            axs[0, i].hist(feature)
            axs[0, i].set_xlabel('X')
            axs[0, i].set_ylabel('Y')
            axs[0, i].set_title(f'特征 {i+1}:')
            
            axs[1, i].hist(feature,bins=8,density=False,color='g',edgecolor='k',alpha=0.75)
            axs[1, i].set_xlabel('X')
            axs[1, i].set_ylabel('Y')
            
            axs[2, i].boxplot(feature)
            axs[2, i].set_xlabel('X')
            axs[2, i].set_ylabel('Y')
                       
            # plt.hist(feature,bins=8,density=False,color='g',edgecolor='k',alpha=0.75)
            # plt.xlabel('检测项目样本')
            # plt.ylabel('数据分布')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        for i in range(data.shape[1]):
            plt.scatter(np.arange(data.shape[0]), data[:,i], label=f'Feature {i+1}')
        
        shape, loc, scale = weibull_min.fit(feature, floc=0)
        sorted_data = np.sort(feature)
        y = np.arange(1, len(feature) + 1) / len(feature)
        plt.plot(sorted_data, y, 'o')
        
        view = plt.boxplot(feature, patch_artist=True, boxprops={'facecolor': 'cyan', 'linewidth': 0.8,'edgecolor': 'red'})
        # plt.show()
        
        stat_mean = np.mean(feature)
        stat_median = np.median(feature)
        stat_std = np.std(feature)
        stat_min = np.min(feature)
        stat_max = np.max(feature)
        print('mean',stat_mean,'median',stat_median,'std',stat_std,'min',stat_min,'max',stat_max)
        
        # 创建数据源
        data_table_data = {
            'name':['part01'],
            'mean': [stat_mean],
            'median': [stat_median],
            'std': [stat_std],
            'min': [stat_min],
            'max': [stat_max],
        }
        scatter_plot_data = {
            'X': feature
            # 'Y': [4, 5, 6]
        }
        data_table_source = ColumnDataSource(data_table_data)
        scatter_plot_source = ColumnDataSource(scatter_plot_data)

        # 创建表格列定义
        columns = [
            TableColumn(field='Name', title='Name', editor=StringEditor()),
            TableColumn(field='mean', title='mean', editor=NumberEditor()),
            TableColumn(field='median', title='median', editor=NumberEditor()),
            TableColumn(field='std', title='std', editor=NumberEditor()),
            TableColumn(field='min', title='min', editor=NumberEditor()),
            TableColumn(field='max', title='max', editor=NumberEditor()),
        ]

        # 创建数据表
        data_table = DataTable(source=data_table_source, columns=columns, editable=True, index_position=-1, index_header='Index')

        # 创建散点图
        scatter_plot = figure(title='Scatter Plot', width=400, height=400)
        scatter_plot.circle('X',  source=scatter_plot_source, size=10)

        # 设置输出文件
        output_file('table_with_scatter_plot.html')
        show(row(data_table, scatter_plot))        
        
class Life_analysis():    

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def life_weibull(datasets):
        
        # 输入：产品寿命一维数组
        # 输出：预期寿命（寿命中位数）；寿命分布图；寿命箱体图；寿命威布尔拟合图；寿命关键参数（最小最大）
        
        for ds_cnt, ds in enumerate(datasets):
            # preprocess dataset, split into training and test part
            # X, y = ds
            data = ds.data
            life_dataset = ds.target
        
        print(life_dataset)
        plt.scatter(np.arange(life_dataset.shape[0]), life_dataset)
        plt.xlabel('水泵样本')
        plt.ylabel('水泵样本寿命（天）')
        plt.title('水泵样本寿命散点图图')
        plt.show()
        
        shape, loc, scale = weibull_min.fit(life_dataset, floc=0)
        sorted_data = np.sort(life_dataset)
        y = np.arange(1, len(life_dataset) + 1) / len(life_dataset)
        f1 = plt.plot(sorted_data, y, 'o')
        plt.xlabel('水泵样本寿命（天）')
        plt.ylabel('寿命累积概率')
        plt.title('水泵样本寿命累积概率图')
        plt.grid()
        # plt.show()

        # 绘制威布尔分布曲线
        x = np.linspace(sorted_data.min(), sorted_data.max(), 100)
        cdf = weibull_min.cdf(x, shape, loc, scale)
        plt.plot(x, cdf, 'r-', label=f'Weibull Distribution (shape={round(shape, 2)}, scale={round(scale, 2)})')

        #  :param reliability_requirement: 所需的可靠性阈值
        # :param beta: 威布尔分布的形状参数
        # :param eta: 威布尔分布的比例参数
        # maintenance_interval = eta * (-math.log(reliability_requirement)) ** (1 / beta)
        # return maintenance_interval
        
        plt.xlabel('水泵样本寿命（天）')
        plt.ylabel('寿命累积概率')#Cumulative Distribution Function
        plt.title('水泵寿命的韦伯累积概率分布')
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.hist(life_dataset,bins=8,density=True,color='g',edgecolor='k',alpha=0.75)
        plt.xlabel('水泵样本寿命（天）')
        plt.ylabel('数据分布比例')
        plt.show()
        
        plt.boxplot(life_dataset, patch_artist=True, boxprops={'facecolor': 'cyan', 'linewidth': 0.8,'edgecolor': 'red'})
        plt.xlabel('项目样本')
        plt.ylabel('数据分布')
        plt.show()
        
        print(f'水泵预期寿命为: {round(np.mean(life_dataset),0)}')
        
        # 指定时间窗口
        start_time = 0
        end_time = 140

        # 计算指定时间窗口的概率
        probability = weibull_min.cdf(end_time, shape, loc, scale) - weibull_min.cdf(start_time, shape, loc, scale)
        print("指定时间窗口的概率：", probability)
        
        service_times = [100,200,300,400,500,600,700]
        time_period = 90
        
        return Life_analysis.calculate_failure_count(service_times, time_period, shape, scale)
    
    @staticmethod
    def calculate_failure_count(service_times, time_period, shape, scale):
        # 计算未来时间窗口的失效个数
        # :param time_period: 未来时间窗口
        # :param service_times:  产品现在的服役时长列表
        # :return: DataFrame, 未来时间窗口的失效个数表格
    
        total_failure_count = 0
        dist = weibull_min(shape, scale=scale)

        for service_time in service_times:
            service_time_next = time_period + service_time
            failure_count = (dist.sf(service_time) - dist.sf(service_time_next))/dist.sf(service_time)
            
            total_failure_count += failure_count

        total_failure_count = int(total_failure_count)

        df = pd.DataFrame({
            '时间窗口（天）': time_period,
            '失效产品个数': total_failure_count
        }, index=[0])
        print(df)
        
        return df
        

class Fault_diagnose():
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def PCA_analysis(datasets):
    # 执行分类分析  
        for ds_cnt, ds in enumerate(datasets):
            # features, labels = ds
            features = ds.data
            labels = ds.target
            
            X_train, X_test, y_train, y_test = model_selection.train_test_split(features,labels,random_state=0)
            svm_method = svm.LinearSVC(C=100)
            svm_method.fit(X_train,y_train)
            print('score',svm_method.score(X_test,y_test))
            
            pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
            m1_score = pipe.fit(X_train, y_train).score(X_test, y_test)
            m2_score = pipe.set_params(svc__C=100).fit(X_train, y_train).score(X_test, y_test)
            print('m1_score:',m1_score)
            print('m2_score:',m2_score)
            # 假设数据位于"feature"列,标签位于"label"列
            
            _, ax = plt.subplots()
            scatter = ax.scatter(features[:, 0], features[:, 1], c=labels)
            # ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
            _ = ax.legend(
                scatter.legend_elements()[0], labels, loc="lower right", title="Classes"
            )
            plt.show()
            
            fig = plt.figure(1, figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

            X_reduced = PCA(n_components=3).fit_transform(features)
            ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                X_reduced[:, 2],
                labels,
                s=40,
            )

            ax.set_title("First three PCA dimensions")
            ax.set_xlabel("1st Eigenvector")
            ax.xaxis.set_ticklabels([])
            ax.set_ylabel("2nd Eigenvector")
            ax.yaxis.set_ticklabels([])
            ax.set_zlabel("3rd Eigenvector")
            ax.zaxis.set_ticklabels([])

            plt.show()
    
    @staticmethod
    def feature_selection(dataset):

        # #############################################################################
        # Import some data to play with

        # The iris dataset
        
        for ds_cnt, ds in enumerate(dataset):
            # X, y = ds
            X = ds.data
            y = ds.target

            # Some noisy data not correlated
            E = np.random.RandomState(42).uniform(0, 0.1, size=(X.shape[0], 20))

            # Add the noisy data to the informative features
            X = np.hstack((X, E))

            # Split dataset to select feature and evaluate the classifier
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y, stratify=y, random_state=0
            )

            plt.figure(1)
            plt.clf()

            X_indices = np.arange(X.shape[-1])

            # #############################################################################
            # Univariate feature selection with F-test for feature scoring
            # We use the default selection function to select the four
            # most significant features

            selector = SelectKBest(f_classif, k=4)
            selector.fit(X_train, y_train)
            scores = -np.log10(selector.pvalues_)
            scores /= scores.max()
            plt.bar(X_indices - .45, scores, width=.2,
                    label=r'Univariate score ($-Log(p_{value})$)')
            plt.show()

            # #############################################################################
            # Compare to the weights of an SVM
            clf = make_pipeline(MinMaxScaler(), LinearSVC())
            clf.fit(X_train, y_train)
            print('Classification accuracy without selecting features: {:.3f}'
                .format(clf.score(X_test, y_test)))

            svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
            svm_weights /= svm_weights.sum()

            plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight')
            plt.title("可疑检测项目排查")
            plt.xlabel('检测项目 [-]')
            plt.yticks(())
            plt.ylabel('检测项目权重')
            plt.axis('tight')
            plt.legend(loc='upper right')
            plt.show()
            
            clf_selected = make_pipeline(
                    SelectKBest(f_classif, k=4), MinMaxScaler(), LinearSVC()
            )
            clf_selected.fit(X_train, y_train)
            print('Classification accuracy after univariate feature selection: {:.3f}'
                .format(clf_selected.score(X_test, y_test)))

            svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
            svm_weights_selected /= svm_weights_selected.sum()

            plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
                    width=.2, label='SVM weights after selection')

            plt.title("可疑检测项目排查")
            plt.xlabel('检测项目 [-]')
            plt.yticks(())
            plt.ylabel('检测项目权重')
            plt.axis('tight')
            plt.legend(loc='upper right')
            plt.show()
                        
            # # 创建并训练决策树分类器
            clf_dtree = Pipeline([('preprocessing', MinMaxScaler()),  ('classifier', DecisionTreeClassifier()) ])
            clf_dtree.fit(X_train, y_train)
            clf_dtree_model = clf_dtree.named_steps['classifier']
            feature_weights = clf_dtree_model.feature_importances_
            plt.bar(X_indices, feature_weights,width=.2, label='decision tree weights')
            plt.title("可疑检测项目排查")
            plt.xlabel('检测项目 [-]')
            plt.yticks(())
            plt.ylabel('检测项目权重')
            plt.axis('tight')
            plt.legend(loc='upper right')
            plt.show()          
            
            stfig_rate = make_subplots(rows=1, cols=1)
            stfig_rate.add_trace(go.Bar(x=X_indices,y=feature_weights))
            stfig_rate.update_xaxes(title_text='检测项目 [-]', tickfont=dict(size=14))
            stfig_rate.update_yaxes(title_text='检测项目权重', tickfont=dict(size=14))
            stfig_rate.update_layout(title_text=f'{ds.description}可疑检测项目排查') 
            
            print('故障产品疑似存在问题的检测项目为','')
            return [stfig_rate , feature_weights]   
  
        
class Classifier():
    
    def __init__(self) -> None:
        pass
    
    
    @staticmethod
    def classifier_comparison(datasets):
    # 分类器比较
    # 输入 数据集
    # 输出 历史数据下各分类器的准确率；各分类器对当前测试数据的分类；投票结果；(数据和图像)
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
            ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, random_state=42),
            SVC(gamma=2, C=1, random_state=42),
            GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            AdaBoostClassifier(random_state=42),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            ]

        figure = plt.figure(figsize=(27, 9))
        i = 1
        # iterate over datasets
        for ds_cnt, ds in enumerate(datasets):
            # preprocess dataset, split into training and test part
            # X, y = ds
            X = ds.data
            y = ds.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42
            )

            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            if ds_cnt == 0:
                ax.set_title("Input data")
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
            # Plot the testing points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

                clf = make_pipeline(StandardScaler(), clf)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                # DecisionBoundaryDisplay.from_estimator(
                #     clf, X[:,0:2], cmap=cm, alpha=0.8, ax=ax, eps=0.5
                # )
                ax.scatter(
                    X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
                )
                # Plot the testing points
                ax.scatter(
                    X_test[:, 0],
                    X_test[:, 1],
                    c=y_test,
                    cmap=cm_bright,
                    edgecolors="k",
                    alpha=0.6,
                )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks(())
                ax.set_yticks(())
                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(
                    x_max - 0.3,
                    y_min + 0.3,
                    ("%.2f" % score).lstrip("0"),
                    size=15,
                    horizontalalignment="right",
                )
                i += 1

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def parameter_optimize(datasets):
        
        for ds_cnt, ds in enumerate(datasets):
            X = ds.data
            y = ds.target

            # 分割数据集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # 尝试不同的K值
            k_values = range(1, 26)
            accuracies = []

            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))

            # 绘制K值和准确率的关系图
            plt.plot(k_values, accuracies)

            plt.xlabel('K近邻算法 K参数值')
            plt.ylabel('测试准确率')
            plt.title('K近邻算法关键参数优化')
            plt.show()

    @staticmethod
    def classifier_vote_pre(datasets):
    # 分类器比较
    # 输入 数据集; 最后一行样本作为需要检测内容
    # 输出 历史数据下各分类器的准确率；各分类器对当前测试数据的分类；投票结果；(数据和图像)
    
        names = [
            "K近邻",#"Nearest Neighbors",
            "线性SVM",#"Linear SVM", svm_classifier.coef_来获取特征权重
            "径向SVM",#"RBF SVM",
            "高斯过程",#"Gaussian Process",
            "决策树",#"Decision Tree",tree_classifier.feature_importances_来获取特征权重
            "随机森林",#"Random Forest",feature_importances_属性来获取特征权重
            "神经网络",#"Neural Net",
            "增强学习",#"AdaBoost",feature_importances_属性来获取特征权重
            "贝叶斯",#"Naive Bayes",
            "二次判别",#"QDA",feature_importances_属性来获取特征权重
            ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, random_state=42),
            SVC(gamma=2, C=1, random_state=42),
            GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            AdaBoostClassifier(random_state=42),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            ]

        figure = plt.figure(figsize=(27, 9))
        i = 1
        # iterate over datasets
        for ds_cnt, ds in enumerate(datasets):
            # preprocess dataset, split into training and test part
            # X, y = ds
            X = ds.data
            y = ds.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42
            )

            i += 1
            # iterate over classifiers
            j=0
            
            y_pred = np.zeros(len(classifiers))
            score = np.zeros(len(classifiers))
            for name, clf in zip(names, classifiers):
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

                clf = make_pipeline(StandardScaler(), clf)
                clf.fit(X_train, y_train)
                score[j] = clf.score(X_test, y_test)
                
                # 使用分类器进行预测
                y_pred[j] = clf.predict([X[-1,:]])
                i += 1
                j += 1
            
            # 统计投票结果
            class_counts = np.bincount(y_pred.astype(np.int64))
            labels = names
            values = score * 100
            # 创建柱状图
            fig, ax = plt.subplots(figsize=(27, 9))
            ax.bar(labels, values,color='g',edgecolor='k',alpha=0.75)
            plt.xticks(fontsize=14)
            ax.set_xlabel('')
            # plt.xticks(rotation=45)
            ax.set_ylabel('测试准确率 %',fontsize =14)
   
            plt.subplots_adjust(top=0.8)
            plt.title(f'{ds.description}分类测试结果', fontsize=14, y=0.98, pad = 20)
            plt.show()
            


            stfig_his = make_subplots(rows=1, cols=1)

            stfig_his.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    # marker_color='green',  # 杆颜色
                    marker_line_color='black',  # 杆边缘颜色
                    marker_line_width=1.5,  # 杆边缘宽度
                    opacity=0.75  # 透明度
                )
            )

            stfig_his.update_xaxes(title_text='', tickfont=dict(size=14))
            stfig_his.update_yaxes(title_text='测试准确率 %', tickfont=dict(size=14))

            stfig_his.update_layout(
                title_text=f'{ds.description}分类测试结果',
                title_font_size=14,
                title_y=0.98,
                title_pad=dict(t=20),
                autosize=False,
                width=27*37.7952755906,  # 转换 inches 到 px
                height=9*37.7952755906,   # 转换 inches 到 px
                margin=dict(t=100)  # 调整上边距以适应标题
            )
            
            fig, ax = plt.subplots(figsize=(27, 9))
            ax.bar(labels, 1-y_pred,color='g',edgecolor='k',alpha=0.75)

            ax.set_title('各分类器分析结果')
            ax.set_xlabel('分类器')
            # plt.xticks(rotation=45)
            ax.set_ylabel('分类器分类结果')
            plt.show()          
                
            stfig_ret = make_subplots(rows=1, cols=1)
            stfig_ret.add_trace(go.Bar(x=labels,y=1-y_pred,))
            stfig_ret.update_xaxes(title_text='分类器', tickfont=dict(size=14))
            stfig_ret.update_yaxes(title_text='分类器分类结果', tickfont=dict(size=14))
            stfig_ret.update_layout(title_text=f'{ds.description}各分类器分析结果',)
            
            most_common_class = np.argmax(class_counts)
            if  most_common_class ==0:
                print('多分类器投票结果：产品合格')
            else:
                print('多分类器投票结果：产品不合格')
            
            
            return [stfig_ret, stfig_his, most_common_class]
                
                

    # ...[其他导入和Classifier类定义]...

    @staticmethod
    def classifier_vote(datasets):
        
        names = [
            "K近邻",#"Nearest Neighbors",
            "线性SVM",#"Linear SVM", svm_classifier.coef_来获取特征权重
            "径向SVM",#"RBF SVM",
            "高斯过程",#"Gaussian Process",
            "决策树",#"Decision Tree",tree_classifier.feature_importances_来获取特征权重
            "随机森林",#"Random Forest",feature_importances_属性来获取特征权重
            "神经网络",#"Neural Net",
            "增强学习",#"AdaBoost",feature_importances_属性来获取特征权重
            "贝叶斯",#"Naive Bayes",
            "二次判别",#"QDA",feature_importances_属性来获取特征权重
            ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, random_state=42),
            SVC(gamma=2, C=1, random_state=42),
            GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            AdaBoostClassifier(random_state=42),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            ]

        figure = plt.figure(figsize=(27, 9))
        i = 1

        # 存储分类器的分类报告
        class_reports = {}
        # 存储每个分类器的混淆矩阵
        conf_matrices = {}

        # iterate over datasets
        for ds_cnt, ds in enumerate(datasets):
            X = ds.data
            y = ds.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42
            )

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            i += 1
            # iterate over classifiers
            j=0
            y_pred = np.zeros(len(classifiers))
            score = np.zeros(len(classifiers))

            # 这里保存每个分类器的分类报告和混淆矩阵
            dataset_reports = {}
            dataset_conf_matrices = {}

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                clf = make_pipeline(StandardScaler(), clf)
                clf.fit(X_train, y_train)
                y_test_pred = clf.predict(X_test)
                y_last_sample_pred = clf.predict([X[-1,:]])

                # 计算并存储每个分类器的分类报告和混淆矩阵
                dataset_reports[name] = classification_report(y_test, y_test_pred, output_dict=True)
                dataset_conf_matrices[name] = confusion_matrix(y_test, y_test_pred)

                # ...[其他的代码，如绘制数据集和分类器的分界等]...

            # 将所有分类报告和混淆矩阵加入全局字典
            class_reports[ds.description] = dataset_reports
            conf_matrices[ds.description] = dataset_conf_matrices

            # ...[绘制投票结果的柱状图等代码]...

            # 绘制每个分类器的混淆矩阵热图
            for name, matrix in dataset_conf_matrices.items():
                plt.figure(figsize=(10, 7))
                sns.heatmap(matrix, annot=True, fmt='g')
                plt.title(f'Confusion Matrix for {name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.show()

        # 汇总所有数据集的分类报告
        all_reports = []
        for ds_desc, reports in class_reports.items():
            for name, report in reports.items():
                df_report = pd.DataFrame(report).transpose()
                df_report['dataset'] = ds_desc
                df_report['classifier'] = name
                df_report = pd.DataFrame(report).transpose()
                all_reports.append(df_report)
        
        summary_report = pd.concat(all_reports, axis=0)
        file_path = os.path.join('data', 'summary_report.csv')
        summary_report.to_csv(file_path, index=False)
        
        print(summary_report)

        # 汇总所有数据集的混淆矩阵热图
        num_classifiers = len(names)
        num_datasets = len(datasets)
        fig, axes = plt.subplots(num_datasets, num_classifiers, figsize=(7*num_classifiers, 3*num_datasets))
        for ds_idx, (ds_desc, matrices) in enumerate(conf_matrices.items()):
            for clf_idx, (name, matrix) in enumerate(matrices.items()):
                ax = axes[ds_idx][clf_idx] if num_datasets > 1 else axes[clf_idx]
                    
                ax.set_title(f'{name}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        
class Regressor():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def predict(datasets):   
        
        # 初始化回归器字典
        regressors = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'SVR': SVR(kernel='linear'),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            # 'MLPRegressor': MLPRegressor(max_iter=1000)
        }

        for ds_cnt, ds in enumerate(datasets):
            X = ds.data
            y = ds.target
            
            # 数据特征         
            try:
                print(ds.data)
                df = pd.DataFrame(ds.data)
            except Exception as e:
                print(f"创建 DataFrame 时发生错误：{e}")
    
            desc_report = df.describe()
            
            # 分割数据集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            
            # Normalize data using MinMaxScaler
            # scaler = MinMaxScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.transform(X_test)

            # # Perform PCA for dimensionality reduction
            # pca = PCA(n_components=0.95)  # Keep 95% of variance
            # X_train_pca = pca.fit_transform(X_train_scaled)
            # X_test_pca = pca.transform(X_test_scaled)

            # 训练并预测
            results = []
            for name, reg in regressors.items():
                scaler = MinMaxScaler()
                # X_train = scaler.fit_transform(X_train)
                reg.fit(X_train, y_train)
                # X_train = scaler.inverse_transform(X_train)
                
                y_pred = reg.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                me = max_error(y_test, y_pred)
                results.append((name, mae, rmse,r2,me))
                
                # Plotting prediction results
                plt.figure(figsize=(15, 5))
                plt.plot(y_test, label='True Values')
                plt.plot(y_pred, label='Predicted Values')
                plt.xlabel('样本 [-]')
                plt.ylabel(ds.target_names)
                plt.title(f'{ds.description}-{name}')
                plt.legend()
                plt.show()
                # ff.create_scatterplotmatrix(y_test)
                stfig = px.scatter_matrix(y_test)
                stfig.update_layout(title=f'{ds.description}-{name}')
                stfig.update_xaxes(title='样本 [-]')
                stfig.update_yaxes(title=''.join(ds.target_names))

            results_df = pd.DataFrame(results, columns=['Regressor', 'MAE', 'rmse','R2','me'])
            print(results_df)

            # # 生成数据报告
            # report = results_df.sort_values(by='R2', ascending=False)
            # report.to_csv('regression_results_report.csv', index=False)
            
            # 生成详细数据报告
            with pd.ExcelWriter('detailed_regression_report.xlsx') as writer:
                desc_report.to_excel(writer, sheet_name='Data Descriptive Stats')
                results_df.to_excel(writer, sheet_name='Regression Results')
                
        return [stfig, desc_report]
   
    
def main_process():
    
    current_directory = os.getcwd()
    print('当前工作目录:', current_directory)

    # life_analysis = Life_analysis()
    
    # 数据读取
    # engine_datasets = Data_Input.read_xls_2022_mainteance()
    # engine_datasets = Data_Input.read_xls_2022_benchtest()
    # engine_datasets =Data_Input.read_xls_waterpump_mt()
    # engine_datasets = Data_Input.read_xls_life_data()
    
    # engine_datasets = LoadDataset(file_path = FILE_2022_BENCH_TEST_FORMAT)
    engine_datasets = LoadDataset(file_path = FILE_PART_LIFE)
    
    # print(vars(engine_datasets))
    
    # 原始数据可视化和统计指标
    # data_input.stats_analysis(engine_datasets)
    # Data_visual.plot_data([engine_datasets])
    
    # 分类训练测试，参数优化，结果对比
    # Classifier.classifier_vote([engine_datasets])
    # Classifier.classifier_vote_pre([engine_datasets])
    # Classifier.parameter_optimize(engine_datasets)
    # Classifier.classifier_comparison(engine_datasets)
    
    # 数据特征选择
    # Fault_diagnose.feature_selection([engine_datasets])
    # Fault_diagnose.PCA_analysis(engine_datasets)
    
    # 连续数据预测回归
    # Life_analysis.life_weibull([engine_datasets])
    Regressor.predict([engine_datasets])
    
    # ToDoList 
    # 神经网络算法，图像识别算法LSTMCNN，声音振动频谱变化算法
    # 报告自动生成，数据库输入输出，

if __name__ == '__main__':
   main_process()


