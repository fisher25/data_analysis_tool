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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
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

import seaborn as sns
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
FILE_PART_LIFE = './data/fakedata_waterpump_life.xlsx'
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

    # # 最后一列是目标列，其他是特征列
        
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
        self.data = df.iloc[:, :-1].values
        self.target = df.iloc[:, -1].values
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
        description = '测试数据和寿命'
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
    def analyze_data(datasets):
        
        """
        分析输入的M*N numpy数据，输出统计特征，分布图，箱线图，相关性热图和成对关系图。
        """
        dataset = datasets[0]
        M, N = dataset.data.shape
        N = min(N,5)
        data = dataset.data[:,0:N]
        names = dataset.feature_names[0:N]
        
        # Convert the numpy array to a pandas DataFrame
        df = pd.DataFrame(data)

        st.write("""
        # 元数据分析工具
        
        元数据分析是数据分析中的重要步骤，通过分析数据的统计特征和分布情况，我们可以发现数据采集中的异常值、离群值、数据的离散程度以及特征之间的相关性。这些信息对于进一步的数据分析和建模至关重要。
        """)

        # Calculate statistical metrics
        stats = df.describe().T
        stats['range'] = stats['max'] - stats['min']
        stats['variance'] = df.var()
        stats['skewness'] = df.skew()
        stats['kurtosis'] = df.kurtosis()

        st.write("## 数据统计特征")
        st.write("以下表格显示了数据的基本统计特征，包括均值、标准差、最小值、最大值、范围、方差、偏度和峰度。")
        st.write("""
        - **均值**反映了数据的集中趋势。
        - **标准差**和**方差**显示了数据的离散程度。
        - **偏度**和**峰度**反映了数据分布的形状特征。
        - **范围**（最大值 - 最小值）提供了数据的跨度信息。
        """)
        st.write(stats)

        # Calculate correlation matrix
        corr_matrix = df.corr()

        st.write("## 每个特征的直方图")
        st.write("以下直方图展示了每个特征的分布情况，有助于理解数据的集中趋势和离散程度。")
        st.write("""
        - 直方图显示了各特征值的频率分布。
        - 可以观察到数据是否呈现正态分布，是否存在偏态或多峰现象。
        """)
        df.hist(bins=30, figsize=(20, 15))
        st.pyplot(plt)

        st.write("## 每个特征的箱线图")
        st.write("以下箱线图展示了每个特征的分布及异常值情况，有助于识别数据中的异常点和四分位数分布。")
        st.write("""
        - 箱线图显示了中位数、上四分位数和下四分位数。
        - 图中还显示了数据的异常值（离群点），这些点可能是数据采集中的异常值。
        """)
        df.plot(kind='box', subplots=True, layout=(int(np.ceil(df.shape[1] / 3)), 3), figsize=(20, 15))
        st.pyplot(plt)

        st.write("## 特征之间的相关性热图")
        st.write("以下热图展示了特征之间的相关性，有助于识别特征之间的相互关系。")
        st.write("""
        - 热图中的颜色表示特征之间的相关性系数。
        - 正相关性用红色表示，负相关性用蓝色表示。
        - 数值越接近1或-1，表示相关性越强。高相关性可以提示我们某些特征之间的线性关系。
        """)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmax=1.0, vmin=-1.0, linewidths=0.1)
        st.pyplot(plt)

        st.write("## 成对特征之间的关系")
        st.write("以下成对关系图展示了特征之间的散点图矩阵，有助于观察特征之间的线性或非线性关系。")
        st.write("""
        - 成对关系图显示了每对特征之间的散点图。
        - 可以观察到特征之间
        """)
        sns.pairplot(df)
        st.pyplot(plt)
    
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
        """
        生成excel 100行 第一列低速流量，第二列高速流量，第三列年龄，第四列寿命，
        读取数据，
        展示数据，
        根据第四列寿命列出weibull分布画图
        给出中位数寿命，寿命概率分布特征
        给出未来几年寿命到期失效产品数量
        根据数据划分训练集测试集
        测试回归算法准确率和分布，给出总评价
        给出同一个产品各算法的预测值，真值。
        给出综合预测寿命的结果。
        """
        
        for ds_cnt, ds in enumerate(datasets):
            # preprocess dataset, split into training and test part
            # X, y = ds
            data = ds.data
            life_dataset = ds.target
        
        
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
        

    @staticmethod
    def remaining_lifetime_distribution(ages, expected_lifetimes):
        # 计算剩余寿命
        remaining_lifetimes = expected_lifetimes - ages

        # 定义季度长度
        quarter_length = 90 

        # 计算剩余寿命的季度分布
        max_remaining_lifetime = max(remaining_lifetimes)
        num_quarters = int(np.ceil(max_remaining_lifetime / quarter_length))
        
        quarters = [f'Q{i+1}' for i in range(num_quarters)]
        remaining_distribution = []
        
        for i in range(num_quarters):
            start_day = i * quarter_length
            end_day = (i + 1) * quarter_length
            count = np.sum((remaining_lifetimes >= start_day) & (remaining_lifetimes < end_day))
            remaining_distribution.append((quarters[i], count))

        remaining_df = pd.DataFrame(remaining_distribution, columns=['季度', '产品数量'])

        # 绘制剩余寿命分布图
        plt.figure(figsize=(10, 6))
        plt.bar(remaining_df['季度'], remaining_df['产品数量'], color='skyblue')
        plt.xlabel('季度')
        plt.ylabel('产品数量')
        plt.title('未来预期失效产品分布')
        plt.xticks(rotation=45)
        
        # 使用 Streamlit 显示图像和表格
        st.write("### 未来预期失效产品")
        st.write(" - 此结果用于维修库存备品参考")
        st.write(" - 根据产品寿命预测结果，未来失效产品数量如下")
        st.pyplot(plt)
        st.write(remaining_df)


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
        for ds_cnt, ds in enumerate(dataset):
            X = ds.data
            y = ds.target

            X_indices = np.arange(X.shape[1])

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

                plt.figure()
                plt.bar(X_indices, feature_importances)
                plt.xlabel('特征索引 [-]')
                plt.ylabel(f'{model_name} 权重')
                plt.title(f'{model_name} 特征重要性')
                st.write(f"## {model_name}")
                st.write(f"-  基于{model_name}模型的特征重要性")
                st.pyplot(plt)
        
       
        st.write(f"## 问题检测项")
        st.write(f"问题检测项索引: {np.argmax(feature_importances)}")
        
        return 
  
        
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
        models = {
            "K近邻": KNeighborsClassifier(3),
            "线性SVM": SVC(kernel="linear", C=0.025, random_state=42),
            "径向SVM": SVC(gamma=2, C=1, random_state=42),
            "高斯过程": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
            "决策树": DecisionTreeClassifier(max_depth=5, random_state=42),
            "随机森林": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
            "神经网络": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            "增强学习": AdaBoostClassifier(random_state=42),
            "贝叶斯": GaussianNB(),
            "二次判别": QuadraticDiscriminantAnalysis()
        }

        for ds in datasets:
            X, y = ds.data, ds.target

            st.write("## 数据展示")
            st.write("单行数据为某发动机台架试验数据")
            st.write(pd.DataFrame(X).head())
            st.write(f"标签分布: {np.unique(y)}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            accuracies = {}
            predictions = []

            for name, model in models.items():
                clf = make_pipeline(StandardScaler(), model)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracies[name] = accuracy_score(y_test, y_pred)
                predictions.append(y_test)
                

            st.write("### 训练算法\n - 将数据分为训练集和测试集")
            st.write(f" - 基于训练集应用机器学习算法: {', '.join(models.keys())}\n")
            st.write(" - 基于测试集验证结果")

            # Plotting the accuracy
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(list(models.keys()), list(accuracies.values()), color='g', edgecolor='k', alpha=0.75)
            ax.set_xticklabels(list(models.keys()), rotation=45, ha='right')
            ax.set_ylabel('质量判定结果准确率 %')
            st.pyplot(fig)
            
            # Select the best model based on accuracy
            best_model_name = max(accuracies, key=accuracies.get)
            best_model = models[best_model_name]

            # Make predictions with all models for voting
            predictions = np.array(predictions)
            final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)

            # Calculate accuracy for the final predictions
            final_accuracy = accuracy_score(y_test, final_predictions)

            # st.write(f"标签分布ytest: {np.unique(y_test)}")
            # st.write(f"标签分布ypred: {np.unique(best_model.predict(X_test))}")
            # Display classification report for the best model
            # st.write("### 最佳模型的分类报告")
            # best_model_report = classification_report(y_test, best_model.predict(X_test), target_names=["Class 0", "Class 1"], output_dict=True)
            # st.write(pd.DataFrame(best_model_report).transpose())

            # Plot confusion matrix for the final predictions
            st.write("### 最终投票结果的混淆矩阵")
            st.write(f"解释：\n- True Negatives (TN): 实际为0且预测为0的数量\n- False Positives (FP): 实际为0但预测为1的数量\n- False Negatives (FN): 实际为1但预测为0的数量\n- True Positives (TP): 实际为1且预测为1的数量")
            cm = confusion_matrix(y_test, final_predictions)
            fig_cm = px.imshow(cm, text_auto=True, x=["预测分类 0", "预测分类 1"], y=["实际分类 0", "实际分类 1"])
            st.plotly_chart(fig_cm)
            
            st.write("### 结论")
            st.write(f"""
            - 采用的模型包括: {', '.join(models.keys())}。
            - 最佳模型是 {best_model_name}，其在测试集上的准确度为 {accuracies[best_model_name]:.2f}。
            - 最终的预测结果通过多模型投票集成法决定，投票结果的准确度为 {final_accuracy:.2f}。
            - 混淆矩阵展示了最终投票结果的分布情况。
            """)
            
            # 创建数据框
            df_results = pd.DataFrame({
                "检测方法": models.keys(),
                "预测结果": ["合格" if pred[0] == 0 else "不合格" for pred in predictions]
            })
            final_result = "合格" if final_predictions[0] == 0 else "不合格"
            # 显示结果
            st.write("### 各分类器的预测结果")
            st.write(df_results)

            st.write(f"### 最终投票结果\n该产品质量评价结果: {final_result}")


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
       
        regressors = {
            '线性回归': LinearRegression(),# LinearRegression
            # '岭回归': Ridge(),# Ridge线性回归的基础上加入L2正则化，防止过拟合
            # 'Lasso回归': Lasso(),# Lasso 在线性回归的基础上加入L1正则化，可以进行特征选择。
            # '弹性网络回归': ElasticNet(),# ElasticNet 结合了L1和L2正则化的回归方法
            # '支持向量回归': SVR(kernel='linear'),# SVR
            '决策树回归': DecisionTreeRegressor(),# DecisionTreeRegressor
            '随机森林回归': RandomForestRegressor(),# RandomForestRegressor
            '梯度提升回归 ': GradientBoostingRegressor(),# GradientBoostingRegressor
            'K邻近回归': KNeighborsRegressor(),# KNeighborsRegressor
        }

        for ds_cnt, ds in enumerate(datasets):
            X = ds.data
            y = ds.target
            df = pd.DataFrame(ds.data)
            desc_report = df.describe()
            print('X' ,X)
            print('y' ,y)
            st.write("## 数据展示")
            st.write("单行数据为某产品检测数据，检测时年龄，和最终寿命数据")
            # ,columns=[ds.feature_names,ds.target_names]
            st.write(pd.DataFrame(np.hstack((X,y.reshape(-1,1)))).head())
            
            st.write("### 训练算法\n - 将数据分为训练集和测试集")
            st.write(f" - 基于训练集应用机器学习回归算法: {', '.join(regressors.keys())}\n")
            st.write(" - 基于测试集验证结果")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

            results = []
            predictions = []
            for name, reg in regressors.items():
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                reg.fit(X_train_scaled, y_train)
                y_pred = reg.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                me = max_error(y_test, y_pred)
                results.append((name, mae, rmse, r2, me))
                predictions.append((name, y_pred))
                
                # Plotting prediction results
                plt.figure(figsize=(15, 5))
                plt.plot(y_test, label='真实值')
                plt.plot(y_pred, label='预测值')
                plt.xlabel('样本 [-]')
                plt.ylabel(ds.target_names)
                plt.title(f'{name}')
                plt.legend()
                plt.show()
                st.write(f"## {name}")
                st.write(f"-  基于{name}回归算法的预测寿命与真实值对比")
                st.pyplot(plt)

            results_df = pd.DataFrame(results, columns=['Regressor', 'MAE', 'RMSE', 'R2', 'ME'])

            # Plotting MAE for each regressor
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Regressor', y='MAE', data=results_df)
            plt.title('MAE for each Regressor')
            plt.xticks(rotation=45)
            plt.show()
            st.write("## 测试结果")
            st.write("- MAE 预测值与实际值之间绝对误差的平均值，反映了预测值与实际值之间的平均偏差程度")
            st.write("-  RMSE 预测值与实际值之间误差的平方平均值的平方根。它反映了预测值与实际值之间的标准差")
            st.write(results_df)
            st.pyplot(plt)
            
            # Create table with predictions
            # pred_df = pd.DataFrame(predictions, columns=['Regressor', 'Predicted Lifetime'])
            # avg_pred = np.mean([y for _, y in predictions], axis=0)
            # pred_df.loc[len(pred_df)] = ['Average', avg_pred]
            # pred_df.loc[len(pred_df)] = ['True Lifetime', y_test]
            # st.write(pred_df)

            Life_analysis.remaining_lifetime_distribution(X_test[:,-1],y_pred)
    
    
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


