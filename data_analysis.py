
from stat import FILE_ATTRIBUTE_ARCHIVE
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, StringEditor, NumberEditor, TableColumn
from bokeh.layouts import column, row
from bokeh.plotting import figure


from datetime import date,datetime, timedelta
import numpy as np
import pandas as pd
import xlrd
import csv
import struct
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d  

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import weibull_min

from sklearn.impute import SimpleImputer
from sklearn import datasets
from sklearn import model_selection
from sklearn import svm

from sklearn.pipeline import Pipeline

from matplotlib.colors import ListedColormap

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

plt.rcParams['font.family'] = 'SimHei'  # 将字体设置为SimHei或其他中文字体


# 定义全局常量文件路径等          
FILE_2022_BENCH_TEST = './data/2022_benchtest.xlsx'
FILE_2022_MAINTEANCE = '../data/2022_benchtest.xlsx'
FILE_FAKE_MAINTEANCE = '../data/fake_mainteance_data.xlsx'
FILE_FAKE_BENCH_CAN = '../data/fake_bench_CANdata.xlsx'

FILE_PART_LIFE = '../data/life_dataset.xlsx'

# 
# 将数据预处理过程封装成函数,包括缺失值处理、标准化等

class CustomDataset():
    def __init__(self, data, target, feature_names=None, target_names=None, description=None):
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_names = target_names
        self.description = description
        
    def temp():
        # 创建自定义数据集对象
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        target = [0, 1, 0]
        feature_names = ['feature1', 'feature2', 'feature3']
        target_names = ['class1', 'class2']
        description = 'Custom dataset example'

        custom_dataset = CustomDataset(data=data, target=target, feature_names=feature_names,
                                    target_names=target_names, description=description)


class Data_stat:
    
    def __init__(self) -> None:
        pass
    
    
    def read_xls_2022_mainteance(self):
        
        filename =FILE_2022_MAINTEANCE
        # data = pd.read_excel(filename,sheet_name='Sheet1',header=None)
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
                cylinder_round = np.array(data.iloc[20:32,5:13].values).reshape(1,96) # 缸套圆度数据
                cylinder_height = np.array(data.iloc[35:43,4:12].values).reshape(1,64) # 缸套高度数据
                flywheel_vet = np.array(data.iloc[46,3]).reshape(1,1) # 飞轮径向总指示跳动
                flywheel_data = np.array(data.iloc[46,6]).reshape(1,1) # 飞轮端面总指示跳动
                
                compressor_flow = np.ones((1,1)) # 增压器流量 
                compressor_pressure = np.ones((1,1)) # 增压器压力
                cyl_head_torque = np.ones((1,1)) # 缸盖螺栓拧紧扭矩 
                
                test_ret = np.array(data.iloc[1,10]).reshape(1,1) # 热试验结果

                variable_name = f"data_{engine_SN}" 
                engine_data = {"engine_SN": engine_SN, "rod_diameter": rod_diameter,"cylinder_round": cylinder_round,"cylinder_height": cylinder_height}
                raw_data.update({variable_name: engine_data})
                feature = np.concatenate((rod_diameter,cylinder_round,cylinder_height,flywheel_vet,flywheel_data),axis=1) # 数组和数组拼接,按列增长方向拼接(水平拼接) 
            
                test_rets = np.vstack((test_rets,test_ret))
                features = np.vstack((features, feature))
                
        print('raw_data',raw_data)   
        test_rets = np.delete(test_rets, 0, axis=0)
        features = np.delete(features, 0, axis=0)   
        print('test_rets',test_rets)  
        print('features',features)
        
        filler = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        filler.fit(features)
        features = filler.transform(features)
        filler.fit(test_rets)
        test_rets = filler.transform(test_rets)
        
        maintain_data = (features, test_rets)

        engine_datasets = [
            maintain_data,
        ]
        
        return  engine_datasets

    
    def read_xls_2022_benchtest(self):
        # 输入：历史数据 2022年热试试验数据
        # 输出：标准格式 dataset 
    
        filename =FILE_2022_BENCH_TEST
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

                # feature = np.concatenate((testdata),axis=1) # 数组和数组拼接,按列增长方向拼接(水平拼接) 
                test_rets = np.vstack((test_rets,test_ret))
                features = np.vstack((features, feature))
        
        data = pd.read_excel(filename,sheet_name='Sheet9',header=None) 
        print('column',data.columns)
        
        test_point = data.iloc[6:12, 2].values
        test_item = data.iloc[3, 3:16].values
        feature_names = [f"{a}+{b}" for a in test_point for b in test_item]
        feature_names = [s.replace(" ", "") for s in feature_names]
        print('feature_names',feature_names)
        # print('test_point',test_point)
        # print('test_item',test_item)

        target_names = ['normal', 'fault']
                
        test_rets = np.delete(test_rets, 0, axis=0)
        features = np.delete(features, 0, axis=0)   

        filler = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        filler.fit(features)
        features = filler.transform(features)
        filler.fit(test_rets)
        test_rets = filler.transform(test_rets)
        
                # 创建自定义数据集对象

        description = 'Custom dataset example'

        mainteance_2022_dataset = CustomDataset(data=features, target=test_rets, feature_names=feature_names,
                                    target_names=target_names, description=description)
        
        return [(features, test_rets)]
    
    def read_xls_bench_CANdata(self):
        pass
    
    def read_xls_mainteance_data(self):
        pass
    
    def read_xls_life_data(self):

        weibull_data = weibull_min.rvs(2, loc=0, scale=500, size=2000).astype(int)
        df = pd.DataFrame(weibull_data, columns=['A'])
        filename = 'tempdata.xlsx'  # 
        df.to_excel(filename, index=False)
        
        
        df = pd.read_excel(FILE_PART_LIFE,sheet_name='Sheet1',header=0)
        life_dataset = df.iloc[:,2].values
        print('life_dataset ',life_dataset)
        return life_dataset
    
    def plot_dataset(self, dataset):
        M, N = dataset.shape
        N = min(N,5)

        fig, axs = plt.subplots(3, N, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.4)
        
        names = ['轴向间隙','径向间隙','A点流量','B点流量','C点流量']
        
        colors = ['#D81B60', '#0188FF', '#FFC107',
            '#B7A2FF', '#000000', '#2EC5AC']

        # 分布图
        for i in range(N):
            axs[0, i].hist(dataset[:, i], color=colors[1])
            axs[0, i].set_title(f'分布图 {names[i]}')

        # 散点图
        for i in range(N):
            axs[1, i].scatter(np.arange(M), dataset[:, i], color=colors[2])
            axs[1, i].set_title(f'散点图 {names[i]}')

        # 箱体图
        for i in range(N):
            axs[2, i].boxplot(dataset[:, i], vert=False, labels=[''], patch_artist=True)
            axs[2, i].set_title(f'箱体图 {names[i]}')

        plt.tight_layout()
        plt.show()

        

    def life_analysis(self, life_dataset =0):
        
        # 输入：产品寿命一维数组
        # 输出：预期寿命（寿命中位数）；寿命分布图；寿命箱体图；寿命威布尔拟合图；寿命关键参数（最小最大）
        
        # life_dataset = [100,102,105,108,110,130]
        
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
        return probability
        
        
    def fault_future(self,current_data):
        # 输入上线日期，上线个数，算出已运行时间
        # 根据
        # 输出指定时间段内，预期零部件故障个数
        signup_date = ["2019-9-26","2020-10-20","2021-11-11"]
        
        item_num = [100,500,800]
        
        # 日期字符串
        date_string = "2021-09-01"

        # 将日期字符串转换为日期对象
        date_object = datetime.strptime(date_string, "%Y-%m-%d")
        
        # 获取当前日期和时间
        now = datetime.now()
        print("当前日期和时间：", now)


        # 减去一周
        one_week = timedelta(weeks=1)
        previous_week = now - one_week
        print("减去一周后的日期和时间：", previous_week)

        # 加上一个月
        one_month = timedelta(days=30)
        next_month = now + one_month
        print("加上一个月后的日期和时间：", next_month)
   
    
    def stats_analysis(self, dataset):
        # 绘制数据集分布图并展示统计信息
        np.random.seed(0)
        # 示例用法
        n = 100  # 样本数
        m = 5    # 特征数
        data = np.random.randn(n, m)
        for ds_cnt, ds in enumerate(dataset):
            dataset, labels = ds
        
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

                
    def data_form_transform():
        # transform data form from multiple matrix to single matrix 
        np.reshape(newshape=(1, 2, 3), a=np.array([[[1, 2, 3], [4, 5, 6]]]))
    
    
    def PCA_analysis(self,datasets):
    # 执行分类分析  
        for ds_cnt, ds in enumerate(datasets):
            features, labels = ds
            
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
            
    def feature_selection(self,dataset):

        # #############################################################################
        # Import some data to play with

        # The iris dataset
        
        for ds_cnt, ds in enumerate(dataset):
            X, y = ds

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
            
            print('故障产品疑似存在问题的检测项目为','')


    def visualize_data(data):
    # 可视化展示
        pass
        # data.plot(kind='box')
        # data.plot(kind='scatter')  PCA降维后的数据可视化
        
    def classifier_comparison(self,datasets):
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
            X, y = ds
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
                
                # 使用分类器进行预测
                # y_pred = clf.predict(X_test)

                # # 统计投票结果
                # class_counts = np.bincount(y_pred)
                # most_common_class = np.argmax(class_counts)

                # Plot the training points
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
        
    def classifier_vote(self,datasets):
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
            X, y = ds
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42
            )

            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
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
            # 提取柱状名字和数值
            labels = names
            values = score * 100
            # 创建柱状图
            fig, ax = plt.subplots(figsize=(27, 9))
            ax.bar(labels, values,color='g',edgecolor='k',alpha=0.75)

            ax.set_title('各分类器分类准确率')
            ax.set_xlabel('分类器')
            # plt.xticks(rotation=45)
            ax.set_ylabel('准确率 %')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(27, 9))
            ax.bar(labels, ~y_pred.astype(np.bool_),color='g',edgecolor='k',alpha=0.75)

            ax.set_title('各分类器分析结果')
            ax.set_xlabel('分类器')
            # plt.xticks(rotation=45)
            ax.set_ylabel('分类器分类结果')
            plt.show()          
            
            most_common_class = np.argmax(class_counts)
            if  most_common_class ==0:
                print('多分类器投票结果：产品合格')
            else:
                print('多分类器投票结果：产品不合格')
        plt.tight_layout()
        plt.show()

    
    def clustering(self,datasets):
        pass
    
    def demension_reduction(self,datasets):
        pass
    
    def model_selection(self,datasets):
        pass
    
    def preprocessing(self,datasets):
        pass

    
def main_process():
    print(vars())
    data_mt = Data_stat()
    # data_mt.stats_analysis()
    
    # data_mt.test()
    # engine_datasets = data_mt.read_xls_2022_mainteance()
    engine_datasets = data_mt.read_xls_2022_benchtest()
    # dataset = data_mt.read_xls_life_data()
    
    # data_mt.stats_analysis(engine_datasets)
    # data_mt.classifier_vote(engine_datasets)
    # data_mt.feature_selection(engine_datasets)
    data_mt.PCA_analysis(engine_datasets)
    # data_mt.classifier_comparison(engine_datasets)
    
    
    # data_mt.life_analysis(dataset)
    
    
if __name__ == '__main__':
   main_process()
