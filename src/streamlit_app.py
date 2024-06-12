import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
from PIL import Image


from data_analysis import Data_Input, Data_visual, Classifier, Fault_diagnose, Regressor
import data_analysis

def feature_viusal():
    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2
    hist_data = [x1, x2, x3]

    group_labels = ['A工况流量', 'B工况流量', 'C工况流量']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[.1, .25, .5])

    st.plotly_chart(fig, use_container_width=True)
    
def fig_show():
    engine_datasets = Data_Input.read_xls_life_data()
    return Regressor.predict(engine_datasets)
    

image = Image.open('./img/logo.png')
st.image(image, use_column_width='auto')


st.write("""
# 产品质量分析工具箱

本工具箱用于分析 **大马力柴油发动机检修项目** 
""")

st.sidebar.markdown("""
[Example EXCEL input file](https://docs.qq.com/sheet/DQ1BqUk5xTlJoTUpw?tab=BB08J2)
""")


type = st.sidebar.selectbox('发动机型号',('930E','5500'))

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input Excel file", type=["xlsx"])
if uploaded_file is not None:
    print("数据已导入")
    input_df = pd.read_excel(uploaded_file)
    engine_datasets = data_analysis.LoadDataset()
    engine_datasets.load_from_dataframe(input_df)

st.sidebar.header('导航栏')
input = "930E_1234567"
SMILES = st.sidebar.text_area("发动机序列号", input)

if st.button('数据分析'):
    st.write('结果展示')
    engine_datasets = data_analysis.LoadDataset(file_path = data_analysis.FILE_PART_LIFE)

    fig, N = Data_visual.plot_data([engine_datasets])
    st.pyplot(fig)
    
    
if st.button('质量评价'):
    
    st.subheader('分析结论')
    
    # engine_datasets = data_analysis.LoadDataset(file_path = data_analysis.FILE_2022_BENCH_TEST_FORMAT)
    stfig_ret, stfig_his, most_common_class = Classifier.classifier_vote_pre([engine_datasets])
    st.plotly_chart(stfig_ret, use_container_width=True)
    st.plotly_chart(stfig_his, use_container_width=True)
    
    st.write('根据大数据程序分析，该产品质量评价结果')
    df = pd.DataFrame({'产品类型': '发动机',
                       '检测结果': "合格" if most_common_class == 0 else "不合格" 
                       },index=[0])
    df.loc[1] = ['水泵','合格']
    st.write(df)
    
if st.button('故障诊断'):
    st.subheader('分析结论')
    
    engine_datasets = data_analysis.LoadDataset(file_path = data_analysis.FILE_2022_BENCH_TEST_FORMAT)
    stfig_rate , feature_weights = Fault_diagnose.feature_selection([engine_datasets])
    st.plotly_chart(stfig_rate, use_container_width=True)
    
    st.write('根据大数据程序分析，该产品检测项疑似故障')
    
    # feature_viusal()
    
if st.button('寿命分析'):
    st.subheader('分析结论')
    st.write('根据大数据程序分析，该产品剩余寿命为1000小时')
    st.write('该产品未来失效数量预测')
    df = pd.DataFrame({'需备品量（第一季度）': '200',
                    '需备品量（第二季度）': '200',
                    '需备品量（第三季度）': '200',
                    '需备品量（第四季度）': '200'
                    },index=[0])
    st.write('根据大数据程序分析，该产品剩余寿命为1000小时')
    
    st.write(df)
    
    # engine_datasets = data_analysis.LoadDataset(file_path = data_analysis.FILE_PART_LIFE)

    fig,report = Regressor.predict([engine_datasets])
    
    st.plotly_chart(fig, use_container_width=True)
    st.write(report)

# genre = st.radio(
#     "What's your favorite movie genre",
#     [":rainbow[Comedy]", "***Drama***", "Documentary :movie_camera:"],
#     captions = ["Laugh out loud.", "Get the popcorn.", "Never stop learning."])

# if genre == ':rainbow[Comedy]':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn\'t select comedy.")
    
    
agree = st.checkbox('I agree')

if agree:
    st.write('Great!')


# # 定义多个页面
# def home_page():
#     st.title("首页")
#     # 首页内容...

# def settings_page():
#     st.title("设置")
#     # 设置页面内容...

# def about_page():
#     st.title("关于")
#     # 关于页面内容...

# # 设置默认页面
# default_page = "首页"

# # 创建多页面应用
# def main():
#     # 侧边栏菜单
#     page = st.sidebar.radio("页面选择", ["首页", "设置", "关于"], index=0)

#     # 根据页面选择显示内容
#     if page == "首页":
#         home_page()
#     elif page == "设置":
#         settings_page()
#     elif page == "关于":
#         about_page()

# if __name__ == "__main__":
#     main()
    
    
