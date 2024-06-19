import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
from PIL import Image


from data_analysis import Data_Input, Data_visual, Classifier, Fault_diagnose, Regressor
import data_analysis

    
def fig_show():
    engine_datasets = Data_Input.read_xls_life_data()
    return Regressor.predict(engine_datasets)
    

image = Image.open('./img/logo.png')
st.image(image, use_column_width='auto')


st.write("""
# 产品质量分析工具箱

本工具箱用于分析 **大马力柴油发动机检修项目** 
""")

# 导航栏界面
st.sidebar.header('导航栏')

data_source = st.sidebar.radio("选择数据输入方式", ("上传数据文件", "使用程序内数据"))
st.sidebar.markdown("""
[输入数据文件示例 ](https://docs.qq.com/sheet/DQ1BqUk5xTlJoTUpw?tab=BB08J2)
""")

if data_source == "上传数据文件":
    uploaded_file = st.sidebar.file_uploader("上传数据文件 格式EXCEL", type=["xlsx"])
    if uploaded_file is not None:
        st.write("数据已导入")
        input_df = pd.read_excel(uploaded_file)
        engine_datasets = data_analysis.LoadDataset()
        engine_datasets.load_from_dataframe(input_df)
else:
    st.write("使用程序内数据")
    engine_datasets = data_analysis.LoadDataset(file_path=data_analysis.FILE_2022_BENCH_TEST_FORMAT)


engine_type = st.sidebar.selectbox('发动机型号',('930E','MT5500','830E','SF33900','MT4400'))
engine_id = st.sidebar.text_area("发动机序列号", "SN-3316-5988")


if st.button('数据分析'):
    st.write('结果展示')
    engine_datasets = data_analysis.LoadDataset(file_path = data_analysis.FILE_2022_BENCH_TEST_FORMAT)
    Data_visual.analyze_data([engine_datasets])
    # fig, N = Data_visual.analyze_data([engine_datasets])
    # st.pyplot(fig)
    
    
if st.button('质量评价'):
    
    st.write("""
    # 故障诊断工具

    通过应用先进机器学习分类算法，根据发动机热试台架试验历史数据，辅助分析判断发动机检修质量
    """)
    
    engine_datasets = data_analysis.LoadDataset(file_path = data_analysis.FILE_2022_BENCH_TEST_FORMAT)
    Classifier.classifier_vote_pre([engine_datasets])

    
if st.button('故障诊断'):
    st.write("""
    # 故障诊断工具

    在发动机出现故障时，通过应用机器学习分类算法中的各特征对故障分类的贡献度，快速发现发动机检测装配时的隐含问题，及时给出可能故障原因和辅助诊断建议。。
    """)
    
    engine_datasets = data_analysis.LoadDataset(file_path = data_analysis.FILE_2022_BENCH_TEST_FORMAT)
    Fault_diagnose.feature_selection([engine_datasets])
    
    # feature_viusal()
    
if st.button('寿命分析'):
    st.write("""
    # 寿命分析工具

    通过应用机器学习中的先进回归算法和分析产品维修检测时的关键数据，预测产品寿命和失效日期，给出更精确的维修备品备库数量和制定更有效的维护计划。
    """)
    engine_datasets = data_analysis.LoadDataset(file_path = data_analysis.FILE_PART_LIFE)
    Regressor.predict([engine_datasets])
    


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
    
    
