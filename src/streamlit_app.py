import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import pandas as pd
from PIL import Image

def feature_viusal():
    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2
    hist_data = [x1, x2, x3]

    group_labels = ['特征 1', '特征 2', '特征 3']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[.1, .25, .5])

    st.plotly_chart(fig, use_container_width=True)
    

image = Image.open('../img/logo.png')

st.image(image, use_column_width='auto')


st.write("""
# 产品质量分析工具箱

本工具箱用于分析 **大马力柴油发动机检修项目** 
""")

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

type = st.sidebar.selectbox('发动机型号',('930E','5500'))

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    

st.sidebar.header('导航栏')
input = "930E_1234567"
SMILES = st.sidebar.text_area("发动机序列号", input)

if st.sidebar.button('数据分析'):
    st.write('结果展示')
    
if st.sidebar.button('产品质量分析'):
    st.write('结果展示')

st.sidebar.button('Sepal width')
st.sidebar.button('Petal length')
st.sidebar.button('Petal width')


df = pd.DataFrame({'产品序列号': '930E_12345678'}, index=[0])
st.subheader('输入产品序列号')
st.write(df)


# st.button("Reset", type="primary")
# if st.button('数据分析'):
#     x = st.slider("选择观测某列")
#     st.write("展示数据", x, "列")
#     st.write('结果展示')
#     feature_viusal()
    
if st.button('质量评价'):
    st.write('评价结果')
    feature_viusal()
    
if st.button('故障诊断'):
    st.write('诊断结果')
    feature_viusal()
    
if st.button('寿命分析'):
    st.write('分析结果')
    feature_viusal()


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
    # st.write('Great!')
    feature_viusal()
    
    

