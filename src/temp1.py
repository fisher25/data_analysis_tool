


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
plt.rcParams['font.family'] = 'SimHei'  # 将字体设置为SimHei或其他中文字体

def remaining_lifetime_distribution(ages, expected_lifetimes):
    # 计算剩余寿命
    remaining_lifetimes = expected_lifetimes - ages

    # 定义季度长度
    quarter_length = 90  # 90天一个季度

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

    # 创建表格
    remaining_df = pd.DataFrame(remaining_distribution, columns=['季度', '产品数量'])

    # 绘制剩余寿命分布图
    plt.figure(figsize=(10, 6))
    plt.bar(remaining_df['季度'], remaining_df['产品数量'], color='skyblue')
    plt.xlabel('季度')
    plt.ylabel('产品数量')
    plt.title('剩余寿命分布')
    plt.xticks(rotation=45)
    
    # 使用 Streamlit 显示图像和表格
    st.pyplot(plt)
    st.write("### 剩余寿命表格")
    st.write(remaining_df)

    return remaining_df

# Streamlit 应用程序
def main():
    st.title("剩余寿命分布计算器")

    # 输入年龄和预期寿命
    st.write("请输入产品的年龄和预期寿命：")
    ages = st.text_area("年龄 (用逗号分隔)", "100, 200, 250, 300, 400, 500")
    expected_lifetimes = st.text_area("预期寿命 (用逗号分隔)", "1000, 1100, 1200, 1300, 1400, 1500")

    if st.button("计算剩余寿命分布"):
        try:
            ages = np.array([int(x) for x in ages.split(",")])
            expected_lifetimes = np.array([int(x) for x in expected_lifetimes.split(",")])
            remaining_df = remaining_lifetime_distribution(ages, expected_lifetimes)
            st.write(remaining_df)
        except ValueError:
            st.error("请输入有效的整数列表。")

if __name__ == "__main__":
    main()