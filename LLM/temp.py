
import streamlit as st

# 假设这是调用语言模型的函数
def get_model_response(question):
    # 这里是模拟的回答，实际应用中你需要调用模型生成回答
    return f"这是针对问题'{question}'的回答。"

# 初始化对话历史，如果还没有的话
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 创建一个表单，用于输入问题并提交
with st.form("qa_form"):
    question = st.text_input("请输入你的问题：", "")
    submit_button = st.form_submit_button("提交")

# 当表单被提交时
if submit_button:
    if question:  # 确保问题不为空
        # 获取模型的回答
        answer = get_model_response(question)
        # 将问题和回答添加到历史记录
        st.session_state['history'].append((question, answer))
    else:
        st.warning("请输入一个问题。")

# 显示历史对话
st.write("### 对话历史")
for q, a in st.session_state['history']:
    st.text_area("问题", value=q, height=30, disabled=True)
    st.text_area("回答", value=a, height=60, disabled=True)