import streamlit as st

# 使用 st.write 函数嵌入 HTML 网页
def embed_html_page():
    html_code = '''

    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>AI发动机维修数字专家</title>
    <style>
        /* 添加样式以美化界面 */
        body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        }
        
        /* 导航栏样式 */
        .navbar {
        background-color: #333;
        color: #fff;
        padding: 10px;
        }
        
        /* 菜单栏样式 */
        .menu {
        background-color: #f4f4f4;
        padding: 10px;
        }
        
        /* 问答栏样式 */
        .qa {
        padding: 10px;
        }
        
        /* 维修知识库栏样式 */
        .knowledge-base {
        background-color: #f4f4f4;
        padding: 10px;
        }
    </style>
    </head>
    <body>
    <!-- 导航栏 -->
    <div class="navbar">
        <h1>AI发动机维修数字专家</h1>
    </div>
    
    <!-- 菜单栏 -->
    <div class="menu">
        <ul>
        <li><a href="#">首页</a></li>
        <li><a href="#">关于</a></li>
        <li><a href="#">联系我们</a></li>
        </ul>
    </div>
    
    <!-- 问答栏 -->
    <div class="qa">
        <h2>问题与答案</h2>
        <form>
        <input type="text" placeholder="请输入您的问题">
        <button type="submit">提交</button>
        </form>
        <div id="answer"></div>
    </div>
    
    <!-- 维修知识库栏 -->
    <div class="knowledge-base">
        <h2>维修知识库</h2>
        <ul>
        <li>常见问题</li>
        <li>故障排除</li>
        <li>维护指南</li>
        </ul>
    </div>
    
    <!-- JavaScript代码，用于处理问答功能 -->
    <script>
        const form = document.querySelector('form');
        const answerDiv = document.getElementById('answer');
        
        form.addEventListener('submit', function(event) {
        event.preventDefault();
        const question = form.querySelector('input').value;

        const options = {
            method: 'POST',
            headers: {'Content-Type': 'application/json', Authorization: 'Bearer Link_am6pmoLrHT43jVRZ5z3NIzZ6Q66BrSpNo5CCkXOcoe'},
            body: JSON.stringify({"app_code":"NhSXB3XH","messages":[{"role":"user","content":question}]})
        };

        // https://link-ai.tech/app/NhSXB3XH
        // https://api.link-ai.chat/v1/chat/completions
        fetch('https://api.link-ai.chat/v1/chat/completions', options)
        .then(response => response.json())
        .then(data => {
            // 读取 content 字段的值
            const content = data.choices[0].message.content;
            answerDiv.innerHTML = content
            console.log(content);
        })
        .then(response => console.log(response))
        .catch(err => console.error('Error:', err));

        });
    </script>
    </body>

    </html>
    '''

    st.write(html_code, unsafe_allow_html=True)

# 主函数
def main():
    st.title("Streamlit 页面中包含 HTML 网页示例")
    st.write("以下是嵌入的 HTML 网页：")
    embed_html_page()

if __name__ == "__main__":
    main()