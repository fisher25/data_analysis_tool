import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
# from weasyprint import HTML

# 生成数据
data = {'Name': ['Product A', 'Product B', 'Product C'],
        'Sales': [100, 150, 200]}
df = pd.DataFrame(data)

# 使用 Matplotlib 生成图表
plt.figure(figsize=(8, 6))
plt.bar(df['Name'], df['Sales'])
plt.xlabel('Product')
plt.ylabel('Sales')
plt.title('Data Analysis')
# 保存图表为 PNG 文件
chart_path = "Data Analysis.png"
plt.savefig(chart_path)
plt.close()

# 创建 Jinja2 环境并加载模板
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('./html/report_template.html')

# 使用模板渲染 HTML
html_out = template.render(chart_path=chart_path, ret_data=df.to_html(index=False))

# 将渲染的 HTML 写入到文件
html_file = './html/report.html'
with open(html_file, 'w') as f:
    f.write(html_out)

# # 使用 WeasyPrint 转换 HTML 为 PDF
# pdf_file = 'report.pdf'
# HTML(html_file).write_pdf(pdf_file)

# print(f'Report has been generated: {pdf_file}')