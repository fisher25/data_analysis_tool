from openai import OpenAI


client = OpenAI(
    api_key="sk-9ukyvrMb1ZmJqq2az6fHMTLTmRRa2bvQev8c68sayHB3P7fx",
    base_url="https://api.moonshot.cn/v1",
)

text = f"""
仓储区实现发动机零部件集中控制管理和配送的自动化要求，从而更好的组织检修任务计划、降低劳动强度和提高工作效率。仓储区主要的设备配置包含多层式物料货架、叉车式AGV、仓储管理员工作平台、物料托盘、仓储管理系统和服务器、NG品区。

"""

# 需要总结的文本内容
prompt = f"""
把用三个反引号括起来的文本总结成一句话。
```{text}```
"""
 
completion = client.chat.completions.create(
  model="moonshot-v1-8k",
  messages=[
    {"role": "system", "content": "你是发动机检修产线智能专家，你会为用户提供安全，有帮助，准确的回答。"},
    {"role": "user", "content": prompt}
  ],
  temperature=0.3,
)
 
print(completion.choices[0].message)


