
import panel as pn  # GUI
import requests
from openai import OpenAI

client = OpenAI(
    api_key="sk-9ukyvrMb1ZmJqq2az6fHMTLTmRRa2bvQev8c68sayHB3P7fx",
    base_url="https://api.moonshot.cn/v1",
)

model_list = client.models.list()
model_data = model_list.data
 
for i, model in enumerate(model_data):
    print(f"model[{i}]:", model.id)
    
file_list = client.files.list()
for file in file_list.data:
    print("uploaded files:",file) # 查看每个文件的信息
    


account_url = "https://api.moonshot.cn/v1/users/me/balance"
response = requests.get(account_url, headers = {
    "Authorization": "sk-9ukyvrMb1ZmJqq2az6fHMTLTmRRa2bvQev8c68sayHB3P7fx"
})
print("account info:",response.text)

pn.extension()

panels = [] # collect display 

def get_completion_from_messages(context):
    try:
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=context,
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error fetching completion: {e}")
        return "Error: Unable to fetch completion."

def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context = [{'role':'system', 'content':"""
你是发动机专家
"""
                } ]  # accumulate messages
    
# 你是发动机三坐标检测设备专家，为设备收集其他零件检测需求信息。
# 你要首先问候工人。然后等待工人回复收集检测信息。
# 收集完信息需要确认现有三坐标设备能不能满足检测要求，并给出答复。
# 最后需要询问检测零件的最晚期限，并送上祝福。

# 请确保明确所有选项，以便从设备中识别出该项唯一的内容。
# 你的回应应该以简短、和友好的风格呈现。

# 设备性能：
# 零件最大尺寸（长、宽、高） 6m、3m、2m
# 零件检测最高精度 0.01mm
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context) 
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        # pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, )))
 
    return pn.Column(*panels)

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard.show()

# C:\Users\yuguang\Desktop\试验台供应商资料