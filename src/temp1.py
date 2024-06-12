# 获取用户输入的两个字符形式的 ASCII 码
ascii_code = input("请输入两个字符形式的 ASCII 码: ")

# 确保输入的是两个字符
if len(ascii_code) != 2:
    print("请输入两个字符形式的 ASCII 码。")
else:
    # 将两个字符拼接成一个完整的 ASCII 码
    ascii_value = int(ascii_code)
    
    # 使用 chr() 函数将 ASCII 码转换为字符
    character = chr(ascii_value)
    
    # 打印对应的字符
    print(f"ASCII 码 {ascii_code} 对应的字符是 '{character}'。")