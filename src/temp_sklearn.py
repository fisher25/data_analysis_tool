import pandas as pd

def calculate_remaining_life_and_spare_parts(expected_lifespans, current_ages):
    """
    计算产品剩余寿命，并预测未来3年每个季度寿命到期的产品个数。
    
    :param expected_lifespans: list, 产品的预期寿命列表
    :param current_ages: list, 产品现在的年龄列表
    :return: DataFrame, 未来3年每个季度的备品个数表格
    """
    # 计算剩余寿命
    remaining_lifes = [expected - current for expected, current in zip(expected_lifespans, current_ages)]
    
    # 创建时间轴，未来3年每个季度
    quarters = pd.date_range(start=pd.Timestamp('now'), periods=12, freq='Q')
    
    # 创建计数列表，用于记录每个季度到期的产品个数
    spare_parts_per_quarter = [0] * len(quarters)
    
    # 遍历每个产品的剩余寿命，计算各个季度的到期个数
    for life in remaining_lifes:
        # 找到剩余寿命在哪个季度到期
        expiry_quarter = next((index for index, date in enumerate(quarters) if life < (date.year - pd.Timestamp('now').year) * 12 + (date.month - pd.Timestamp('now').month) / 3), None)
        # 如果在3年内到期，则在相应季度计数
        if expiry_quarter is not None:
            spare_parts_per_quarter[expiry_quarter] += 1
    
    # 创建输出表格
    df = pd.DataFrame({
        'Quarter': quarters,
        'Spare Parts Needed': spare_parts_per_quarter
    })
    
    return df

# 示例数据
expected_lifespans = [10, 15, 20, 5, 7]  # 预期寿命年数
current_ages = [2, 3, 10, 1, 4]         # 当前年龄年数

# 计算并打印表格
spare_parts_table = calculate_remaining_life_and_spare_parts(expected_lifespans, current_ages)
print(spare_parts_table)