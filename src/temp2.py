class CustomDataset():
    def __init__(self, data, target, feature_names=None, target_names=None, description=None):
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_names = target_names
        self.description = description
        
    def temp(self):
        # 创建自定义数据集对象
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        target = [0, 1, 0]
        feature_names = ['feature1', 'feature2', 'feature3']
        target_names = ['class1', 'class2']
        description = 'Custom dataset example'

        custom_dataset = CustomDataset(data=data, target=target, feature_names=feature_names,
                                    target_names=target_names, description=description)
        return custom_dataset


data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
target = [0, 1, 0]
cus = CustomDataset(data,target)
print(vars(cus.temp()))
print(vars(cus))