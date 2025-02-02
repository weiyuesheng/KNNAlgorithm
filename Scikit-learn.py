"""
    任务：鸢尾花识别
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_file = "Iris_Data.csv"

SPECIES = {
    "Iris-setosa" : 0,  # 山鸢尾
    "Iris-versicolor" : 1,  # 变色鸢尾
    "Iris-virginica" : 2    # 维吉尼亚鸢尾 
} 

FEAT_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def main():
    """
        主函数
    """
    # 读取数据集
    iris_data = pd.read_csv(DATA_file, index_col='Id')  # 返回类型为DATAFrame类型数据结构
    iris_data["Label"] = iris_data["species"].map(SPECIES)   # 映射为标签。 0 1 2 

    # 获取数据集特征
    X = iris_data[FEAT_COLS].values  # 最后变为一个二维numpy的形式

    # 获取数据标签
    y = iris_data["Label"].values  # 这个就是一维numpy
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=11)  # X为特诊， y为标签
    
    # 声明模型
    Knn_model = KNeighborsClassifier()  # 默认参数

    # 训练模型
    Knn_model.fit(X_train, y_train)

    # 模型测试
    accuracy = Knn_model.score(X_test, y_test)

    print("预测准确率为{:.2f}".format(accuracy * 100))

    # 取单个样本测试
    test_simple_feature = [X_test[0, :]]
    y_true = y_test[0]
    y_pred = Knn_model.predict(test_simple_feature)  # 输出对应的预测标签
    print(y_pred)  # 单列表形式
    print(y_true)

if __name__ == "__main__":
    main()



