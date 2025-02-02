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

# 特征列
FEAT_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 因为特征太多，所以将特征分开进行处理
def investigate_knn(iris_data, sel_cols, k_value):
    """
        # 不同k值对模型的影响
    """
    X = iris_data[sel_cols].values
    y = iris_data["Label"].values  # iris_data 转换为Series类型，再转换为numpy类型数据

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

    # 声明模型
    knn_model = KNeighborsClassifier(n_neighbors=k_value)  # 指定k值
    
    # 训练模型
    knn_model.fit(X_train, y_train)

    # 准确率
    accuracy = knn_model.score(X_test, y_test)
    print("k={}, accuracy={}".format(k_value, accuracy*100))

def main():
    """
        主函数
    """
    # 读取数据集
    iris_data = pd.read_csv(DATA_file, index_col="Id")  # 返回一个DATAFrame数据结构
    iris_data["Label"] = iris_data["species"].map(SPECIES)  # DATAFrame 数据结构新增一列数据

    k_values = [3, 5, 10, 12]
    sel_cols = ["sepal_length", "sepal_width"]
    for k_value in k_values:
        print('k = {}'.format(k_value))
        investigate_knn(iris_data, sel_cols, k_value) 

if __name__ == "__main__":
    main()


