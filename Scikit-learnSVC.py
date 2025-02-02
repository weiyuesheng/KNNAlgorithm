from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier    # KNN
from sklearn.linear_model import LogisticRegression   # 逻辑回归
from sklearn.svm import SVC                           # 向量机

DATA_file = "Iris_Data.csv"

SPECIES = {
    "Iris-setosa" : 0,  # 山鸢尾
    "Iris-versicolor" : 1,  # 变色鸢尾
    "Iris-virginica" : 2    # 维吉尼亚鸢尾 
} 

# 特征列
FEAT_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def main():
    """
        主函数
    """
    iris_data = pd.read_csv(DATA_file, index_col="Id")   # 返回一个DataFrame数据结构
    iris_data["Label"] = iris_data["species"].map(SPECIES)   # Series数据类型的map操作

    # 获取数据集特征
    X = iris_data[FEAT_COLS].values    # Series数据类型转变为一个numpy数据类型

    # 获取数据标签
    y = iris_data["Label"].values      # 也是numpy数据类型数据

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

    # 相当于是声明模型
    model_dict = {"KNN" : KNeighborsClassifier(n_neighbors=7),
                    "Logistic Regression" : LogisticRegression(C=1e4),
                    "SVM" : SVC(C=1e4)}
    for model_name, model in model_dict.items():
        # 训练模型
        model.fit(X_train, y_train)

        # 验证模型
        accuracy = model.score(X_test, y_test)

        # 输出准确率
        print("模型名称为：{}, 正确率为: {}%".format(model_name, accuracy * 100))

if __name__ == "__main__":
    main()