from sklearn.model_selection import train_test_split, GridSearchCV
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
    # 读取数据集
    iris_data = pd.read_csv(DATA_file, index_col="Id")    # 返回的是DATAFrame类型的数据结构
    iris_data["Label"] = iris_data["species"].map(SPECIES)   # 增加标签列转换为数字形式

    # 获取数据集特征
    X = iris_data[FEAT_COLS].values

    # 获取数据集标签
    y = iris_data["Label"].values   # 返回的是numpy类型的数据

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

    model_dict = {"KNN" : (KNeighborsClassifier(), {"n_neighbors" : [5, 15, 25], "p" : [1, 2]}),
                    "Logistic Regression" : (LogisticRegression(), {"C" : [1e-2, 1e2]}),
                    "SVM" : (SVC(), {"C" : [1e-2, 1, 1e2]})}
    for model_name, (model, model_pra) in model_dict.items():
        # 训练模型
        clf = GridSearchCV(estimator=model, param_grid=model_pra, cv=5)
        clf.fit(X_train, y_train)

        # 找到最好的参数对应的模型
        best_model = clf.best_estimator_

        # 验证
        acc = best_model.score(X_test, y_test)
        print("{}的模型预测准确率:{:.2f}%".format(model_name, acc * 100))
        print("{}的模型的最优参数:{}%".format(model_name, clf.best_params_))

if __name__ == "__main__":
    main()
