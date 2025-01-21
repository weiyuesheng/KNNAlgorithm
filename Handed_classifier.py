import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean  # 欧式距离
import numpy as np
"""
    任务：鸢尾花识别
"""

DATA_file = "Iris_Data.csv"

SPECIES = ["Iris-setosa",  # 山鸢尾
            "Iris-versicolor",  # 变色鸢尾
              "Iris-virginica"] # 维吉尼亚鸢尾 

FEAT_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def get_pred_label(test_sample_feat, train_data):
    """
        近朱者赤 找最近距离的训练样本将其标签作为预测样本的标签
    """
    dis_list = []
    for idx, row in train_data.iterrows():
        # 训练样本特征
        train_sample_feat = row[FEAT_COLS].values

        # 计算距离
        dis = euclidean(test_sample_feat, train_sample_feat)
        dis_list.append(dis)
    
    # 找最小值的位置
    position = np.argmin(dis_list)
    return train_data.iloc[position]["species"]



def main():
    """
        主函数
    """
    # 读取数据集
    iris_data = pd.read_csv(DATA_file, index_col="Id")  # 返回值为DataFrame类型数据结构

    # 划分数据集  用于将数据集拆分为训练集和测试集
    train_data, test_data = train_test_split(iris_data, test_size=1/3, random_state=12)  # 1/3 为测试集， 2/3为训练集。
    # print(train_data)
    # print(test_data)

    # 正确预测个数
    acc_count = 0

    # 分类器 测试集
    for idx, row in test_data.iterrows():
        # 测试样本的特征  得到一个numpy对象，值为数据的四个特征。
        test_sample_feat = row[FEAT_COLS].values    # row 是该行的数据，是一个pandas.series类型数据

        # 得到预测标签
        pred_label = get_pred_label(test_sample_feat, train_data)  # 测试样本数据和 训练数据

        # 真是标签
        true_label = row["species"]
        # print("样本{}真实标签{}, 预测标签{}".format(idx, true_label, pred_label))
        if pred_label == true_label:
            acc_count += 1
    
    # 准确率
    accuracy = acc_count / test_data.shape[0]  # 返回dataframe数据的维度（行数和列数，式一个列表）
    print("准确率为{}".format(accuracy * 100))


if __name__ == '__main__':
    main()




 