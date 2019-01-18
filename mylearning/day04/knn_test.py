from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import jieba

def knncls():
    """
    K-近邻算法
    :return: None
    """
    # 读取数据
    data = pd.read_csv("./data/train.tsv",sep="\t")

    # 处理数据
    pd.to_datetime()

    # 构造一些特征

    # 取出数据中的特征值和特征目标

    # 进行数据分割：训练集和测试集

    # 特征工程（标准化）

    # 对训练集和测试集的特征值进行标准化

    # 进行算法流程

    # fit,predict,score

    # 得出预测结果



    return None

if __name__ == "__main__":
    knncls()