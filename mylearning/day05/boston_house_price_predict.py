from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def mylinear():
    """
    线性回归直接预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_test)

    # 进行标准化处理
    # 特征值标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值标准化
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimator估计器预测

    # 正规方程方式优化预测
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("损失最小时的权重为：\n", lr.coef_)
    # 预测测试集的房子价格
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("正规方程测试集里每个房子的预测价格：\n", y_lr_predict)
    print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
    # 保存训练好的模型
    joblib.dump(lr, "./temp_model/lr_model.pkl")
    # 加载训练好的模型
    lr_model = joblib.load("./temp_model/lr_model.pkl")
    y_predict = std_y.inverse_transform(lr_model.predict(x_test))
    print("保存的模型：\t", y_predict)

    # 梯度下降法优化预测
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print("损失最小时的权重为：\n", sgd.coef_)
    # 预测测试集的房子价格
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("梯度下降法测试集里每个房子的预测价格：\n", y_sgd_predict)
    print("梯度下降法的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    # 岭回归进行房价预测
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print("损失最小时的权重为：\n", rd.coef_)
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    print("岭回归测试集里每个房子的预测价格：\n", y_sgd_predict)
    print("岭回归的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    return None


def logistic():
    """
    逻辑回归做二分类癌症预测（根据细胞的属性特征）
    :return:None
    """
    # 构造列标签名
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']

    # 读取数据
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    data = pd.read_csv(data_url, names=column)  # "cancer.csv"
    print(data)

    # 缺失值处理
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna(how='any')

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 标准化处理
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(solver='liblinear', C=1.0)
    lg.fit(x_train, y_train)
    y_predict = lg.predict(x_test)
    print(lg.coef_)
    print("准确率：", lg.score(x_test, y_test))
    print("召回率：", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))

    return None


if __name__ == "__main__":
    # mylinear()
    logistic()
