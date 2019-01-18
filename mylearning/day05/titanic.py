import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return: None
    """
    # 获取数据
    titan = pd.read_csv("titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan.loc[:, ['pclass', 'age', 'sex']]
    y = titan.loc[:, 'survived']

    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # （特征工程）进行处理特征->类别->one-hot编码
    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))
    print(dict.get_feature_names())

    x_test = dict.transform(x_test.to_dict(orient='records'))

    # 使用决策树进行预测
    # since = time.time()
    #
    # clf = DecisionTreeClassifier(max_depth=8)
    # clf.fit(x_train,y_train)    #
    # # 预测准确率： 0.8237082066869301
    # print("预测准确率：",clf.score(x_test,y_test))
    #
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))  # 打印出来时间

    # # 导出决策树结构图
    # feature_name = ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
    # target_name = ['survived']
    # export_graphviz(clf, out_file="./tree.dot", feature_names=feature_name,class_names=target_name)

    # 随机森林进行预测
    since = time.time()

    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    print("准确率为：", gc.score(x_test, y_test))
    print("查看现在的参数模型：", gc.best_params_)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.3f}s'.format(
        time_elapsed // 60, time_elapsed % 60))  # 打印出来时间

    """
    执行结果：
    准确率为： 0.8206686930091185
    查看现在的参数模型： {'max_depth': 5, 'n_estimators': 120}
    Training complete in 1m 37.064s
    """

    return None


if __name__ == "__main__":
    decision()
