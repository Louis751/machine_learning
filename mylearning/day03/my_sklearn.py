from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy as np
import jieba

def dictvec():
    """
    字典数据抽取
    :return: None
    """

    # 实例化
    dict = DictVectorizer()

    X = [{'city':'北京','temperature':37},{'city':'上海','temperature':32},{'city':'广州','temperature':39}]
    # 调用fit_transform得到sparse矩阵
    data = dict.fit_transform(X)
    data1 = dict.transform(X)
    # 得到类别名称
    print(dict.get_feature_names())
    # 得到转换之前数据格式，此处为X
    print(dict.inverse_transform(data))
    print(data)
    print("*"*100)
    print(data1)
    return None


def countvec():
    """
    对文本进行特征值化
    :return: None
    """
    cv = CountVectorizer()

    X = ["life is short,i like python","life is long,i dislike python"]
    data = cv.fit_transform(X)
    print(cv.get_feature_names())
    print(data.toarray())
    return None


def cutwords():
    """
    jieba分词
    :return: None
    """
    con1 = jieba.cut("腾讯QQ黄钻三个月QQ黄钻3个月季卡官方自动充值可查时间可续费")

    # 转换成list
    content1 = list(con1)

    # 把列表转换成字符串
    c1 = ' '.join(content1)
    return c1


def cnvec():
    """
    中文特征值化
    :return: None
    """
    c1=cutwords()
    print(c1)

    cv = CountVectorizer()

    data = cv.fit_transform(c1)
    print(cv.get_feature_names())
    print(data.toarray())
    print(data)
    return None


def tfidvec():
    """
    中文特征值化，考虑词频，逆文本频率指数
    :return: None
    """
    c1 = cutwords()
    print(c1)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1])
    print(type([c1,c1]))
    print(tf.get_feature_names())
    print(data.toarray())
    print("*"*100)
    print([c1])
    return None


def minmaxscal():
    """
    归一化处理
    :return: None
    """
    mms = MinMaxScaler()

    data = mms.fit_transform([[90,2,75,25],[60,4,70,20],[75,6,60,30]])
    print(data)
    return None


def stdscal():
    """
    标准化缩放
    :return: None
    """
    std = StandardScaler()
    data = std.fit_transform(([[90,2,75,25],[60,4,70,20],[75,6,60,30]]))
    print(data)
    return None


def im():
    """
    缺失值处理
    :return: None
    """
    # NaN,nan
    im = Imputer(missing_values='NaN',strategy='mean',axis=0)
    data = im.fit_transform([[1,2],[np.nan,3],[7,6]])
    print(data)
    return None


def varthreshold():
    """
    数据降维：特征值选择—删除低方差特征
    :return: None
    """
    var = VarianceThreshold()

    X = [[0,2,0,3],
         [0,1,4,3],
         [0,1,1,3]]
    data = var.fit_transform(X)
    print(X)
    print(data)

def pca():
    """
    主成分分析进行特征降维
    :return: None
    """
    pca = PCA(n_components=0.9)

    X = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    data = pca.fit_transform(X)
    print(data)
    return None


if __name__ == "__main__":
    # dictvec()    # 字典数据抽取
    # countvec()    # 对文本进行特征值化
    # cnvec()    # 对中文字体特征化
    tfidvec()    # 中文特征值化，考虑词频，逆文本频率指数
    # minmaxscal()    # 归一化处理
    # stdscal()    # 标准化处理
    # im()    # 缺失值填补
    # varthreshold()
    # pca()

