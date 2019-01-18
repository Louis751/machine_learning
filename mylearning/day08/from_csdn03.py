import os

from sklearn.datasets.base import Bunch
import pickle  # 持久化类
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法包
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类


# from TF_IDF import space_path


def readbunchobj(path):
    file_obj = open(path, "rb")

    bunch = pickle.load(file_obj)

    file_obj.close()

    return bunch


def writebunchobj(path, bunchobj):
    file_obj = open(path, "wb")

    pickle.dump(bunchobj, file_obj)

    file_obj.close()


def readfile(path):
    fp = open(path, "r", encoding='gb2312', errors='ignore')

    content = fp.read()

    fp.close()

    return content


# 导入分词后的词向量bunch对象

path = "test_word_bag/test_set.dat"

bunch = readbunchobj(path)

# 停用词

stopword_path = "train_word_bag/hlt_stop_words.txt"

stpwrdlst = readfile(stopword_path).splitlines()

# 构建测试集TF-IDF向量空间

testspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})

# 导入训练集的词袋

trainbunch = readbunchobj("train_word_bag/tfidfspace.dat")

# 使用TfidfVectorizer初始化向量空间

vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5, vocabulary=trainbunch.vocabulary)

transformer = TfidfTransformer();

testspace.tdm = vectorizer.fit_transform(bunch.contents)

testspace.vocabulary = trainbunch.vocabulary

# 创建词袋的持久化

space_path = "test_word_bag/testspace.dat"

writebunchobj(space_path, testspace)


def readbunchobj(path):
    file_obj = open(path, "rb")

    bunch = pickle.load(file_obj)

    file_obj.close()

    return bunch


# 导入训练集向量空间

trainpath = "train_word_bag/tfidfspace.dat"

train_set = readbunchobj(trainpath)

# d导入测试集向量空间

testpath = "test_word_bag/testspace.dat"

test_set = readbunchobj(testpath)

# 应用贝叶斯算法

# alpha:0.001 alpha 越小，迭代次数越多，精度越高

clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

# 预测分类结果

predicted = clf.predict(test_set.tdm)

total = len(predicted);
rate = 0

for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):

    if flabel != expct_cate:
        rate += 1

        print(file_name, ": 实际类别：", flabel, "-->预测分类：", expct_cate)

# 精度

print("error_rate:", float(rate) * 100 / float(total), "%")


#评估
def metrics_result(actual,predict):

    print("精度：{0:.3f}".format(metrics.precision_score(actual,predict)))

    print("召回：{0:0.3f}".format(metrics.recall_score(actual,predict)))

    print("f1-score:{0:.3f}".format(metrics.f1_score(actual,predict)))


metrics_result(test_set.label,predicted)

