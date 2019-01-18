# encoding=utf-8
from __future__ import print_function, unicode_literals
from optparse import OptionParser
import jieba.posseg as pseg
import jieba.analyse
import jieba
import sys
sys.path.append("../")

def jieba_test():
    """

    :return:
    """
    jieba.load_userdict("./dict/user_dict.txt")

    jieba.add_word('石墨烯')
    jieba.add_word('凱特琳')
    jieba.del_word('自定义词')

    test_sent = (
        "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
        "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
        "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
    )
    words = jieba.cut(test_sent)
    print('/'.join(words))

    print("=" * 40)

    result = pseg.cut(test_sent)

    for w in result:
        print(w.word, "/", w.flag, ", ", end=' ')

    print("\n" + "=" * 40)

    terms = jieba.cut('easy_install is great')
    print('/'.join(terms))
    terms = jieba.cut('python 的正则表达式是好用的')
    print('/'.join(terms))

    print("=" * 40)
    # test frequency tune
    testlist = [
        ('今天天气不错', ('今天', '天气')),
        ('如果放到post中将出错。', ('中', '将')),
        ('我们中出了一个叛徒', ('中', '出')),
    ]

    for sent, seg in testlist:
        print('/'.join(jieba.cut(sent, HMM=False)))
        word = ''.join(seg)
        print('%s Before: %s, After: %s' % (word, jieba.get_FREQ(word), jieba.suggest_freq(seg, True)))
        print('/'.join(jieba.cut(sent, HMM=False)))
        print("-" * 40)

    return None


def extract_tags():
    """

    :return:
    """
    USAGE = "usage:    python extract_tags.py [file name] -k [top k]"
    parser = OptionParser(USAGE)
    parser.add_option("-k", dest="topK")
    opt, args = parser.parse_args()

    if len(args) < 1:
        print(USAGE)
        sys.exit(1)

    file_name = args[0]

    if opt.topK is None:
        topK = 10
    else:
        topK = int(opt.topK)

    content = open(file_name, 'rb').read()

    tags = jieba.analyse.extract_tags(content, topK=topK)

    print(",".join(tags))

    return None


def extract_tags_idfpath():
    """

    :return:
    """
    USAGE = "usage:    python extract_tags_idfpath.py [file name] -k [top k]"

    parser = OptionParser(USAGE)
    parser.add_option("-k", dest="topK")
    opt, args = parser.parse_args()

    if len(args) < 1:
        print(USAGE)
        sys.exit(1)

    file_name = args[0]

    if opt.topK is None:
        topK = 10
    else:
        topK = int(opt.topK)

    content = open(file_name, 'rb').read()

    jieba.analyse.set_idf_path("./dict/idf.txt.big");

    tags = jieba.analyse.extract_tags(content, topK=topK)

    print(",".join(tags))

    return None


def extract_tags_stop_words():
    """

    :return:
    """
    USAGE = "usage:    python extract_tags_stop_words.py [file name] -k [top k]"

    parser = OptionParser(USAGE)
    parser.add_option("-k", dest="topK")
    opt, args = parser.parse_args()

    if len(args) < 1:
        print(USAGE)
        sys.exit(1)

    file_name = args[0]

    if opt.topK is None:
        topK = 10
    else:
        topK = int(opt.topK)

    content = open(file_name, 'rb').read()

    jieba.analyse.set_stop_words("./dict/stop_words.txt")
    jieba.analyse.set_idf_path("./dict/idf.txt.big");

    tags = jieba.analyse.extract_tags(content, topK=topK)

    print(",".join(tags))

    return None


def extract_tags_with_weight():
    """

    :return:
    """
    USAGE = "usage:    python extract_tags_with_weight.py [file name] -k [top k] -w [with weight=1 or 0]"

    parser = OptionParser(USAGE)
    parser.add_option("-k", dest="topK")
    parser.add_option("-w", dest="withWeight")
    opt, args = parser.parse_args()

    if len(args) < 1:
        print(USAGE)
        sys.exit(1)

    file_name = args[0]

    if opt.topK is None:
        topK = 10
    else:
        topK = int(opt.topK)

    if opt.withWeight is None:
        withWeight = False
    else:
        if int(opt.withWeight) is 1:
            withWeight = True
        else:
            withWeight = False

    content = open(file_name, 'rb').read()

    tags = jieba.analyse.extract_tags(content, topK=topK, withWeight=withWeight)

    if withWeight is True:
        for tag in tags:
            print("tag: %s\t\t weight: %f" % (tag[0], tag[1]))
    else:
        print(",".join(tags))

    return None

def test_file():
    """

    :return:
    """
    import time
    sys.path.append("../../")

    jieba.enable_parallel()

    url = sys.argv[1]
    content = open(url, "rb").read()
    t1 = time.time()
    words = "/ ".join(jieba.cut(content))

    t2 = time.time()
    tm_cost = t2 - t1

    log_f = open("1.log", "wb")
    log_f.write(words.encode('utf-8'))

    print('speed %s bytes/second' % (len(content) / tm_cost))

    return None

if __name__ == "__main__":
    jieba_test()
    # extract_tags()
    # extract_tags_idfpath()
    # extract_tags_stop_words()
    # extract_tags_with_weight()
    # test_file()
