# 导入包
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np


# # 特征抽取
# # 实例化CountVectorizer
#
# vector = CountVectorizer()
#
# # 调用fit_transform输入并转换数据
#
# res = vector.fit_transform(["life is short,i like python","life is too long,i dislike python"])
#
# # 打印结果
# print(vector.get_feature_names())
#
# print(res.toarray())


def dictvec():
    """
    字典特征抽取
    :return: None
    """
    # 实例化
    dict = DictVectorizer(sparse=False)

    # 调用fit_transform
    data = dict.fit_transform([{'city': '北京','temperature':100},{'city': '上海','temperature':60},{'city': '深圳','temperature':30}])

    # 打印处理之后列名称
    print(dict.get_feature_names())
    print(data)

    return None


def countvec():
    """
    文本特征抽取
    :return: None
    """
    cv = CountVectorizer()

    data = cv.fit_transform(["life is short,i i like python", "life is too too long,i dislike python"])

    print(cv.get_feature_names())

    print(data.toarray())

    return None


def cutword():

    # 进行分词
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 处理分词结果
    content1 = []
    content2 = []
    content3 = []

    for word in con1:
        content1.append(word)
    for word in con2:
        content2.append(word)
    for word in con3:
        content3.append(word)

    # 空格隔开的字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def tfidfvec():
    """
    中文文本tfidf特征抽取
    :return: None
    """
    # 分词处理
    c1, c2, c3 = cutword()

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())

    print(data.toarray())

    return None


def mm():
    """
    归一化处理
    :return:
    """
    mm = MinMaxScaler(feature_range=(0, 1))

    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)

    return None


def stand():
    """
    标准化处理
    :return:None
    """
    std = StandardScaler()

    data = std.fit_transform([[ 1., -1., 3.],[ 2., 4., 2.],[ 4., 6., -1.]])

    print(data)

    return None


def im():
    """
    处理缺失值
    :return:None
    """
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)
    return None


def var():
    """
    删除地方差特征
    :return: None
    """
    var = VarianceThreshold(threshold=0.4)

    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    print(data)

    return None


def pca():
    """
    对数据进行主成分分析
    :return: None
    """
    pc = PCA(n_components=0.3)

    data = pc.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])

    print(data)

    return None


if __name__ == "__main__":
    pca()



# 数据集构成： 特征值 + 目标值

# 特征工程
# 1、 特征抽取
# 字典数据：one-hot编码     特征当中存在类别的数据

# 文本数据：建立词的列表（所有的词（重复的只当做一次））
#         CountVectorizer：每篇文档当中词的数量统计（对着词的列表）
#         TfidfVectorizer(常用)：词的重要性（越大越好） tf    *   idf逆文档频率log(总文档数/该词出现过的文档数量)

# 图片数据


# 2、 特征预处理
# 数值型：
# 防止某一个特征在计算的时候对最终结果影响特别大

# 归一化：所有数据默认转换成0,1之间，
# 缺点：对于异常数据处理不好（鲁棒性差）

# 标准化： 平均值，方差（常用）
# 适合处理异常数据


# 3、 数据的降维：减少特征的数量

# 特征选择：filter:删除地方差特征
# 方差大小意义：小：数据集中 大：数据离散
# embedded:正则化，决策树

# 主成分分析（公式了解）
# PCA：针对数据的特征数量非常大的时候，去进行简化数据集（根据你的模型的结果， 看运算的时间（性能））

# 监督学习：特征值+目标值
# 分类
# 回归
# 非监督学习：特征值
# 聚类


































