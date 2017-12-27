from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


li = load_iris()

# print(li.data)
# print(li.target)
# print(li.DESCR)
# print(li.feature_names)
# print(li.target_names)

# 划分数据, x_train,x_test, y_train, y_test

# x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
#
# print(x_train, y_train)
# print("--------")
# print(x_test, y_test)

# 获取大数据集
# news = fetch_20newsgroups(subset='all')
#
# print(news.data)

# 获取回归数据集
# lb = load_boston()
#
# print(lb.data)
# print(lb.target)


# k-近邻算法预测入住位置

def knncls():
    """
    k-近邻算法预测入住位置
    :return: None
    """
    # 读取数据
    data = pd.read_csv("./data/FBlocation/train.csv")

    # 特征工程
    # 1、缩小数据的范围x, y
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 2、进行时间戳转换， 多构造一些特征
    time_value = pd.to_datetime(data['time'], unit='s')

    time_value = pd.DatetimeIndex(time_value)

    # 增加特征
    data['weekday'] = time_value.weekday
    data['day'] = time_value.day
    data['hour'] = time_value.hour

    # 3、删除不要的特征time
    data = data.drop(['time'], axis=1)

    # 4、目标值进行筛选，避免预测类别太多（入住位置少于多少人的地址直接忽略）
    place_count = data.groupby('place_id').count()

    tf = place_count[place_count.row_id > 3].reset_index()

    data = data[data['place_id'].isin(tf.place_id)]

    print(data)

    # 拿出数据的特征值和目标值
    y = data['place_id']

    x = data.drop(['place_id'], axis=1)

    # 进行数据集的分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行标准化
    std = StandardScaler()

    x_train = std.fit_transform(x_train)

    x_test = std.transform(x_test)

    # 算法估计器
    knn = KNeighborsClassifier()

    # knn.fit(x_train, y_train)
    #
    # # 预测结果
    # y_predict = knn.predict(x_test)
    #
    # print("预测的准确率：", knn.score(x_test, y_test))

    # 用交叉验证和网格搜索来进行预估
    param = {"n_neighbors": [3, 5, 7]}

    gc = GridSearchCV(knn, param_grid=param, cv=2)

    gc.fit(x_train, y_train)

    print("测试集的结果：", gc.score(x_test, y_test))

    # 查看超参数搜索的整个结果情况
    print("在交叉验证当中最好结果：", gc.best_score_)

    print("最好的参数模型：", gc.best_estimator_)

    print("每次交叉验证的结果：", gc.cv_results_)

    return None


def naviebayes():
    """
    朴素贝叶斯算法进行新闻的分类
    :return:
    """
    # 首先获取数据集
    news = fetch_20newsgroups(subset='all')

    # 划分数据集为训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # print("训练集的文章和文章类型：", y_train)

    # 进行特征抽取tfidf
    tf = TfidfVectorizer()

    x_train = tf.fit_transform(x_train)

    x_test = tf.transform(x_test)

    # 进行预测
    mlb = MultinomialNB(alpha=1.0)

    mlb.fit(x_train, y_train)

    y_predict = mlb.predict(x_test)

    print("预测的类别：", y_predict)

    print("真实的类别：", y_test)

    print("准确率：", mlb.score(x_test, y_test))

    # 打印评估指标
    print("每个类别精确率和召回率为：", classification_report(y_test, y_predict, target_names=news.target_names))

    return None


# 决策树预测乘客是否存活

def descsion():
    """
    预测泰坦尼克号人员存活
    :return: None
    """
    # 获取数据， 提取出特征值和目标值
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    x = titan[['pclass', 'age', 'sex']]

    y = titan[['survived']]

    # 处理缺失值
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行字典特征抽取（针对类别特征） [['1st', 24, 'female'],[]]----->[{"pclass":1st, "age":24, "sex":female},{},{},{}]
    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient="records"))

    x_test = dict.transform(x_test.to_dict(orient="records"))

    # 查看列名字
    print(dict.get_feature_names())

    # print(x_train)

    # 进行决策树预测
    # dec = DecisionTreeClassifier(max_depth=10)
    #
    # dec.fit(x_train, y_train)
    #
    # print("准确率：", dec.score(x_test, y_test))

    # 树的结构本地化保存
    # export_graphviz(dec, out_file="./tree.dot", feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # 随机森林
    rf = RandomForestClassifier(n_estimators=15, max_depth=10)

    rf.fit(x_train, y_train)

    print(rf.score(x_test, y_test))

    return None


if __name__ == "__main__":
    descsion()

