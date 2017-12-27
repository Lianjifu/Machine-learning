from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.externals import joblib

import pandas as pd
import numpy as np


def mylinear():
    """
    线性回归实现房价预测
    :return: None
    """
    # 获取数据集，进行分割
    lb = load_boston()

    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # 进行标准化, 特征值进行标准化，  目标值需要进行标准化的，值的大小不一样 （重点）
    std_x = StandardScaler()

    # [[0.1,0.02,3,4,5,6,7,8,9,0,0,0]]                  [[234]]
    x_train = std_x.fit_transform(x_train)

    x_test = std_x.transform(x_test)

    std_y = StandardScaler()

    y_train = std_y.fit_transform(y_train)

    y_test = std_y.transform(y_test)

    # 进行预测
    # 线性回归之正规方程预测
    lr = LinearRegression()

    # 进行预测，训练数据， 正规方程求出w值
    lr.fit(x_train, y_train)

    # 已经可以进行保存模型
    joblib.dump(lr, "./tmp/test.pkl")

    # # 转换标准化之前的值
    # lr_predict = std_y.inverse_transform(lr.predict(x_test))
    #
    # # 打印权重系数
    # print("正规方程得出的系数：", lr.coef_)
    #
    # print("正规方程预测结果为：", lr_predict)
    #
    # print("正规方程的均方误差为：", mean_squared_error(std_y.inverse_transform(y_test), lr_predict))

    # 直接利用加载之后的模型预测
    model = joblib.load("./tmp/test.pkl")

    print(std_y.inverse_transform(model.predict(x_test)))

    # # 线性回归之梯度下降
    # sgd = SGDRegressor()
    #
    # sgd.fit(x_train, y_train)
    #
    # sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    #
    # # 打印权重系数
    # print("梯度下降得出的系数：", sgd.coef_)
    #
    # print("梯度下降预测结果为：", sgd_predict)
    #
    # print("梯度下降的均方误差为：", mean_squared_error(std_y.inverse_transform(y_test), sgd_predict))
    #
    # # 使用岭回归进行预测
    # rd = Ridge(alpha=1.0)
    #
    # rd.fit(x_train, y_train)
    #
    # rd_predict = std_y.inverse_transform(rd.predict(x_test))
    #
    # # 打印权重系数
    # print("岭回归得出的系数：", rd.coef_)
    #
    # print("岭回归预测结果为：", rd_predict)
    #
    # print("岭回归的均方误差为：", mean_squared_error(std_y.inverse_transform(y_test), rd_predict))

    return None


def logistic():
    """
    逻辑回归预测癌症
    :return: None
    """
    # 构造列名字，建立索引
    column_names = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

    # 获取数据，处理数据
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column_names)

    print(data)

    # 处理缺失值
    data = data.replace(to_replace='?', value=np.nan)

    # 删除空值
    data = data.dropna()

    print(data)

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25)

    # 标准化处理
    std = StandardScaler()

    x_train = std.fit_transform(x_train)

    x_test = std.transform(x_test)

    # 进行预测
    lr = LogisticRegression()

    lr.fit(x_train, y_train)

    print(lr.coef_)

    lr_predict = lr.predict(x_test)

    print("准确率：", lr.score(x_test, y_test))

    print("精确率和召回率：", classification_report(y_test, lr_predict, labels=[2, 4], target_names=["良性", "恶性"]))

    return None

if __name__ == "__main__":
    logistic()



# 前三天内容总结


# 一、特征工程
# 1、特征抽取（自然语言处理，文本分类）
# 字典特征抽取：特征值里含有类别的特征进行转换one-hot
# 文本特征抽取：统计词频，    tfidf：词的重要性    tf:词频     idf:逆文档频率log(总文档数/该词出现的文档数量)

# 2、特征处理
# 数值型数据：
# 归一化：最大值，最小值 每一列 ， 容易受异常点影响
# 标准化：平均值，方差    使用标准化的算法：K-近邻算法，线性回归，岭回归，SGDRegressor, 逻辑回归

# 类别：one-hot

# 日期：分割日期

# 缺失值：填补，删除

# 3、特征降维
# 特征选择：
# 过滤式：删除低方差特征(方差的大小，反应数据的什么情况)
# 包裹式：正则化 （消除了某些（高次项）特征的重要性），决策树:（根据信息增益等去进行分类，相当于特征的筛选）

# 主成分分析(PCA): 当特征数量非常大的时候才会去使用

# 二、算法
# 监督学习：特征值+ 目标值
# 1、分类：目标值是离散型

# k-近邻算法（掌握）：基本上很少使用   k值超参数：需要调优
# 原理：欧式距离计算最近的样本
# 优缺点：K值取值问题，性能问题

# 朴素贝叶斯算法（重点）  应用：文本分类，垃圾邮件，情感分析等等   "朴素"：特征之间独立    生成模型
# 原理：联合概率，条件概率， 贝叶斯公式
# 特点：需要求的先验概率
# 优缺点：条件独立的限制， 准确率高，发源于古典数学理论

# 决策树与随机森林（重点）：（任何场景都可以使用，效果较好）
# 原理：信息熵，信息增益 （信息是和不确定性相关
# 优缺点：决策树：容易造成过拟合
# 随机森林： 随机又放回的抽样(bootstrap抽样)  是一种集成学习方法：建立多个分类器，投票抉择， 需要调优树木的个数

# 逻辑回归（重点））：需要一些分类概率的场景， 癌症等等,算法内部具有参数：w,b， 正则化力度需要调优
# 特点：只用在二分类问题 ， sigmoid函数：将一个连续型的值转换成[0, 1]可以理解成概率，输入：线性回归的值

# 分类评估：准确率，     精确率，召回率（混淆矩阵）


# 2、回归：目标值是连续型
# 算法 +  策略 + 优化

# 线性回归：范围：数据可以是线性的也可以是非线性
# 特点：w1x1+w2x2+w3x3+....= y_predict
# 策略：（损失函数）误差大小：最小二乘法:预测值与真实值的平方和
# 优化：正规方程， 梯度下降（重点）：指定学习率（调优），方向

# SGD：使用于大数据集（10万样本以上）

# 岭回归：解决线性回归在拟合非线性数据的时候出现过拟合
# 解决办法：L2正则化，使得权重趋向于0（重点：正则化的力度）

# 回归评估：均方误差


# 模型的选择与调优：
# 交叉验证
# 网格搜索


# 非监督学习：只有特征值
# k-means聚类：主要是对用户聚类， 一般做在分类之前
# 聚类的性能评估：轮廓系数：[-1, 1]之间，趋近于1更好


# 机器学习：主要应用在传统领域：金融，电力，电商，医疗。。。。。   （数据挖掘工程，机器学习工程师）
# scikit-learn:含有机器学习的算法， 不含有深度学习的算法，

# 深度学习：自然语言处理，图像识别
# tensorflow:有神经网络，能够使用设备（cpu,GPu）


