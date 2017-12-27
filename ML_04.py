import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# 实现加法运算
# 会话只能开启默认的图，可以通过指定其他的图运行

# 创建一个新的图
# g = tf.Graph()
# with g.as_default():
#
#     con = tf.constant(1.0)
#     print(con.graph)
#
#
# # 在tensorflow当中不能称之为变量
# a = tf.constant(3.0)
#
# b = tf.constant(4.0)
#
# sum = tf.add(a, b)
#
# print(sum)
#
# print(tf.get_default_graph())
#
# # 会话
# with tf.Session(graph=g) as sess:
#     print(sess.run(con))
#
#     print(a.graph)
#     print(b.graph)
#     print(sess.graph)


# 会话
# 交互式会话，有一个eval()可以使用，必须在会话上下文环境使用
# 如果普通类型和tensorflow结构进行运算，此时运算符号会被重载成一个op
# 实时训练数据的时候，提供数据使用feed_dict,实时提供数据训练可以使用这样的机制
# 如果数据的形状不固定，可以使用None指定
# 0为, () , 1维(10, ),  2维(2, 2), 3维(2, 2, 2)

# 定义Python程序
# var1 = 0.0
# var2 = 1.0
# # c = var1 + var2
#
# a = tf.constant(3.0)
#
# b = tf.constant(4.0)
#
# c = a + var1
#
# print(c)
#
# sum = tf.add(a, b)
#
# # 定义placeholder
# plt1 = tf.placeholder(tf.float32, [2, 2])
# # plt2 = tf.placeholder(tf.float32)
#
# print(plt1)
#
# # sum_plt = tf.add(plt1, plt2)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#
#     print(sess.run(plt1, feed_dict={plt1: [[1, 2], [3, 4]]}))
#
#     print(sum.op)
#     print("---------")
#     print(sum.name)
#     print("---------")
#     print(sum.shape)


# 张量的形状改变    静态形状，动态形状
# 关于静态形状修改不能跨阶数（维度）转变
# 对于初始形状没有固定的张量，可以去使用静态形状修改,一旦固定形状就不能修改

# 动态形状，单独创建对象（消耗内存）出来
# 注意：元素的胡亮一定要匹配

# plt1 = tf.placeholder(tf.float32, [None, 2])
#
# plt1.set_shape([3, 2])
#
# print(plt1.shape)
#
# # plt1.set_shape([4, 2])  # 不能静态形状修改
#
# plt_reshape = tf.reshape(plt1, [2, 3])
#
# print(plt_reshape)
#
# with tf.Session() as sess:
#     print(sess.run(plt1, feed_dict={plt1: [[1,1],[1,1],[1,1]], plt_reshape: [[1, 2, 3], [3, 4, 5]]}))

# 变量
# 能进行存储的张量variable,称之为变量
# tensorboard默认会读取最新的一次events文件
# a = tf.constant(3.0, name="x")
# b = tf.constant(4.0, name="w")
#
# var = tf.Variable(tf.random_normal([3, 4], mean=0.0, stddev=1.0))
#
# print(var)
#
# sum = tf.add(a, b)
#
# # 定义一个初始化变量的op
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 运行初始化
#     sess.run(init_op)
#
#     print(sess.run([sum, var]))
#
#     # 写入事件文件，指定图和路径
#     filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

# 自实现线性回归预测
tf.app.flags.DEFINE_integer("max_step", 100, "模型训练的步数")

FLAGS = tf.app.flags.FLAGS


def main(argv):

    with tf.variable_scope("data"):
        # 准备数据 特征值x  [100, 1]    y_true [100]
        x = tf.random_normal([100, 1], mean=1.75, stddev=1.0, name="x_data")

        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        #  建立线性回归的模型 y_predict = xw1+ bias, 要训练的参数必须得以变量去定义
        # 随机初始化权重 二维
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="w")
        bias = tf.Variable(0.0, name="b")

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        # 计算损失
        loss = tf.reduce_mean(tf.square(y_predict - y_true))

    with tf.variable_scope("optimizer"):
        # 梯度下降优化损失
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 收集变量
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weightes", weight)

    # 定义初始化变量op
    init_op = tf.global_variables_initializer()

    # 定义一个合并变量的op
    merged = tf.summary.merge_all()

    # 创建一个保存模型的saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        # 初始化的参数
        print("随机初始化的参数:权重：%f, 偏置:%f" % (weight.eval(), bias.eval()))

        filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

        # 加载模型进行训练
        if os.path.exists("./tmp/ckpt/checkpoint"):
            # 路径+模型名字
            saver.restore(sess, "./tmp/ckpt/test")

        # 循环训练线性回归模型,指定步数去训练
        for i in range(FLAGS.max_step):

            sess.run(train_op)

            print("第%d次训练参数：权重：%f, 偏置:%f" % (i, weight.eval(), bias.eval()))

            # 观察每次的值变化
            # 运行merge
            summary = sess.run(merged)

            # 添加到文件当中
            filewriter.add_summary(summary, i)

        saver.save(sess, "./tmp/ckpt/test")

    return None


if __name__ == "__main__":
    tf.app.run()
















