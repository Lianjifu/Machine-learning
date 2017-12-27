import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 完成一个同步的过程
# 1、定义好队列大小，数据类型
# Q = tf.FIFOQueue(3, tf.float32)
#
# # 2、放入一些数据，定义出队列， +1， 入队列操作
# en_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])
#
# out_q = Q.dequeue()
#
# data = out_q + 1
#
# en_q = Q.enqueue(data)
#
# # 3、会话运行
# with tf.Session() as sess:
#     # 运行入队列的数据
#     sess.run(en_many)
#
#     # 处理数据
#     for i in range(10):
#         sess.run(en_q)
#
#     # 等待前面处理完成，才取数据训练
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))


# 模拟数据处理与读取的异步过程

# 1、定义队列的大小
# Q = tf.FIFOQueue(1000, tf.float32)
#
# # 2、定义子线程的一些数据准备和处理操作
# # 定义变量
# var = tf.Variable(0.0)
#
# encrement_op = tf.assign_add(var, tf.constant(1.0), )
#
# # 将数据放入队列
# en_q = Q.enqueue(encrement_op)
#
# # 3、定义多少个子线程去完成操作
# qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)
#
# # 定义初始化变量的op
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 初始化变量
#     sess.run(init_op)
#
#     # 创建协调器，回收子线程
#     coord = tf.train.Coordinator()
#
#     # 开启子线程执行操作
#     threads = qr.create_threads(sess, coord=coord, start=True)
#
#     # 主线程立刻执行操作
#     for i in range(200):
#         print(sess.run(Q.dequeue()))
#
#     # 主线程已经结束，子线程并没有结束
#     coord.request_stop()
#
#     # 回收子线程
#     coord.join(threads)


# 1、找到所有文件， 构造列表， 路径+名字
# 2、构造文件的队列
# 3、构造阅读器进行读取
# 4、对内容进行解码
# 5、会话运行结果

# def csvread(file_list):
#     """
#     读取CSV文件数据到张量
#     :param file_list: 路径+文件名字的列表
#     :return:
#     """
#     # 构造文件队列
#     file_queue = tf.train.string_input_producer(file_list)
#
#     # 构造阅读器读取数据
#     reader = tf.TextLineReader()
#
#     # key:文件名字，value:每一行的内容
#     key, value = reader.read(file_queue)
#
#     print(value)
#
#     # 指定CSV每行中每列的类型和默认值 [[2],[3.0]]
#     records = [["None"], ["None"]]
#
#     # 进行解码
#     feature1, feature2 = tf.decode_csv(value, record_defaults=records)
#
#     # 进行批处理
#     # batch_size:每批次去的数据数量
#     feature1_batch, feature2_batch = tf.train.batch([feature1, feature2], batch_size=9, num_threads=1, capacity=9)
#
#     return feature1_batch, feature2_batch
#
#
# if __name__ == "__main__":
#     # 找到文件
#     filename = os.listdir("./data/csvdata/")
#
#     # 加上文件的路径
#     file_list = [os.path.join("./data/csvdata/", file) for file in filename]
#
#     f1, f2 = csvread(file_list)
#
#     # 开启会话
#     with tf.Session() as sess:
#
#         # 创建线程协调器
#         coord = tf.train.Coordinator()
#
#         # 开启线程
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         # 打印读取的内容
#         print(sess.run([f1, f2]))
#
#         # 回收线程
#         coord.request_stop()
#
#         coord.join(threads)

#
# def picread(file_list):
#     """
#     狗图片的读取
#     :param file_list: 路径+名字的列表
#     :return:
#     """
#     # 构造队列
#     file_queue = tf.train.string_input_producer(file_list)
#
#     # 图片读取器
#     reader = tf.WholeFileReader()
#
#     key, value = reader.read(file_queue)
#
#     # 对一张图片解码
#     image = tf.image.decode_jpeg(value)
#
#     print(image)
#
#     # 进行图片大小缩放
#     image_resize = tf.image.resize_images(image, [200, 200])
#
#     print(image_resize)
#
#     # 形状没有固定， 通过静态形状修改的方式 (batch要求)
#     image_resize.set_shape([200, 200, 3])
#
#     print(image_resize)
#
#     # 进行批处理
#     image_batch = tf.train.batch([image_resize], batch_size=100, num_threads=1, capacity=100)
#
#     print(image_batch)
#
#     return image_batch
#
#
# if __name__ == "__main__":
#     # 找到文件
#     filename = os.listdir("./data/dog/")
#
#     # 加上文件的路径
#     file_list = [os.path.join("./data/dog/", file) for file in filename]
#
#     image_batch = picread(file_list)
#
#     # 开启会话
#     with tf.Session() as sess:
#
#         # 创建线程协调器
#         coord = tf.train.Coordinator()
#
#         # 开启线程
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         # 打印读取的内容
#         print(sess.run(image_batch))
#
#         # 回收线程
#         coord.request_stop()
#
#         coord.join(threads)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cifar_dir", "./data/cifar10/cifar-10-batches-bin/", "cifar数据目录")
tf.app.flags.DEFINE_string("tfrecord_dir", "./tmp/cifar10.tfrecords", "cifar tfrecords文件路径")

# 读取二进制文件

class CifarRead(object):

    def __init__(self, file_list):
        """
        初始化图片信息
        :param file_list: 文件列表
        """
        # 文件路径+名字列表
        self.file_list = file_list

        # 定义图片的属性
        self.height = 32
        self.width = 32
        self.channel = 3
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        # 每一个图片样本读取的字节数
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        """
        读取二进制文件，转换成张量
        :return:
        """
        # 构造文件的队列
        file_queue = tf.train.string_input_producer(self.file_list)

        # 构造二进制阅读器，读取指定字节的内容
        reader = tf.FixedLengthRecordReader(self.bytes)

        key, value = reader.read(file_queue)

        # 进行二进制格式解码
        label_image = tf.decode_raw(value, tf.uint8)

        print(label_image)

        # 进行数据分割，标签值 + 特征值
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)

        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

        print(image, label)
        # 对图片形状进行处理
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        print(image_reshape)

        # 进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch

    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将图片数据内容存储到tfrecords文件当中，方便以后使用训练
        :param image_batch: 特征值
        :param label_batch: 目标值
        :return: None
        """
        # 1、构造文件的存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_dir)

        # 2、写入文件，是对于每个样本写入
        for i in range(10):

            # 取出对应的每个样本的特征和标签值[10, 32, 32, 3] , [10, 1]
            image_string = image_batch[i].eval().tostring()

            # [1]   ---> 1
            label = int(label_batch[i].eval()[0])

            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

            writer.write(example.SerializeToString())

        # 结束文件
        writer.close()

        return None

    def read_from_tfrecords(self):
        """
        从tfrecords文件当中读取内容
        :return:
        """
        # 构造文件队列
        file_queue = tf.train.string_input_producer(["./tmp/cifar10.tfrecords"])

        # 构造tfrecords的阅读器
        reader = tf.TFRecordReader()

        key, value = reader.read(file_queue)

        # 解析example，返回字典数据
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })

        # feature包含图片特征值和目标值,如果类型是字节的类型，那需要解码，整型就不需要

        image = tf.decode_raw(feature["image"], tf.uint8)

        print(image)

        # 形状改变
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        label = tf.cast(feature["label"], tf.int32)

        # 批处理数据
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        print(image_batch, label_batch)
        return image_batch, label_batch


if __name__ == "__main__":
    # 找到文件
    filename = os.listdir(FLAGS.cifar_dir)

    # 加上文件的路径
    file_list = [os.path.join(FLAGS.cifar_dir, file) for file in filename if file[-3:] == "bin"]

    cr = CifarRead(file_list)

    # image_batch, label_batch = cr.read_and_decode()

    image_batch, label_batch = cr.read_from_tfrecords()

    # 开启会话
    with tf.Session() as sess:

        # 创建线程协调器
        coord = tf.train.Coordinator()

        # 开启线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 打印读取的内容
        print(sess.run([image_batch, label_batch]))

        # cr.write_to_tfrecords(image_batch, label_batch)

        # 回收线程
        coord.request_stop()

        coord.join(threads)















