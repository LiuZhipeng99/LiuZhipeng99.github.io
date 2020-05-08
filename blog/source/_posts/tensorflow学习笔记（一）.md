---
title: Tensorflow学习笔记(一）
tags: [Tensorflow,笔记]
categories: Tensorflow
date: 2020-05-08 17:37:31
---

  学习tensorflow的经验和总结
<!--more-->

> 想学习tf的人，或许啃完了网上的机器学习课程（吴恩达），或许正在上学校里的机器学习课程，抑或是只解皮毛而想快速入门。
> 学前建议：
>  1.对于没有任何基础的兄台还是老老实实的回去看看高数，不然走不下去。
>  2.对于正在学机器学习的人来说，你或许已经了解梯度下降，知道若干种损失函数激活函数，但建议先自己动手做网络构建走基础路线。例如吴恩达课程和本人[用的](https://github.com/microsoft/ai-edu/tree/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B)（基于python）。
>  3.对于基础扎实的同学来说，tf将会是你大有作为的舞台。尽管去看那些网上丰富的文档相信你不会迷路。

#### 关于搭建环境

  如果是初学TF建议使用cpu版本，这样省去诸多麻烦。例如gpu版本需要电脑gpu支持（nividia，需要驱动），并且安装cuDnn和cudatoolkit（需要和tensorflow版本对应），另外python高版本还未适配tensorflow所以不能安装过高版本，此时anaconda闪亮登场，它不仅能虚拟环境让py、包等与系统隔离开，指定python版本和自动下相应cuda搭配。当然如果用cpu版本无需考虑。

  ```python
import tensorflow as tf
print(tf.__version__) #打印即成功
  ```

#### 关于keras和tf.keras

  现在都是直接下载tensorflow里面内置了keras，官方也推荐使用后者，相当于后者是将其收纳了，实际上关系是：tensorflow作为其计算工具，后者是tensorflow的高级api，简单来说就是协作关系，keras是只笔，tensorflow是张纸。

#### 初识Tensorflow

> 前馈神经网络：**分为多层, 相邻层之间全连接, 不存在同层连接与跨层连接**. 整个网络中无反馈，可用一个有向无环图表示.用标准bp算法训练（基于SDG）。
>
> 以下从特性到建立模型和训练。从代码实战开始

1. 了解其特效和基础--自动求导机制

```python
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
  y = 3*tf.square(x)
  z = y+3*x
z_grad = tape.gradient(z,x)
print(y,z,z_grad)
#利用其求偏导
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print(L, w_grad, b_grad)

# 线性回归示例：生成data
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables=[a,b]
num_epoch = 10000
opt = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
  with tf.GradientTape() as tape:
      y_pred = a * X + b
      loss = tf.reduce_sum(tf.square(y_pred-y))
  grads = tape.gradient(loss,variables)
  opt.apply_gradients(grads_and_vars=zip(grads,variables))
print(a,b)
```

2. 创建自己的模型作用如上面的运算和variable（下面的方法都是各类框架通用，没有使用高级api，都是自定义网络）

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output

    # 还可以添加自定义的方法
#由于继承自model其有variable这一属性获取所有变量
#将上例的y=ax+b模型化
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units = 1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()

            )
    def call(self,input):
        output = self.dense(input)
        return output

model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print((model.variables),model.call(X))
```

3. 编写个MLP即多层全连接的前馈神经网络，所看教程最后的数字识别

```python
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist #下面的load函数第一次会download  
        (self.train_data,self.train_label),(self.test_data,self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)#返回列表
        return self.train_data[index, :], self.train_label[index]
data = MNISTLoader()
# print(data.train_data[0][],data.train_label) #traindata=(6000,28,28)
# test = np.array([[1,2],[2,3]])
# print(np.expand_dims(test/5,axis=-1)) 这种操作结果增加一维度但实际不变，增加的一维就是替换掉原本

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__() 
        self.flatten = tf.keras.layers.Flatten() # Flatten层将除第一维（batch_size）以外的维度展平 用于输入层
        self.dense1 = tf.keras.layers.Dense(units=256,activation=tf.nn.relu)#初始accuracy: 0.928500，loss=0.15
        self.dense2 = tf.keras.layers.Dense(units=64,activation=tf.nn.relu)#加了这层后accuracy: 0.935200，loss降到了0.08，如果是sigmiod比初始还差
        self.dense3 = tf.keras.layers.Dense(units=10)
    def call(self,inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        output = tf.nn.softmax(x)
        return output 
```

4. mnist再用卷积神经网络来分类除了全连接层还有池化卷积，用时翻倍但准确率也高

```python
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output



#训练Mnist的方法，传入构造的模型
def trainMLP(model):    
    batch_size = 100
    num_epochs = 3
    learning_rate = 0.008 #调试各个参数提高准确率,在翻倍epoch后达到了97
    # model = Model()
    data_loader = MNISTLoader()
    opt = tf.keras.optimizers.Adam(learning_rate)
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        opt.apply_gradients(grads_and_vars=zip(grads, model.variables))
    def metrics():
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        num_batches = int(data_loader.num_test_data // batch_size)
        for batch_index in range(num_batches):
            start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
            y_pred = model.predict(data_loader.test_data[start_index: end_index])
            sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
        print("test accuracy: %f" % sparse_categorical_accuracy.result())
    metrics()        
# traincnn()
# print(model.variables)在训练前为空列表
```
5. 循环神经网络：文本处理，task以RNN生成尼采风格文本。
```python
class TextLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt',
            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index+seq_length])
            next_char.append(self.text[index+seq_length])
        return np.array(seq), np.array(next_char)       # [batch_size, seq_length], [num_batch]
class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        inputs = tf.one_hot(inputs, depth=self.num_chars)       # [batch_size, seq_length, num_chars]
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)
        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)
    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs, from_logits=True)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                         for i in range(batch_size.numpy())])

def trainRnn():
    num_batches = 1000
    seq_length = 40
    batch_size = 50
    learning_rate = 1e-3
    data_loader = TextLoader()
    model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(seq_length, batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    #雪球式预测下个char
    X_, _ = data_loader.get_batch(seq_length, 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        X = X_
        print("diversity %f:" % diversity)
        for t in range(400):
            y_pred = model.predict(X, diversity)
            print(data_loader.indices_char[y_pred[0]], end='', flush=True)
            X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
        print("\n")
trainRnn()
```

6. 利用keras的高级api建立模型 sequential通过向 tf.keras.models.Sequential() 提供一个层的列表，就能快速地建立一个 tf.keras.Model 模型并返回：

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation=tf.nn.sigmoid),
    # tf.keras.layers.Dense(64,activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
    ])
#Keras 提供了 Functional API，帮助我们建立更为复杂的模型，例如多输入 / 输出或存在参数共享的模型
def FunctionalCreate():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(units=10)(x)
    outputs = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
# trainMLP(MLP())
def apitrain():
    model = MLP()
    #使用kerasmodel的三个方法训练和评估
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        ,loss=tf.keras.losses.sparse_categorical_crossentropy
        ,metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    #complle接受 优化器，损失函数，评估指标都从对应的类里选


    model.fit(MNISTLoader().train_data,MNISTLoader().train_label,epochs=3,batch_size=100)
    #返回列表包含最终loss和accuracy 在这里测试之前的模型准确率都上升了
    print('\n',model.evaluate(MNISTLoader().test_data,MNISTLoader().test_label))
apitrain()
```