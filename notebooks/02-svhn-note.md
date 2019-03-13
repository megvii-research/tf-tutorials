## configure
用于配置工程的超参数, 对应到01-svhn工程下的common.py. 
```python
minibatch_size = 128  # the number of instances in a batch
nr_channel = 3 # the channels of image
image_shape = (32, 32) # the image shape (height, width)
nr_class = 10 # the number of classes
nr_epoch = 60 # the max epoch of training
weight_decay = 1e-10 # a strength of regularization
test_interval = 5 # test in every ${test_interval} epochs
show_interval = 10 # print a message of training in every ${show_interval} minibatchs 
```

nr_class根据分类任务的类别数进行指定, 
weight_decay指代的是 **正则化部分对整个loss的影响程度**
```python
nr_class = 10 # the number of classes
weight_decay = 1e-10 # a strength of regularization
```
----
用于决定print的频率, 单位都是epoch
```python
test_interval = 5 # test in every ${test_interval} epochs
show_interval = 10 # print a message of training in every ${show_interval} minibatchs 
```
---
## Dataset
由于deep learning是data-driven, 所以数据的处理是整个工程当中最重要的部分。01-svhn工程专门用一个dataset.py中的Dataset()进行处理。
```python
# import required modules
import os
import cv2
import numpy as np
from scipy import io as scio

class Dataset():
    dataset_path = '../../dataset/SVHN'  # a path saves dataset
    dataset_meta = {
        'train': ([os.path.join(dataset_path, 'train_32x32.mat')], 73257),
        'test': ([os.path.join(dataset_path, 'test_32x32.mat')], 26032),
    }
    
    def __init__(self, dataset_name):
        self.files, self.instances = self.dataset_meta[dataset_name]
    
    def load(self):
        '''Load dataset metas from files'''
        datas_list, labels_list = [], []
        for f in self.files:
            samples = scio.loadmat(f)
            datas_list.append(samples['X'])
            labels_list.append(samples['y'])
        self.samples = {
            'X': np.concatenate(datas_list, axis=3), # datas
            'Y': np.concatenate(labels_list, axis=0), # labels
        }
        return self

    def instance_generator(self):
        '''a generator to yield a sample'''
        for i in range(self.instances):
            img = self.samples['X'][:, :, :, i]
            label = self.samples['Y'][i, :][0]
            if label == 10:
                label = 0
            img = cv2.resize(img, image_shape)
            yield img.astype(np.float32), np.array(label, dtype=np.int32)
    
    @property
    def instances_per_epoch(self):
        return 25600 # set for a fast experiment
        #return self.instances
    
    @property
    def minibatchs_per_epoch(self):
        return 200 # set for a fast experimetn
        #return self.instances // minibatch_size
```
### Code explained
根据**数据的保存路径**来修改dataset_path
```python
dataset_path = '../../dataset/SVHN'  # a path saves dataset
```
维护数据的各个子集, 包括**它们的文件位置及数据量**
```python
 dataset_meta = {
        'train': ([os.path.join(dataset_path, 'train_32x32.mat')], 73257),
        'test': ([os.path.join(dataset_path, 'test_32x32.mat')], 26032),
    }
```
load函数作用一般就是**解压缩**, 将数据集的数据load到内存, 便于后续访问。常用的也有把所有数据(eg. 图片)的访问路径读取成一个list, 便于后面读取。
```python
    def load(self):
        '''Load dataset metas from files'''
        datas_list, labels_list = [], []
        for f in self.files:
            samples = scio.loadmat(f)
            ....
```
利用python的生成器来构造一个sample, 用于后续配合tensorflow的dataset API进行使用。sample = (data, label)
**Note**: 由于svhn数据集把0的label标记成数字10, 而tensorflow的默认实现分类任务的label都是从0开始(**svhn是matlab格式, 从1作为下标开始**)，所以要强制更改一下。 
```python
            if label == 10:
                label = 0
```
---
## Show
在编写完dataset.py的代码之后, 可以编写一个简单的测试函数来检验书写的正确性, 同时也可以看看数据集的图片内容。后续如果大家开始对数据进行augment之后, 这种方式还可以用来检验augment之后的数据情况。
```python
# show an img from dataset
%matplotlib inline
import matplotlib.pyplot as plt

ds = Dataset('train').load()
ds_gen = ds.instance_generator()

imggrid = []
for i in range(25):
    img, label = next(ds_gen) # yield a sample
    cv2.putText(img, str(label), (0, image_shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0), 2) # put a label on img
    imggrid.append(img) 

# make an img grid from an img list
imggrid = np.array(imggrid).reshape((5, 5, img.shape[0], img.shape[1], img.shape[2]))
imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((5*img.shape[0], 5*img.shape[1], 3)) 
imggrid = cv2.cvtColor(imggrid.astype('uint8'), cv2.COLOR_BGR2RGB)

# show
plt.figure()
plt.imshow(imggrid)
plt.show()
```
为了在jupyter notebook可以使用matplotlib在线实时查看生成的图片.
```python
%matplotlib inline
import matplotlib.pyplot as plt
```
用于将多张子图片整合成一张大图片
```python
imggrid = np.array(imggrid).reshape((5, 5, img.shape[0], img.shape[1], img.shape[2]))
imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((5*img.shape[0], 5*img.shape[1], 3)) 
```

由于openCV采用的是BGR格式, 而matplotlib采用的是RGB格式, 所以在利用matplotlib进行显示的时候需进行一次转换。
```python
imggrid = cv2.cvtColor(imggrid.astype('uint8'), cv2.COLOR_BGR2RGB)
```
---
## Build a graph
tensorflow属于符号式编程, 其需要先定义一个compute graph, graph描述了各个tensor之间的相互关系以及operators, 然后再将定义好的graph进行编译, 利用session进行run. 在run的过程当中, graph不会发生改变。所以, 大家一定要形成一个意识: graph的定义和计算是分隔开的。下面, 先关注how to build a graph。为了后续和我们提供的检查工具tf-model-manip.py进行配合, **强烈推荐**大家按照01-svhn的工程架构进行后续代码的书写。即在工程代码中创建一个model.py文件, 内含一个Model class, 在Model class中有一个method: build.

---
在\_\_init\_\_函数中指定param使用的初始化方式(这里使用MSRA初始化方式, 感兴趣可以查看 https://arxiv.org/pdf/1502.01852.pdf\)。简单而言, 就是一个均值为0的正态分布, 其方差由输入神经元个数决定。也可以通过修改mode来进行修改方差的计算方式。
```python
def __init__(self):
        self.weight_init = tf_contrib.layers.variance_scaling_initializer(factor=1.0,
                                mode='FAN_IN', uniform=False)
        self.bias_init = tf.zeros_initializer()
        self.reg = tf_contrib.layers.l2_regularizer(weight_decay)
```
三种mode(详细资料: [tensorflow-init](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer) )
```
  if mode='FAN_IN': # Count only number of input connections.
    n = fan_in
  elif mode='FAN_OUT': # Count only number of output connections.
    n = fan_out
  elif mode='FAN_AVG': # Average number of inputs and output connections.
    n = (fan_in + fan_out)/2.0
```

_conv_layer实现是一个默认带 BN和relu 的 conv2d. 
```python
   def _conv_layer(self, name, inp, kernel_shape, stride, padding='SAME',is_training=False):
        '''a conv layer = conv + bn + relu'''
        
        with tf.variable_scope(name) as scope:
            conv_filter = tf.get_variable(name='filter', shape=kernel_shape,
                                          initializer=self.weight_init, regularizer=self.reg)
            conv_bias = tf.get_variable(name='bias', shape=kernel_shape[-1],
                                        initializer=self.bias_init)
            x = tf.nn.conv2d(inp, conv_filter, strides=[1, stride, stride, 1],
                             padding=padding, data_format='NHWC')
            x = tf.nn.bias_add(x, conv_bias, data_format='NHWC')
            x = tf.layers.batch_normalization(x, axis=3, training=is_training)
            x = tf.nn.relu(x)
        return x
```
通过利用variable_scope形成命名空间, 实现命名的复用。 e.g. name='conv1', 则conv_filter的最后名称会变为: 'conv1/filter'
```python
        with tf.variable_scope(name) as scope:
            conv_filter = tf.get_variable(name='filter', ....)
```
在tensorflow当中, conv2d是一个operation, 必须注意的是**参数strides与data_format的搭配**
```python
x = tf.nn.conv2d(inp, conv_filter, strides=[1, stride, stride, 1], data_format='NHWC')
x = tf.nn.conv2d(inp, conv_filter, strides=[1, 1, stride, stride], data_format='NCHW')
```
BN操作同样也要注意与 data_format 的配合, 因为在 CNN 当中BN是针对channel进行跨batch操作。所以, 要显式指定 channel 维度。
```python
x = tf.layers.batch_normalization(x, axis=3, training=is_training) # NHWC
x = tf.layers.batch_normalization(x, axis=1, training=is_training) # NCHW
```
BN另一个需要注意的点是: 它的运行方式是有分 train 和 inference 两个状态的。也就是说在train阶段, BN的统计数(mean, variance)会发生变化, 而inference阶段会保持统计数不变, 取得是train阶段的统计数的moving average。
```python
x = tf.layers.batch_normalization(x, axis=3, training=True) # train
x = tf.layers.batch_normalization(x, axis=3, training=False) # inference
```
---

目前_pool_layer只支持两种方式: Max, Avg. 后续还会有需要的情况下会继续增加操作。
```python
    def _pool_layer(self, name, inp, ksize, stride, padding='SAME', mode='MAX'):
        '''a pool layer which only supports avg_pooling and max_pooling(default)'''
        
        assert mode in ['MAX', 'AVG'], 'the mode of pool must be MAX or AVG'
        if mode == 'MAX':
            x = tf.nn.max_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        elif mode == 'AVG':
            x = tf.nn.avg_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        return x
```
在使用pooling的时候, 同样也要注意与 data_format 的配合。
```python
 x = tf.nn.max_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], data_format='NHWC') 
 x = tf.nn.max_pool(inp, ksize=[1, 1, ksize, ksize], strides=[1, 1, stride, stride], data_format='NCHW') 
```
---
由于fc_layer一般是接在 conv or pool layer后面, 所以会先做拉直操作(flatten)。
```python
    def _fc_layer(self, name, inp, units, dropout=0.5):
        '''a full connect layer'''
        
        with tf.variable_scope(name) as scope:
            shape = inp.get_shape().as_list() # get the shape of input
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(inp, [-1, dim]) # flatten
            if dropout > 0: # if with dropout
                x = tf.nn.dropout(x, keep_prob=dropout, name='dropout')
            x = tf.layers.dense(x, units, kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init, kernel_regularizer=self.reg)
        return x
```
---
build函数利用之前的基础模块: conv, pool, fc 进行组合, 定义graph
```python
  def bulid(self):
        # set inputs
        data = tf.placeholder(tf.float32, shape=(None,)+image_shape+(nr_channel,),
                              name='data')
        label = tf.placeholder(tf.int32, shape=(None,), name='label')
        label_onehot = tf.one_hot(label, nr_class, dtype=tf.int32) 
        is_training = tf.placeholder(tf.bool, name='is_training') # a flag of bn
        
        # conv1
        x = self._conv_layer(name='conv1', inp=data,
                             kernel_shape=[3, 3, nr_channel, 16], stride=1,
                             is_training=is_training) # Nx32x32x32
        x = self._pool_layer(name='pool1', inp=x, ksize=2, stride=2, mode='MAX') # Nx16x16x16

        # conv2
        x = self._conv_layer(name='conv2a', inp=x, kernel_shape=[3, 3, 16, 32],
                             stride=1, is_training=is_training)
        x = self._conv_layer(name='conv2b', inp=x, kernel_shape=[3, 3, 32, 32],
                             stride=1, is_training=is_training)
        x = self._pool_layer(name='pool2', inp=x, ksize=2, stride=2, mode='MAX') # Nx8x8x32
        
        # conv3
        x = self._conv_layer(name='conv3a', inp=x, kernel_shape=[3, 3, 32, 64],
                             stride=1, is_training=is_training)
        x = self._conv_layer(name='conv3b', inp=x, kernel_shape=[3, 3, 64, 64],
                             stride=1, is_training=is_training)
        x = self._pool_layer(name='pool3', inp=x, ksize=2, stride=2, mode='MAX') # Nx4x4x64

        # conv4
        x = self._conv_layer(name='conv4a', inp=x, kernel_shape=[3, 3, 64, 128],
                             stride=1, is_training=is_training)
        x = self._conv_layer(name='conv4b', inp=x, kernel_shape=[3, 3, 128, 128],
                             stride=1, is_training=is_training)
        x = self._pool_layer(name='pool4', inp=x, ksize=4, stride=4, mode='AVG') # Nx1x1x128
        
        # fc
        logits = self._fc_layer(name='fc1', inp=x, units=nr_class, dropout=0)
        
        # softmax
        preds = tf.nn.softmax(logits)
        
        placeholders = {
            'data': data,
            'label': label,
            'is_training': is_training,
        }
        return placeholders, label_onehot, logits, preds
```
由于graph在run的时候与外界是隔离的, 所以必须要通过某种接口与外界的数据进行交互。这种接口就是: **placeholder**. 
在shape参数指定**None**, 表示可以接受任意数值。
```
data = tf.placeholder(tf.float32, shape=(None,)+image_shape+(nr_channel,), name='data')
label = tf.placeholder(tf.int32, shape=(None,), name='label')
```
---
## Training
在定义完graph之后, 就可以开始raining。对应于01-svhn工程中的train.py

---
tensorflow在数据供给方面提供了几种方式。01-svhn工程中采用利用generator进行配合的方式。
```python
def get_dataset_batch(ds_name):
    '''get a batch generator of dataset'''
    dataset = Dataset(ds_name)
    ds_gnr = dataset.load().instance_generator
    ds = tf.data.Dataset().from_generator(ds_gnr, output_types=(tf.float32, tf.int32),)
    if ds_name == 'train':
        ds = ds.shuffle(dataset.instances_per_epoch)
        ds = ds.repeat(nr_epoch)
    elif ds_name == 'test':
        ds = ds.repeat(nr_epoch // test_interval)
    ds = ds.batch(minibatch_size, drop_remainder=True)
    ds_iter = ds.make_one_shot_iterator()
    sample_gnr = ds_iter.get_next()
    return sample_gnr, dataset
```
为了加速训练, tensorflow在shuffle时会要求指定数据量, 01-svhn默认指定一个epoch的数据量。所以, 有些同学会发现使用extra_32x32.mat之后会出现内存爆炸的问题。因此, 大家可以通过指定较小的数据量来avoid.    
```python
ds = ds.shuffle(dataset.instances_per_epoch)
```
用于指定数据集的遍历次数, 一般与训练的epoch数保持一致。
```python
ds = ds.repeat(nr_epoch)
```
获取一个batch生成器, 即每次都会生成数据的batch形式. drop_remainder用于指定: 若当最后一次batch数据量 < minibatch_size, 若true由丢弃不用, 默认是使用小batch的数据。
```python
ds = ds.batch(minibatch_size, drop_remainder=True)
ds_iter = ds.make_one_shot_iterator()
sample_gnr = ds_iter.get_next()
```
---
```
# load datasets
train_batch_gnr, train_set = get_dataset_batch(ds_name='train')
test_batch_gnr, test_set = get_dataset_batch(ds_name='test')

# build a compute graph
network = Model() 
placeholders, label_onehot, logits, preds = network.bulid()
loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
loss = tf.losses.softmax_cross_entropy(label_onehot, logits) + loss_reg

# set a performance metric
correct_pred = tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                        tf.cast(tf.argmax(label_onehot, 1), dtype=tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# learn rate config
global_steps = tf.Variable(0, trainable=False) # a cnt to record the num of minibatchs
boundaries = [train_set.minibatchs_per_epoch*15, train_set.minibatchs_per_epoch*40] 
values = [0.01, 0.001, 0.0005]
lr = tf.train.piecewise_constant(global_steps, boundaries, values)

opt = tf.train.AdamOptimizer(lr) # use adam as optimizer

# in order to update BN in every iter, a trick in tf
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = opt.minimize(loss)
```
tf.get_collection用于遍历图中的变量, 将具有共性的变量收集起来形成一个list。由于有正则化的变量会带有REGULARIZATION_LOSSES的key, 所以将这些变量值进行相加即得到正则化项的loss.
```python
loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
```
由于tensorflow没有直接实现的cross_entropy函数, 而是将softmax和cross_entropy直接集合成一个函数, 所以这里给定的参数必须是logits, 而不是preds.
```python
tf.losses.softmax_cross_entropy(label_onehot, logits)
```
将accuracy作为对模型性能的评测标准。
```python
correct_pred = tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                        tf.cast(tf.argmax(label_onehot, 1), dtype=tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```
learning rate是训练NN最重要的超参数, 在01-svhn当中采用阶梯退火的方式来更改学习率。boundaries指定变化的minibatchs点, values指定数值。 
```python
global_steps = tf.Variable(0, trainable=False) # a cnt to record the num of minibatchs
boundaries = [train_set.minibatchs_per_epoch*15, train_set.minibatchs_per_epoch*40] 
values = [0.01, 0.001, 0.0005]
lr = tf.train.piecewise_constant(global_steps, boundaries, values)
```
指定训练时使用的optimizer. 01-svhn工程使用adam, adam会有一些额外参数可以指定, 但必须指定的只有学习率。
```python
opt = tf.train.AdamOptimizer(lr) # use adam as optimizer
```
在tensorflow中, 要正确实现BN比其他框架要更为困难。由于BN的统计量是每个minibatch都要变化, 所以必须通过tf.control_dependencies来显式控制. 它会使得每一次run都会采用update_ops对应的最新值。
```python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = opt.minimize(loss)
```
---
tensorflow利用session来控制graph的running.
```python
# create a session
tf.set_random_seed(12345) # ensure consistent results
global_cnt = 0 # a cnt to record the num of minibatchs

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # init all variables
    # training
    for e in range(nr_epoch): 
        for _ in range(train_set.minibatchs_per_epoch):
            global_cnt += 1
            images, labels = sess.run(train_batch_gnr) # get a batch of (img, label)
            feed_dict = { # assign data to placeholders respectively
                    placeholders['data']: images,
                    placeholders['label']: labels,
                    global_steps: global_cnt,
                    placeholders['is_training']: True, # in training phase, set True
            }
            # run train, and output all values you want to monitor
            _, loss_v, acc_v, lr_v = sess.run([train, loss, accuracy, lr], 
                                              feed_dict=feed_dict)
            
            if global_cnt % show_interval == 0:
                print(
                    "e:{},{}/{}".format(e, global_cnt % train_set.minibatchs_per_epoch,
                                        train_set.minibatchs_per_epoch),
                    'loss: {:.3f}'.format(loss_v),
                    'acc: {:.3f}'.format(acc_v),
                    'lr: {:.3f}'.format(lr_v),
                )

    # validation
    if epoch % test_interval == 0:
        loss_sum, acc_sum = 0, 0 # init
        for i in range(test_set.minibatchs_per_epoch):
            images, labels = sess.run(test_batch_gnr)
            feed_dict = {
                    placeholders['data']: images,
                    placeholders['label']: labels,
                    global_steps: global_cnt,
                    placeholders['is_training']: False, # in test phase, set False
                }
            loss_v, acc_v = sess.run([loss, accuracy], feed_dict=feed_dict)
            loss_sum += loss_v # update
            acc_sum += acc_v # update
        print("\n**************Validation results****************")
        print('loss_avg: {:.3f}'.format(loss_sum / test_set.minibatchs_per_epoch),
              'accuracy_avg: {:.3f}'.format(acc_sum / test_set.minibatchs_per_epoch))
        print("************************************************\n")    

print('Training is done, exit.')
```
在开始进行trainning之前, 需要对graph中所有的变量进行init
```python
sess.run(tf.global_variables_initializer()) # init all variables
```
运行batch_gnr可以生成batch形式的images, labels. 在这里可以看出tensorflow只有利用run才会真正运行, 之前都只是定义符号。
```python
 images, labels = sess.run(train_batch_gnr) # get a batch of (img, label)
```
通过在list当中指定想run的目标。在train阶段, 必须要运行优化函数, 才能实现对loss的优化。所以, 问: 如何想实时获取learning rate的数值, 该怎么办？
```python
train = opt.minimize(loss)
  _, loss_v, acc_v = sess.run([train, loss, accuracy], feed_dict=feed_dict)
```

