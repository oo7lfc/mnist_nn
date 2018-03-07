import os
import sys
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#删除src路径下所有文件
log_dir = 'log'
def delete_file_folder(src):
    '''delete files and folders'''
    if os.path.isfile(src):
        try:
            os.remove(src)
        except:
            pass
    elif os.path.isdir(src):
        for item in os.listdir(src):
            itemsrc=os.path.join(src,item)
            delete_file_folder(itemsrc) 
        try:
            os.rmdir(src)
        except:
            pass

#载入数据集
mnist = input_data.read_data_sets("mnistdata",one_hot=True)
#定义一些常量
learn_rate = tf.Variable(0.0001,dtype=tf.float32) 
epoch_step = 3
display_step = 50
#loss_mode = ["square_mean","cross_entory"]
train_mode = ["Grad","Adam"]
#定义卷积核大小
conv_k = [5,5]
#两层网络神经元数
layer1_dim = 40
layer2_dim = 20
#每个批次的大小
batch_size = 50
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
#定义两个placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')
    with tf.name_scope('x_image'):
        #改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
        x_image = tf.reshape(x,[-1,28,28,1],name='x_image')

#定义权重和偏置
def weights_variable(shape):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return w
def biases_variable(shape):
    b = tf.Variable(tf.zeros(shape)+0.1)
    return b
# 计算参数的均值，并使用tf.summary.scaler记录
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)#标准差
            tf.summary.scalar('max', tf.reduce_max(var))#最大值
            tf.summary.scalar('min', tf.reduce_min(var))#最小值
            tf.summary.histogram('histogram', var)#直方图
#卷积元
def Conv2d(x,W):
    #x input tensor of shape `[batch, in_height, in_width, in_channels]`
    #W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #`strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化元
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#定义神经网络层
def nn_layer(x_input,input_dim,output_dim,layer_name,act_style):
    act = {
           "tanh":tf.nn.tanh,
           "relu":tf.nn.relu,
           "sigmoid":tf.nn.sigmoid,
           "softmax":tf.nn.softmax
         }
    with tf.name_scope(layer_name):
        # 调用之前的方法初始化权重w和偏置b，并且调用参数信息的记录方法
        with tf.name_scope('weights'):          
            weights = weights_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = biases_variable([output_dim])
            variable_summaries(biases)
        # 执行wx+b的线性计算，并且用直方图记录下来
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(x_input, weights) + biases
            tf.summary.histogram('linear', preactivate)
        # 返回激励层的最终输出
        with tf.name_scope('activations'):  
            activations = act[act_style](preactivate, name='activation')
            tf.summary.histogram('activations', activations)

    return activations
#定义卷积神经网络层
def cnn_layer(x_input,filter_height,filter_width,input_dim,output_dim,layer_name,act_style):
    act = {
           "tanh":tf.nn.tanh,
           "relu":tf.nn.relu,
           "sigmoid":tf.nn.sigmoid,
           "softmax":tf.nn.softmax
         }
    with tf.name_scope(layer_name):
        # 调用之前的方法初始化权重w和偏置b，并且调用参数信息的记录方法
        with tf.name_scope('weights'):          
            weights = weights_variable([filter_height,filter_width,input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = biases_variable([output_dim])
            variable_summaries(biases)
        # 执行conv2d卷积
        with tf.name_scope('conv2d'):
            conv2d = Conv2d(x_input, weights) + biases
          #  tf.summary.histogram('conv2d', conv2d)
        #作用激活函数
        with tf.name_scope('activations'):  
            activations = act[act_style](conv2d, name='activation')
            tf.summary.histogram('activations', activations)
        # 执行池化元max_pool_2x2
        with tf.name_scope('h_pool'):  
            h_pool = max_pool_2x2(activations)
          #  tf.summary.histogram('h_pool', h_pool)

    return h_pool
#定义loss函数
def lossfunction(pred,y_input):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_input,logits= pred))
        tf.summary.scalar('loss',loss)
    return loss
#定义训练函数train
def trainfunction(train_style,learnrate,cost): 

    train_mode = {"Grad":tf.train.GradientDescentOptimizer,
                  "Adam":tf.train.AdamOptimizer
                 }
    with tf.name_scope('train'):
        train_step = train_mode[train_style](learning_rate= learnrate).minimize(cost)

    return train_step
#执行网络 
layer1 = cnn_layer(x_image,conv_k[0],conv_k[1],1,32,'conv2_l1','relu')
layer2 = cnn_layer(layer1,conv_k[0],conv_k[1],32,64,'conv2_l2','relu')
#把池化层2的输出扁平化为1维
with tf.name_scope('h_pool2_flat'):
    h_pool2_flat = tf.reshape(layer2,[-1,7*7*64],name='h_pool2_flat')
layer3 = nn_layer(h_pool2_flat,7*7*64,1024,'layer3','relu')
out = nn_layer(layer3,1024,10,'output','softmax')
loss = lossfunction(out,y)

train_step =  trainfunction(train_mode[1],learn_rate,loss)

#初始化变量
init = tf.global_variables_initializer()
#saver = tf.train.Saver()
with tf.name_scope('accuracy'): 
    with tf.name_scope('correct_prediction'): 
        #结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(out,1))#argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
#合并所有的summary
merged = tf.summary.merge_all()

#删除之前生成的log
if os.path.exists(log_dir + '/train_cnn'):
    delete_file_folder(log_dir + '/train_cnn')   

print("；layercombin:"+str(layer1_dim)+","+str(layer2_dim))
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(log_dir + '/train_cnn', sess.graph) 
    test_writer = tf.summary.FileWriter(log_dir + '/test_cnn', sess.graph)    
    for epoch in range(epoch_step):
       # sess.run(tf.assign(learn_rate,0.01*(0.97**epoch)))
        for batch in range(n_batch):
            time_start = time.time()
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged, train_step],feed_dict={x:batch_xs,y:batch_ys})
        #train_writer.add_summary(summary,epoch)
            if batch % display_step == 0:  
                summary_test,acc = sess.run([merged,accuracy],feed_dict={x:mnist.test.images,y:mnist.test.labels})
                time_end = time.time()
                testtime = time_end - time_start
                print("Iter " + str(batch+epoch*1100) + ",Testing Accuracy " + str(acc)+",Time Cost "+str(testtime)+",Timenow"+str(time_end))
                test_writer.add_summary(summary_test,batch+epoch*1100)
    train_writer.close()
    test_writer.close()