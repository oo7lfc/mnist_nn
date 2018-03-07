import os
import sys
import numpy as np 
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
epoch_step = 9
display_step = 1
#loss_mode = ["square_mean","cross_entory"]
train_mode = ["Grad","Adam","Moment","RMSProp"]

# 输入图片是28*28
n_inputs = 28 #输入一行，一行有28个数据
max_time = 28 #一共28行
lstm_size = 50 #隐层单元
output_size = 10 # 10个分类
batch_size = 50 #每批次50个样本
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
#定义两个placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')

#定义权重和偏置
def weights_variable(shape):
    w = tf.Variable(tf.truncated_normal(shape))
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

#定义神经网络
def network(x_input,input_dim,output_dim,layer_name,act_style):
    act = {
           "tanh":tf.nn.tanh,
           "relu":tf.nn.relu,
           "sigmoid":tf.nn.sigmoid,
           "softmax":tf.nn.softmax,
           "elu":tf.nn.elu,
           "relu6":tf.nn.relu6
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
#定义RNN网络
def RNN_layer(x_input,input_dim,output_dim,layer_name):
    # inputs=[batch_size, max_time, n_inputs]
    inputs = tf.reshape(x_input,[-1,max_time,n_inputs])
    with tf.name_scope(layer_name):
    # 调用之前的方法初始化权重w和偏置b，并且调用参数信息的记录方法
        with tf.name_scope('weights'):          
            weights = weights_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = biases_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('lstm_cell'):
            #定义LSTM基本CELL
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(input_dim,reuse=tf.AUTO_REUSE) 

            # final_state[0]是cell state
            # final_state[1]是hidden_state
            _,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
        with tf.name_scope('output'):    
            output = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)
            tf.summary.histogram('output', output)
    return output
#定义loss函数
def lossfunction(pred,y_input):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_input,logits= pred))
        tf.summary.scalar('loss',loss)
    return loss
#定义训练函数train
def trainfunction(train_style,learnrate,cost): 

    train_mode = {"Grad":tf.train.GradientDescentOptimizer,
                  "Adam":tf.train.AdamOptimizer,
                  "Moment":tf.train.MomentumOptimizer,
                  "RMSProp":tf.train.RMSPropOptimizer
                 }
    with tf.name_scope('train'):
        train_step = train_mode[train_style](learning_rate= learnrate).minimize(cost)

    return train_step
#删除之前生成的log
if os.path.exists(log_dir + '/train'):
    delete_file_folder(log_dir + '/train')       
#执行函数

out = RNN_layer(x,lstm_size,output_size,'rnn')
loss = lossfunction(out,y)

train_step =  trainfunction(train_mode[1],learn_rate,loss)

#初始化变量
init = tf.global_variables_initializer()

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

#saver = tf.train.Saver()

#print("；layercombin:"+str(layer1_dim)+","+str(layer2_dim))
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(log_dir + '/train_rnn', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test_rnn', sess.graph)
    for epoch in range(epoch_step):
        sess.run(tf.assign(learn_rate,0.001*(0.95**epoch)))
        #sess.run(tf.assign(learn_rate,0.01/np.sqrt(epoch+1)))
     #   r = -2*np.random.rand()-2
     #   lr = 10**r
     #   sess.run(tf.assign(learn_rate,lr))
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged, train_step],feed_dict={x:batch_xs,y:batch_ys})
        train_writer.add_summary(summary,epoch)
        if epoch % display_step == 0:  
            summary_test,acc = sess.run([merged, accuracy],feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
            test_writer.add_summary(summary_test,epoch)
    #saver.save(sess,'net/my_net.ckpt')
    train_writer.close()
    test_writer.close()