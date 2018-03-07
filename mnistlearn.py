import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("mnistdata",one_hot=True)
#定义一些常量
learn_rate = tf.Variable(0.005,dtype=tf.float32) 
epoch_step = 51
display_step = 1
loss_mode = ["square_mean","cross_entory"]
train_mode = ["Grad","Adam"]

#两层网络神经元数
layer1 = 40
layer2 = 20
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
#定义两个placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')

#定义权重和偏置
w = {"h1":tf.Variable(tf.truncated_normal([784,layer1])),
     "h2":tf.Variable(tf.truncated_normal([layer1,layer2])),
     "out":tf.Variable(tf.truncated_normal([layer2,10])),
    }
b = {"h1":tf.Variable(tf.zeros([layer1])+0.1),
     "h2":tf.Variable(tf.zeros([layer2])+0.1),
     "out":tf.Variable(tf.random_normal([10])),
    }

#定义神经网络
def network(x_input,weight,biases):
    net1 = tf.nn.tanh(tf.matmul(x_input,weight["h1"])+biases["h1"])
    net2 = tf.nn.tanh(tf.matmul(net1,weight["h2"])+biases["h2"])
    output = tf.nn.softmax(tf.matmul(net2,weight["out"])+biases["out"]) 
    return output
#定义loss函数
def lossfunction(loss_style,pred,y_input):
    loss = {"square_mean":tf.reduce_mean(tf.square(y-pred)),
            "cross_entory":tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_input,logits= pred))
           }
    return loss[loss_style]
#定义训练函数train
def trainfunction(train_style,learnrate,cost):
    train_step = {"Grad":tf.train.GradientDescentOptimizer(learning_rate = learnrate).minimize(cost),
                  "Adam":tf.train.AdamOptimizer(learning_rate= learnrate).minimize(cost)
                  }
    return train_step[train_style]
#执行函数
prediction = network(x,w,b)

loss = lossfunction(loss_mode[1],prediction,y)

train_step =  trainfunction(train_mode[1],learn_rate,loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("lossmode:"+loss_mode[1]+"；layercombin:"+str(layer1)+","+str(layer2))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_step):
        sess.run(tf.assign(learn_rate,0.01*(0.97**epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        if epoch % display_step == 0:  
            acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))