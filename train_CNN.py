# -*-coding:utf8-*-#

import os
import sys
import time
import cv2
import random
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

name={"0":'wang','1':'xu','2':'liu','3':'zhang'}

def load_data():
    global name
    img=[]
    labels=[]
    scan=True
    label=0
    for i in sorted(name.keys()):
        img_path='../data/'+name[i]
	
        if (not (os.path.exists(img_path))):
            print 'no dictory found'
	    sys.exit(0)
        else:
            for j in xrange(120):
		
                img_name=img_path+'/'+str(j+1)+'.jpg'
		
		src = cv2.imread(img_name,1)	
                dst=cv2.resize(src,(84,84))
	
                temp=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
                img.append(temp)
                labels.append(eval(i))
       
            
    img_ndarray = numpy.array(img, dtype='float64')/256
    labels=numpy.array(labels,dtype='int32')
    #faces=numpy.empty((165,10000))
    faces=img_ndarray.reshape(480,84*84)
    
    tdata=[]
    ffaces=[]
    flabels=[]
    for i in xrange(len(faces)):
	tdata.append((faces[i],labels[i]))
    random.shuffle(tdata)

    for i in xrange(len(tdata)):
	ffaces.append(tdata[i][0])
	flabels.append(tdata[i][1])
    
    #分成训练集、验证集、测试集，大小如下
    train_data=numpy.empty((440,84*84))
    train_label=numpy.empty(440)
    valid_data=numpy.empty((20,84*84))
    valid_label=numpy.empty(20)
    test_data=numpy.empty((20,84*84))
    test_label=numpy.empty(20)

    for i in xrange(4):
        train_data[i*110:i*110+110]=ffaces[i*120:i*120+110]
        valid_data[i*5:i*5+5]=ffaces[i*120+110:i*120+115]
        test_data[i*5:i*5+5]=ffaces[i*120+115:i*120+120]


        train_label[i*110:i*110+110]=flabels[i*120:i*120+110]
        valid_label[i*5:i*5+5]=flabels[i*120+110:i*120+115]
        test_label[i*5:i*5+5]=flabels[i*120+115:i*120+120]
    
    #将数据集定义成shared类型，才能将数据复制进GPU，利用GPU加速程序。
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')



    train_set_x, train_set_y = shared_dataset(train_data,train_label)
    valid_set_x, valid_set_y = shared_dataset(valid_data,valid_label)
    test_set_x, test_set_y = shared_dataset(test_data,test_label)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval



#分类器，即CNN最后一层，采用逻辑回归（softmax）
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


#全连接层，分类器前一层
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 2
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


#卷积+采样层（conv+maxpooling）
class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # 卷积
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # 子采样
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


#保存训练参数的函数
def save_params(param1,param2,param3,param4,param5,param6):
	
        import cPickle  
        write_file = open('params_5layer.txt', 'wb+')
        cPickle.dump(param1, write_file, -1)
        cPickle.dump(param2, write_file, -1)
        cPickle.dump(param3, write_file, -1)
        cPickle.dump(param4, write_file, -1)
        cPickle.dump(param5, write_file, -1)
        cPickle.dump(param6, write_file, -1)
        write_file.close()  



"""
上面定义好了CNN的一些基本构件，下面的函数将CNN应用于olivettifaces这个数据集，CNN的模型基于LeNet。
采用的优化算法是批量随机梯度下降算法，minibatch SGD，所以下面很多参数都带有batch_size，比如image_shape=(batch_size, 1, 57, 47)
可以设置的参数有：
batch_size,但应注意n_train_batches、n_valid_batches、n_test_batches的计算都依赖于batch_size
nkerns=[5, 10]即第一二层的卷积核个数可以设置
全连接层HiddenLayer的输出神经元个数n_out可以设置，要同时更改分类器的输入n_in
另外，还有一个很重要的就是学习速率learning_rate.
"""

def cal_cnn(learning_rate=0.006, n_epochs=1000,nkerns=[16,32,64], batch_size=10):   
    #参数文件判断
    if (os.path.exists('params_5layer.txt')):
	os.remove('params_5layer.txt') 
    #随机数生成器，用于初始化参数
    rng = numpy.random.RandomState(12345)
    #加载数据
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    #计算各数据集的batch个数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    #定义几个变量，x代表人脸数据，作为layer0的输入
    index = T.lscalar()
    x = T.matrix('x')  
    y = T.ivector('y')



    ######################
    #建立CNN模型:
    #input+layer0(LeNetConvPoolLayer)+layer1(LeNetConvPoolLayer)+layer2(HiddenLayer)+layer3(LogisticRegression)
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 57 * 47)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (57, 47) is the size of  images.
    layer0_input = x.reshape((batch_size, 1, 84, 84))

    # 第一个卷积+maxpool层

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 84, 84),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # 第二个卷积+maxpool层,输入是上层的输出，即(batch_size, nkerns[0], 44, 44)
 
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 40, 40),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

  # 第三个卷积+maxpool层,输入是上层的输出，即(batch_size, nkerns[1], 21, 21)
 
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 18, 18),
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        poolsize=(2, 2)
    )
   
    # HiddenLayer全连接层，它的输入的大小是(batch_size, num_pixels)，也就是说要将每个样本经layer0、layer1后得到的特征图整成一个一维的长向量，   
    #有batch_size个样本，故输入的大小为(batch_size, num_pixels)，每一行是一个样本的长向量
    #因此将上一层的输出(batch_size, nkerns[2], 9, 9)转化为(batch_size, nkerns[2] * 9 * 9),用flatten
    layer3_input = layer2.output.flatten(2)
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 7 * 7,
        n_out=2048,      #全连接层输出神经元的个数，自己定义的，可以根据需要调节
        activation=T.tanh
    )

    layer4 = HiddenLayer(
        rng,
        input=layer3.output,
        n_in=2048,
        n_out=1024,      #全连接层输出神经元的个数，自己定义的，可以根据需要调节
        activation=T.tanh
    )
    #输出层
    layer5 = LogisticRegression(input=layer4.output, n_in=1024, n_out=4)   #n_in等于全连接层的输出，n_out等于40个类别


    ###############
    # 定义优化算法的一些基本要素：代价函数，训练、验证、测试model、参数更新规则（即梯度下降）
    ###############
    # 代价函数
    cost = layer5.negative_log_likelihood(y)
    
    test_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # 所有参数
    params =layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    #各个参数的梯度
    grads = T.grad(cost, params)
    #参数更新规则
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    #train_model在训练过程中根据MSGD优化更新参数
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    ###############
    # 训练CNN阶段，寻找最优的参数。
    ###############
    print '... training'
    #在LeNet5中，batch_size=500,n_train_batches=50000/500=100，patience=10000
    #在olivettifaces中，batch_size=15,n_train_batches=120/15=8, paticence可以相应地设置为800，这个可以根据实际情况调节，调大一点也无所谓
    patience = 1000
    patience_increase = 2  
    improvement_threshold = 0.99  
    validation_frequency = min(n_train_batches, patience / 2) 


    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
	    
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
		    save_params(layer0.params,layer1.params,layer2.params,layer3.params,layer4.params,layer5.params)#保存参数

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))




if __name__ == '__main__':
	cal_cnn()
