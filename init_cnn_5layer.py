# -*-coding:utf8-*-#
"""
本程序实现的功能：
在train_CNN_olivettifaces.py中我们训练好并保存了模型的参数，利用这些保存下来的参数来初始化CNN模型，
这样就得到一个可以使用的CNN系统，将人脸图输入这个CNN系统，预测人脸图的类别。

@author:wepon(http://2hwp.com)
讲解这份代码的文章：http://blog.csdn.net/u012162613/article/details/43277187
"""

import os
import sys
import cPickle
import cv2
import numpy
from PIL import Image
import random
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

name={"0":'wang','1':'xu','2':'liu','3':'zhang'}

def load_params(params_file):
    f=open(params_file,'rb')
    layer0_params=cPickle.load(f)
    layer1_params=cPickle.load(f)
    layer2_params=cPickle.load(f)
    layer3_params=cPickle.load(f)
    layer4_params=cPickle.load(f)
    layer5_params=cPickle.load(f) 
    f.close()
    return layer0_params,layer1_params,layer2_params,layer3_params,layer4_params,layer5_params


class LogisticRegression(object):
    def __init__(self, input, params_W,params_b,n_in, n_out):
        self.W = params_W
        self.b = params_b
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


class HiddenLayer(object):
    def __init__(self, input, params_W,params_b, n_in, n_out,
                 activation=T.tanh):
        self.input = input
        self.W = params_W
        self.b = params_b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

	
#卷积+采样层（conv+maxpooling）
class LeNetConvPoolLayer(object):
    def __init__(self,  input,params_W,params_b, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = params_W
        self.b = params_b
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
        self.params = [self.W, self.b]


def use_CNN(img,params_file='params_5layer.txt',nkerns=[16,32,64]):   
    

    face_num = 1   #有多少张人脸图
  
    #读入参数
    layer0_params,layer1_params,layer2_params,layer3_params,layer4_params,layer5_params=load_params(params_file)
    
    x = T.matrix('x')  #用变量x表示输入的人脸数据，作为layer0的输入

    ######################
    #用读进来的参数初始化各层参数W、b
    ######################
    layer0_input = x.reshape((face_num, 1, 84,84)) 
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
        image_shape=(face_num, 1, 84,84),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        input=layer0.output,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        image_shape=(face_num, nkerns[0], 40,40),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )
    layer2 = LeNetConvPoolLayer(
        input=layer1.output,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        image_shape=(face_num, nkerns[1], 18,18),
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        poolsize=(2, 2)
    )

    layer3_input = layer2.output.flatten(2)
    layer3 = HiddenLayer(
        input=layer3_input,
        params_W=layer3_params[0],
        params_b=layer3_params[1],
        n_in=nkerns[2] * 7 * 7,
        n_out=2048,      
        activation=T.tanh
    )
    
    layer4 = HiddenLayer(
        input=layer3.output,
        params_W=layer4_params[0],
        params_b=layer4_params[1],
        n_in=2048,
        n_out=1024,      
        activation=T.tanh
    )
    layer5 = LogisticRegression(input=layer4.output, params_W=layer5_params[0],params_b=layer5_params[1],n_in=1024, n_out=4)   
     
    #定义theano.function，让x作为输入，layer3.y_pred（即预测的类别）作为输出
    f = theano.function(
        [x],    #funtion 的输入必须是list，即使只有一个输入
        layer5.y_pred
    )

    p = theano.function(
        [x],    #funtion 的输入必须是list，即使只有一个输入
        layer5.p_y_given_x
    )    
    
    src=cv2.resize(img,(84,84))
    
    if (len(img.shape)==3):
	temp=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    else:
	temp=src
    
    dst=temp.reshape(1,84*84)
    face = numpy.array(dst, dtype='float64')/256
    
    pred = f(face)
    problity=p(face)
    print ' probability of corresponding person :'
    print name['0'],'  : ',problity[0][0]
    print name['1'],'    : ',problity[0][1]
    print name['2'],'   : ',problity[0][2]
    print name['3'],' : ',problity[0][3]
    print '-'*30
    return pred
   
