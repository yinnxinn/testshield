# -*-coding:utf8-*-#


import os
import sys
import cPickle
import cv2
import numpy
from PIL import Image
import random
import theano
import theano.tensor as T
import gamma
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
    f.close()
    return layer0_params,layer1_params,layer2_params,layer3_params,layer4_params


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


def use_CNN(img,params_file='params_5layer.txt',nkerns=[10,40]):   
    

    face_num = 1   #有多少张人脸图
  
    #读入参数
    layer0_params,layer1_params,layer2_params,layer3_params,layer4_params=load_params(params_file)
    
    x = T.matrix('x')  #用变量x表示输入的人脸数据，作为layer0的输入

    ######################
    #用读进来的参数初始化各层参数W、b
    ######################
    layer0_input = x.reshape((face_num, 1, 57,47)) 
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
        image_shape=(face_num, 1, 57,47),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        input=layer0.output,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        image_shape=(face_num, nkerns[0], 26,21),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        input=layer2_input,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        n_in=nkerns[1] * 11 * 8,
        n_out=3500,      
        activation=T.tanh
    )
    
    layer3 = HiddenLayer(
        input=layer2.output,
        params_W=layer3_params[0],
        params_b=layer3_params[1],
        n_in=3500,
        n_out=1500,      
        activation=T.tanh
    )
    layer4 = LogisticRegression(input=layer3.output, params_W=layer4_params[0],params_b=layer4_params[1],n_in=1500, n_out=4)   
     
    #定义theano.function，让x作为输入，layer3.y_pred（即预测的类别）作为输出
    f = theano.function(
        [x],    #funtion 的输入必须是list，即使只有一个输入
        layer4.y_pred
    )
    
    f_prob = theano.function(
        [x],    #funtion 的输入必须是list，即使只有一个输入
        layer4.p_y_given_x
    )

    #input data to get result
    src=cv2.resize(img,(57,47))
    
    if (len(img.shape)==3):
	temp=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    else:
	temp=src
    tmp=gamma.gamma_img(temp)
    dst=tmp.reshape(1,2679)
    face = numpy.array(dst, dtype='float64')/256
    
    pred = str(f(face))
    probility = f_prob(face)
    print '-----probility-----'
    print '-- wang : %.3f' %probility[0][0]
    print '-- xu   : %.3f' %probility[0][1]
    print '-- liu  : %.3f' %probility[0][2]
    print '--zhang : %.3f' %probility[0][3]
    if (max(probility[0])<0.4):
	
    	pred='unknown'
   
    return pred
   
