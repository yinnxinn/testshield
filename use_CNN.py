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
from theano.tensor.signal import pool
from theano.tensor.nnet import conv


name={"0":'wang','1':'xu','2':'liu','3':'zhang'}
#读取之前保存的训练参数
#layer0_params~layer3_params都是包含W和b的,layer*_params[0]是W，layer*_params[1]是b
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


#读取图像，返回numpy.array类型的人脸数据以及对应的label
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
            for j in xrange(100):
		
                img_name=img_path+'/'+str(j+1)+'.jpg'
		
		src = cv2.imread(img_name,1)	
                dst=cv2.resize(src,(84,84))
	
                temp=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
                img.append(temp)
                labels.append(eval(i))
    img_ndarray = numpy.array(img, dtype='float64')/256
    faces=img_ndarray.reshape(400,84*84)
    return faces,labels



"""
train_CNN_olivettifaces中的LeNetConvPoolLayer、HiddenLayer、LogisticRegression是随机初始化的
下面将它们定义为可以用参数来初始化的版本
"""
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


"""
用之前保存下来的参数初始化CNN，就得到了一个训练好的CNN模型，然后使用这个模型来测图像
注意：n_kerns跟之前训练的模型要保持一致。dataset是你要测试的图像的路径，params_file是之前训练时保存的参数文件的路径
"""
def use_CNN(params_file='params_5layer.txt',nkerns=[16,32,64]):   
    
    #读取测试的图像，这里读取整个olivettifaces.gif，即全部样本，得到faces、label
    faces,label=load_data()
    face_num = faces.shape[0]   #有多少张人脸图
  
    #读入参数
    layer0_params,layer1_params,layer2_params,layer3_params,layer4_params,layer5_params=load_params(params_file)
    
    x = T.matrix('x')  #用变量x表示输入的人脸数据，作为layer0的输入

    ######################
    #用读进来的参数初始化各层参数W、b
    ######################
    layer0_input = x.reshape((face_num, 1, 84, 84)) 
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
        image_shape=(face_num, 1, 84, 84),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        input=layer0.output,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        image_shape=(face_num, nkerns[0], 40, 40),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2 = LeNetConvPoolLayer(
        input=layer1.output,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        image_shape=(face_num, nkerns[1], 18, 18),
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
    
    #预测的类别pred
    pred = f(faces)
    print pred
    print faces.shape

   #将预测的类别pred与真正类别label对比，输出错分的图像
    for i in range(face_num): 
	 if pred[i] != label[i]:
                print 'picture is person %s, but mis-predicted as person %s' %(name[str(label[i])], name[str(pred[i])])
		img=faces[i].reshape(84,84)
		cv2.imshow("show",img)
		c=cv2.waitKey(0)
		if c==ord('q'):
			sys.exit(0)
'''
while(True):
	num=random.randint(0,139)
	img=faces[num].reshape(47,57)
	cv2.imshow('img',img)
	c=cv2.waitKey(0)
	if c & 0xFF==ord('q'):
		sys.exit(0)'''

if __name__ == '__main__':
	
	use_CNN()




