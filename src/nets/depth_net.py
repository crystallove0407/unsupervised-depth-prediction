from __future__ import division
import functools
import numpy as np

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from models.common import conv1x1, depthwise_conv3x3, conv1x1_block, conv3x3_block, ChannelShuffle, SEBlock,\
    GluonBatchNormalization, MaxPool2d, get_channel_axis, flatten, dwconv3x3_block, conv1x1

from models.mobilenetv3 import get_mobilenetv3
from models.mobilenetv2 import get_mobilenetv2
from models.shufflenetv2 import get_shufflenetv2
from models.mnasnet import get_mnasnet
import time


def get_encoder(net_name, input_shape, training=False):
    if net_name == 'mobilenetv3':
        net = get_mobilenetv3(input_shape=input_shape,
                                version='small',
                                model_size='S',
                                training=training)
    if net_name == 'mobilenetv2':
        net = get_mobilenetv2(input_shape=input_shape,
                                model_size='S',
                                training=training)
    if net_name == 'shufflenetv2':
        net = get_shufflenetv2(input_shape=input_shape, model_size='S', training=training)
    
    if net_name == 'mnasnet':
        net = get_mnasnet(input_shape=input_shape,
                        version='small',
                        model_size='S',
                        training=training)
    return net

def SubpixelConv2D(input_shape, scale=4, name='subpixel'):
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)


    return nn.Lambda(subpixel, output_shape=subpixel_shape, name=name)




def Depth_net(net_name, input_shape=(128,416,3), training=False):
    inputs = tf.keras.Input(shape=input_shape)
    net = get_encoder(net_name=net_name, input_shape=input_shape, training=training)
    feature = net(inputs)
    x, skip = feature
    
    for i, sk in enumerate(skip):
        print('skip[{}]:{}'.format(i, sk.shape))
    
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_1')(x)
    x = dwconv3x3_block(in_channels=x.shape[-1] // 4, out_channels=x.shape[-1] // 4 , name='dw_block1')(x, training=training)
    x = conv1x1_block(in_channels=x.shape[-1] // 4, out_channels=64, activation=None, name='pw_block1')(x, training=training)
    x = dwconv3x3_block(in_channels=64, out_channels=64 , name='dw_block2')(x, training=training)
    x = conv1x1_block(in_channels=64, out_channels=64, activation=None, name='pw_block2')(x, training=training)
    x = dwconv3x3_block(in_channels=64, out_channels=64 , name='dwblock3')(x, training=training)
    x = conv1x1_block(in_channels=64, out_channels=64, activation=None, name='pw_block3')(x, training=training)
    x = dwconv3x3_block(in_channels=64, out_channels=64 , name='dw_block4')(x, training=training)
    x = conv1x1_block(in_channels=64, out_channels=64, activation=None, name='pw_block4')(x, training=training)
    
    x = nn.concatenate([x, skip[3]])
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_2')(x)
    x = dwconv3x3_block(in_channels=64 // 4, out_channels=16 ,name='dwblock5')(x, training=training)
    x = conv1x1_block(in_channels=16, out_channels=32, activation=None, name='pwblock5')(x, training=training)
    x = dwconv3x3_block(in_channels=32, out_channels=32 ,name='dwblock6')(x, training=training)
    x = conv1x1_block(in_channels=32, out_channels=32, activation=None, name='pwblock6')(x, training=training)
    x = dwconv3x3_block(in_channels=32, out_channels=32 ,name='dwblock7')(x, training=training)
    x = conv1x1_block(in_channels=32, out_channels=32, activation=None, name='pwblock7')(x, training=training)
    output_1 = x
    
    x = nn.concatenate([x, skip[2]])
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_3')(x)
    x = dwconv3x3_block(in_channels=32 // 4, out_channels=8 ,name='dwblock8')(x, training=training)
    x = conv1x1_block(in_channels=8, out_channels=24, activation=None, name='pwblock8')(x, training=training)
    x = dwconv3x3_block(in_channels=24, out_channels=24 ,name='dwblock9')(x, training=training)
    x = conv1x1_block(in_channels=24, out_channels=24, activation=None, name='pwblock9')(x, training=training)
    x = dwconv3x3_block(in_channels=24, out_channels=24 ,name='dwblock10')(x, training=training)
    x = conv1x1_block(in_channels=24, out_channels=24, activation=None, name='pwblock10')(x, training=training)
    output_2 = x
    
    x = nn.concatenate([x, skip[1]])
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_4')(x)
    x = dwconv3x3_block(in_channels=24 // 4, out_channels=6 ,name='dwblock11')(x, training=training)
    x = conv1x1_block(in_channels=6, out_channels=16, activation=None, name='pwblock11')(x, training=training)
    x = dwconv3x3_block(in_channels=16, out_channels=16 ,name='dwblock12')(x, training=training)
    x = conv1x1_block(in_channels=16, out_channels=16, activation=None, name='pwblock12')(x, training=training)
    output_3 = x
    
    x = nn.concatenate([x, skip[0]])
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_5')(x)
    x = dwconv3x3_block(in_channels=24 // 4, out_channels=6 ,name='dwblock13')(x, training=training)
    x = conv1x1_block(in_channels=6, out_channels=16, activation=None, name='pwblock13')(x, training=training)
    x = dwconv3x3_block(in_channels=16, out_channels=16 ,name='dwblock14')(x, training=training)
    x = conv1x1_block(in_channels=16, out_channels=16, activation=None, name='pwblock14')(x, training=training)
    output_4 = x
    
#     output_1 = conv1x1(in_channels=64, out_channels=1)(output_1)
#     output_2 = conv1x1(in_channels=32, out_channels=1)(output_2)
#     output_3 = conv1x1(in_channels=24, out_channels=1)(output_3)
#     output_4 = conv1x1(in_channels=16, out_channels=1)(output_4)
    output_1 = get_pred()(output_1)
    output_2 = get_pred()(output_2)
    output_3 = get_pred()(output_3)
    output_4 = get_pred()(output_4)
    
    
    model = tf.keras.Model(inputs, [output_1, output_2, output_3, output_4])
    return model
    
    

class get_pred(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.cnv = tf.keras.layers.Conv2D(1, 3, 1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = tf.pad(tensor=inputs, paddings=[
                   [0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.cnv(x)
        x = self.bn(x, training=training)
        x = tf.nn.elu(x)
        x = 5 * x + 0.01  # 預測深度值範圍: 0.01 ~ 5
        return x





if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
#     if(tf.executing_eagerly()):
#         print('[Info] Eager execution')
#         print('Eager execution is enabled (running operations immediately)\n')
#         print(('Turn eager execution off by running: \n{0}\n{1}').format('' \
#             'from tensorflow.python.framework.ops import disable_eager_execution', \
#             'tf.compat.v1.disable_eager_execution()'))
#     else:
#         print('[Info] Graph execution')
#         print('You are not running eager execution. TensorFlow version >= 2.0.0' \
#               'has eager execution enabled by default.')
#         print(('Turn on eager execution by running: \n\n{0}\n\nOr upgrade '\
#                'your tensorflow version by running:\n\n{1}').format(
#                'tf.compat.v1.enable_eager_execution()',
#                '!pip install --upgrade tensorflow\n' \
#                '!pip install --upgrade tensorflow-gpu'))
    
    
    @tf.function
    def test(inputs, model):
        x = model(inputs)
    
    model_name = ['shufflenetv2', 'mobilenetv2', 'mnasnet', 'mobilenetv3']
    
    #calculate fps
    for name in model_name:
        model = Depth_net(net_name=name, input_shape=(256, 832, 3), training=False)
        model.summary()
        
        
        averageFPS = 0
        for times in range(10):
            start_time = time.time()
            for num in range(10):
                x = tf.random.normal((1, 256, 832, 3))
                test(x, model)
            total_time = time.time() - start_time
            FPS = 10 / total_time
            averageFPS += FPS
        averageFPS /= 10
        print("[Info] {:>15} FPS: {:.3f}".format(name, averageFPS))