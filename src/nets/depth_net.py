from __future__ import division
import functools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def D_Net(inputs, weight_reg=0.0004, is_training=False, reuse=False):
    D_Model = ShuffleNetV2(input_holder=inputs, var_scope='depth_net', model_scale=0.5,
                           shuffle_group=2, is_training=is_training, decoderType='separable')
    return D_Model.build_model()


class ShuffleNetV2():
    '''
    Args:
        input_holder, var_scope, model_scale=1.0, shuffle_group=2, is_training=False, decoderType='separable'

    call function: 
        dec_block, shufflenetv2_block, shuffle_unit, conv_bn_relu, get_pred, upsample

    decoderType : 
        separable, mobile, shuffle
    '''
    first_conv_channel = 24

    def __init__(self, input_holder, var_scope, model_scale=1.0, shuffle_group=2, is_training=False, decoderType='separable'):
        self.input = input_holder
        self.output = None
        self.shuffle_group = shuffle_group
        self.channel_sizes = self._select_channel_size(model_scale)
        self.var_scope = var_scope
        self.is_training = is_training
        self.output_channel_sizes = [512, 256, 128, 64, 32]

    def _select_channel_size(self, model_scale):
        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
        if model_scale == 0.5:
            return [(48, 4), (96, 8), (192, 4), (1024, 1)]
        elif model_scale == 1.0:
            return [(116, 4), (232, 8), (464, 4), (1024, 1)]
        elif model_scale == 1.5:
            return [(176, 4), (352, 8), (704, 4), (1024, 1)]
        elif model_scale == 2.0:
            return [(244, 4), (488, 8), (976, 4), (2048, 1)]
        else:
            raise ValueError('Unsupported model size.')

    def build_model(self):
        '''Returns:
            pred[::-1], skip[-1]'''
        with tf.compat.v1.variable_scope(self.var_scope) as sc:
            with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                skip = []
                with tf.compat.v1.variable_scope('encoder'):
                    with tf.compat.v1.variable_scope('init_block'):
                        out = conv_bn_relu(
                            self.input, self.first_conv_channel, 3, 2)
                        skip.append(out)
                        out = slim.max_pool2d(out, 3, 2, padding='SAME')
                        skip.append(out)

                    for idx, block in enumerate(self.channel_sizes[:-1]):
                        with tf.compat.v1.variable_scope('shuffle_block_{}'.format(idx)):
                            out_channel, repeat = block

                            # First block is downsampling
                            out = shufflenetv2_block(
                                out, out_channel, 3, 2, shuffle_group=self.shuffle_group)

                            # Rest blocks
                            for i in range(repeat-1):
                                out = shufflenetv2_block(
                                    out, out_channel, 3, shuffle_group=self.shuffle_group)

                            skip.append(out)

                    with tf.compat.v1.variable_scope('end_block'):
                        out = conv_bn_relu(out, self.channel_sizes[-1][0], 1)

                for idx, sk in enumerate(skip):
                    print("skip[%d]:" % idx, sk.shape)

                with tf.compat.v1.variable_scope('decoder'):
                    # DECODER
                    pred = []
                    x = out
                    for idx, channel in enumerate(self.output_channel_sizes):
                        x = upsample(x, 2)
                        x = dec_block(x, channel, scope='up_block_%d' %
                                      idx, decoderType='separable')

                        if idx != 4:
                            x = tf.concat([x, skip[3-idx]], 3)
                        if idx > 1:
                            x = tf.concat([x, upsample(pred[idx-2], 2)], 3)

                        x = dec_block(x, channel, scope='block_%d' %
                                      idx, decoderType='separable')

                        if idx != 0:
                            pred.append(get_pred(x))

        return pred[::-1], skip[-1]


def dec_block(input_tensor, output_channel, scope=None, decoderType='separable'):
    '''
    input_tensor
    call function: 
        slim.separable_conv2d, slim.conv2d, sufflenet_v2_block
    '''
    with tf.compat.v1.variable_scope(scope, default_name='dec_block'):
        input_tensor = tf.identity(input_tensor, name='input')
        net = input_tensor

        # define depthwise function
        depthwise_func = functools.partial(slim.separable_conv2d,
                                           num_outputs=None,
                                           kernel_size=3,
                                           depth_multiplier=1,
                                           stride=1,
                                           rate=1,
                                           normalizer_fn=None,
                                           padding='SAME',
                                           scope='dwise')

        # main block
        if decoderType == 'separable':
            net = depthwise_func(net)
            net = slim.conv2d(net, output_channel, 1, activation_fn=None, scope='pwise')

        elif decoderType == 'mobile':
            net = slim.conv2d(net, output_channel, 1, scope='pwise0')
            net = depthwise_func(net)
            net = slim.conv2d(net, output_channel, 1, activation_fn=None, scope='pwise1')

        elif decoderType == 'shuffle':
            net = shufflenetv2_block(
                net, output_channel, kernel_size=3, stride=1, dilation=1, shuffle_group=2)

        return tf.identity(net, name='output')


def shufflenetv2_block(x, out_channel, kernel_size, stride=1, dilation=1, shuffle_group=2):
    # call function: conv_bn_relu, depthwise_conv_bn, shuffle_unit
    with tf.compat.v1.variable_scope(None, 'shuffle_v2_block'):
        half_channel = out_channel // 2

        # stride == 1
        if stride == 1 and x.shape[-1] == out_channel:
            top, bottom = tf.split(x, num_or_size_splits=2, axis=3)

            top = conv_bn_relu(top, half_channel, kernel_size=1, scope='0')
            top = depthwise_conv_bn(top, kernel_size, stride, dilation)
            top = conv_bn_relu(top, half_channel, kernel_size=1, scope='1')

            out = tf.concat([top, bottom], axis=3)
            out = shuffle_unit(out, shuffle_group)

        # stride == 2
        else:
            b0 = conv_bn_relu(x, half_channel, kernel_size=1, scope='R0')
            b0 = depthwise_conv_bn(b0, kernel_size, stride, dilation)
            b0 = conv_bn_relu(b0, half_channel, kernel_size=1, scope='R1')

            b1 = depthwise_conv_bn(x, kernel_size, stride, dilation)
            b1 = conv_bn_relu(b1, half_channel, kernel_size=1, scope='L0')

            out = tf.concat([b0, b1], axis=3)
            out = shuffle_unit(out, shuffle_group)

        return out


def shuffle_unit(x, groups):
    with tf.compat.v1.variable_scope(None, 'shuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor(
            value=[tf.shape(input=x)[0], h, w, groups, c // groups]))
        x = tf.transpose(a=x, perm=tf.convert_to_tensor(value=[0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor(
            value=[tf.shape(input=x)[0], h, w, c]))
    return x


def conv_bn_relu(x, out_channel, kernel_size, stride=1, dilation=1, scope=None):
    if scope != None:
        scopename = 'conv_bn_relu_' + scope
    else:
        scopename = scope
    with tf.compat.v1.variable_scope(scopename, 'conv_bn_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
    return x


def depthwise_conv_bn(x, kernel_size, stride=1, dilation=1):
    with tf.compat.v1.variable_scope(None, 'depthwise_conv_bn'):
        n, h, w, c = x.get_shape().as_list()
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=stride,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x


def get_pred(x):
    '''預測深度值範圍: 0.01 ~ 5, 太高loss 會爆炸'''
    x = tf.pad(tensor=x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    x = slim.conv2d(x, 1, 3, 1, 'VALID', activation_fn=tf.nn.elu,
                    normalizer_fn=slim.batch_norm)
    disp = 5 * x + 0.01
    return disp


def upsample(x, ratio):
    n, h, w, c = x.get_shape().as_list()
    return tf.image.resize(x, [h * ratio, w * ratio], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
