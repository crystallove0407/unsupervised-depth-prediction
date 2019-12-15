from __future__ import division
import functools
import numpy as np
import tensorflow as tf

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from models.common import conv1x1, depthwise_conv3x3, conv1x1_block, conv3x3_block, ChannelShuffle, SEBlock,\
    GluonBatchNormalization, MaxPool2d, get_channel_axis, flatten, dwconv3x3_block, conv1x1


class ShuffleUnit(nn.Layer):
    """
    ShuffleNetV2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    downsample : bool
        Whether do downsample.
    use_se : bool
        Whether to use SE block.
    use_residual : bool
        Whether to use residual connection.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffleUnit, self).__init__(**kwargs)
        self.data_format = data_format
        self.downsample = downsample
        self.use_se = use_se
        self.use_residual = use_residual
        mid_channels = out_channels // 2

        self.compress_conv1 = conv1x1(
            in_channels=(in_channels if self.downsample else mid_channels),
            out_channels=mid_channels,
            data_format=data_format,
            name="compress_conv1")
        self.compress_bn1 = GluonBatchNormalization(
            # in_channels=mid_channels,
            data_format=data_format,
            name="compress_bn1")
        self.dw_conv2 = depthwise_conv3x3(
            channels=mid_channels,
            strides=(2 if self.downsample else 1),
            data_format=data_format,
            name="dw_conv2")
        self.dw_bn2 = GluonBatchNormalization(
            # in_channels=mid_channels,
            data_format=data_format,
            name="dw_bn2")
        self.expand_conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="expand_conv3")
        self.expand_bn3 = GluonBatchNormalization(
            # in_channels=mid_channels,
            data_format=data_format,
            name="expand_bn3")
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                data_format=data_format,
                name="se")
        if downsample:
            self.dw_conv4 = depthwise_conv3x3(
                channels=in_channels,
                strides=2,
                data_format=data_format,
                name="dw_conv4")
            self.dw_bn4 = GluonBatchNormalization(
                # in_channels=in_channels,
                data_format=data_format,
                name="dw_bn4")
            self.expand_conv5 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                data_format=data_format,
                name="expand_conv5")
            self.expand_bn5 = GluonBatchNormalization(
                # in_channels=mid_channels,
                data_format=data_format,
                name="expand_bn5")

        self.activ = nn.ReLU()
        self.c_shuffle = ChannelShuffle(
            channels=out_channels,
            groups=2,
            data_format=data_format,
            name="c_shuffle")

    def call(self, x, training=None):
        if self.downsample:
            y1 = self.dw_conv4(x)
            y1 = self.dw_bn4(y1, training=training)
            y1 = self.expand_conv5(y1)
            y1 = self.expand_bn5(y1, training=training)
            y1 = self.activ(y1)
            x2 = x
        else:
            y1, x2 = tf.split(x, num_or_size_splits=2,
                              axis=get_channel_axis(self.data_format))
        y2 = self.compress_conv1(x2)
        y2 = self.compress_bn1(y2, training=training)
        y2 = self.activ(y2)
        y2 = self.dw_conv2(y2)
        y2 = self.dw_bn2(y2, training=training)
        y2 = self.expand_conv3(y2)
        y2 = self.expand_bn3(y2, training=training)
        y2 = self.activ(y2)
        if self.use_se:
            y2 = self.se(y2)
        if self.use_residual and not self.downsample:
            y2 = y2 + x2
        x = tf.concat([y1, y2], axis=get_channel_axis(self.data_format))
        x = self.c_shuffle(x)
        return x


class ShuffleInitBlock(nn.Layer):
    """
    ShuffleNetV2 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffleInitBlock, self).__init__(**kwargs)
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            data_format=data_format,
            name="conv")
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            ceil_mode=True,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x1 = self.conv(x, training=training)
        x2 = self.pool(x1)
        return x1, x2


def shufflenetv2(inputs,
                 channels,
                init_block_channels,
                final_block_channels,
                training=None,
                use_se=False,
                use_residual=False,
                in_channels=3,
                in_size=(224, 224),
                data_format="channels_last",
                model_name='shufflenetv2_wd2',
                **kwargs):
    skip = []
    x1, x = ShuffleInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block")(inputs, training=training)
    skip.append(x1)
    skip.append(x)
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        stage = tf.keras.Sequential(name="stage{}".format(i + 1))
        for j, out_channels in enumerate(channels_per_stage):
            downsample = (j == 0)
            stage.add(ShuffleUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=downsample,
                use_se=use_se,
                use_residual=use_residual,
                data_format=data_format,
                name="unit{}".format(j + 1)))
            in_channels = out_channels
        x = stage(x, training=training)
        skip.append(x)
    x = conv1x1_block(
        in_channels=in_channels,
        out_channels=final_block_channels,
        data_format=data_format,
        name="final_block")(x, training=training)
    features = tf.keras.Model(inputs=inputs, outputs=[x, skip], name=model_name)
    return features

def get_shufflenetv2(input_shape, model_size='M', **kwargs):
    """
    Create ShuffleNetV2 model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    model_size : str, default 'S'
        Model size: XS, S, M, L -> 0.5, 1, 1.5, 2
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    
    inputs = tf.keras.Input(shape=input_shape, name='input_image')
    init_block_channels = 24
    final_block_channels = 1024
    layers = [4, 8, 4]
    channels_per_layers = [116, 232, 464]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    
    if model_size == 'S':
        width_scale = (12.0 / 29.0)
    if model_size == 'M':
        width_scale = 1.0
    if model_size == 'L':
        width_scale = (44.0 / 29.0)
    if model_size == 'XL':
        width_scale = (61.0 / 29.0)

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        if width_scale > 1.5:
            final_block_channels = int(final_block_channels * width_scale)

    net = shufflenetv2(
        inputs=inputs,
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        model_name='shufflenetv2_'+model_size,
        **kwargs)

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

def Depth_net(input_shape=(128,416,3), training=False):
    inputs = tf.keras.Input(shape=input_shape)
    shufflenetv2 = get_shufflenetv2(input_shape=input_shape, model_size='S')
    feature = shufflenetv2(inputs)
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
    net = Depth_net(input_shape=(256, 832, 3), training=False)
    
    for var in net.trainable_variables:
        print(var.name)
    for wei in net.weights:
        print(wei)
    
#     for layer in net.layers:
#         print(layer.name)

    net.summary()