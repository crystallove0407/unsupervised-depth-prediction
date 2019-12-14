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
#     inputs = tf.keras.Input(shape=(256, 832, 3), name='input_image')
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

def get_shufflenetv2(input_shape, model_size='S', **kwargs):
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
    
    if model_size == 'XS':
        width_scale = (12.0 / 29.0)
    if model_size == 'S':
        width_scale = 1.0
    if model_size == 'M':
        width_scale = (44.0 / 29.0)
    if model_size == 'L':
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
    output_1 = x
    
    x = nn.concatenate([x, skip[3]])
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_2')(x)
    x = dwconv3x3_block(in_channels=64 // 4, out_channels=16 ,name='dwblock5')(x, training=training)
    x = conv1x1_block(in_channels=16, out_channels=32, activation=None, name='pwblock5')(x, training=training)
    x = dwconv3x3_block(in_channels=32, out_channels=32 ,name='dwblock6')(x, training=training)
    x = conv1x1_block(in_channels=32, out_channels=32, activation=None, name='pwblock6')(x, training=training)
    x = dwconv3x3_block(in_channels=32, out_channels=32 ,name='dwblock7')(x, training=training)
    x = conv1x1_block(in_channels=32, out_channels=32, activation=None, name='pwblock7')(x, training=training)
    output_2 = x
    
    x = nn.concatenate([x, skip[2]])
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_3')(x)
    x = dwconv3x3_block(in_channels=32 // 4, out_channels=8 ,name='dwblock8')(x, training=training)
    x = conv1x1_block(in_channels=8, out_channels=24, activation=None, name='pwblock8')(x, training=training)
    x = dwconv3x3_block(in_channels=24, out_channels=24 ,name='dwblock9')(x, training=training)
    x = conv1x1_block(in_channels=24, out_channels=24, activation=None, name='pwblock9')(x, training=training)
    x = dwconv3x3_block(in_channels=24, out_channels=24 ,name='dwblock10')(x, training=training)
    x = conv1x1_block(in_channels=24, out_channels=24, activation=None, name='pwblock10')(x, training=training)
    output_3 = x
    
    x = nn.concatenate([x, skip[1]])
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_4')(x)
    x = dwconv3x3_block(in_channels=24 // 4, out_channels=6 ,name='dwblock11')(x, training=training)
    x = conv1x1_block(in_channels=6, out_channels=16, activation=None, name='pwblock11')(x, training=training)
    x = dwconv3x3_block(in_channels=16, out_channels=16 ,name='dwblock12')(x, training=training)
    x = conv1x1_block(in_channels=16, out_channels=16, activation=None, name='pwblock12')(x, training=training)
    output_4 = x
    
    output_1 = conv1x1(in_channels=64, out_channels=1)(output_1)
    output_2 = conv1x1(in_channels=32, out_channels=1)(output_2)
    output_3 = conv1x1(in_channels=24, out_channels=1)(output_3)
    output_4 = conv1x1(in_channels=16, out_channels=1)(output_4)
    
    model = tf.keras.Model(inputs, [output_1, output_2, output_3, output_4])
    return model
    
    




class shufflenetv2_encoder(tf.keras.layers.Layer):
    def __init__(self, model_scale=1.0, shuffle_group=2):
        super().__init__()
        # arg
        self.inputs = inputs
        self.channel_sizes = self._select_channel_size(model_scale)
        self.shuffle_group = shuffle_group

        # others
        self.outputs = None
        self.output_channel_sizes = [512, 256, 128, 64, 32]
        self.cnv1 = conv_bn_relu(24, 3)
        self.maxpool = tf.keras.layers.MaxPooling2D(3, 2, padding='same')
        self.cnv2 = conv_bn_relu(self.channel_sizes[-1][0], 1)

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

    def _make_shufflenetv2_block(self, inputs, channel_size, training=False):
        out_channel, repeat = channel_size
        x = shufflenetv2_unit(out_channel, 3, strides=2, shuffle_group=self.shuffle_group)(
            inputs, training=training)
        for i in range(repeat-1):
            x = shufflenetv2_unit(out_channel, 3, strides=1, shuffle_group=self.shuffle_group)(
                x, training=training)

        return x
    

    def call(self, inputs, training=False):
        skip = []

        x = cnv1(inputs, training=training)
        skip.append(x)

        x = maxpool(x)
        skip.append(x)

        for cs in self.channel_sizes[:-1]:
            x = self._make_shufflenetv2_block(x, cs, training=training)
            skip.append(x)

        x = cnv2(x, training=training)

        for idx, sk in enumerate(skip):
            print("skip[%d]:" % idx, sk.shape)

        return x, skip


class decoder(tf.keras.layers.Layer):
    def __init__(self, decType='separable'):
        super().__init__()
        self.decType = decType

    def _upsample(self, inputs, ratio):
        n, h, w, c = inputs.get_shape().as_list()
        return tf.image.resize(inputs, [h * ratio, w * ratio], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def call(self, inputs, skip, training=False):
        pred = []
        x = inputs
        for idx, channel in enumerate(self.output_channel_sizes):
            x = _upsample(x, 2)
            x = dec_block(channel, decType=self.decType)(x, training=training)

            if idx != 4:
                x = tf.concat([x, skip[3-idx]], 3)
            if idx > 1:
                x = tf.concat([x, _upsample(pred[idx-2], 2)], 3)

            x = dec_block(channel, decType=self.decType)(x, training=training)

            if idx != 0:
                pred.append(get_pred()(x, training=training))

        return pred[::-1], skip[-1]


class dec_block(tf.keras.layers.Layer):
    def __init__(self, output_channel, decType='separable', expansion_ratio=6):
        super().__init__()
        self.output_channel = output_channel
        self.decType = decType
        self.expansion_ratio = expansion_ratio

        if self.decType == 'separable':
            self.conv = tf.keras.layers.SeparableConv2D(
                output_channel, 3, use_bias=False)
            self.bn1 = tf.keras.layers.BatchNormalization()
        if self.decType == 'mobilenetv2':
            self.dwise = tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, strides=1, use_bias=False)
            self.pwise = tf.keras.layers.conv2d(
                output_channel, 1, use_bias=False)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
        if self.decType == 'shufflenetv2':
            self.shuffle_unit = shufflenetv2_unit(self.out_channel, 3)

    def call(self, inputs, training=False):
        if self.decType == 'separable':
            x = self.conv(inputs)
        if self.decType == 'mobilenetv2':
            # pw
            bottleneck_dim = round(
                expansion_ratio * inputs.get_shape().as_list()[-1])
            x = tf.keras.layers.conv2d(
                bottleneck_dim, 1, use_bias=False)(inputs)  # pwise
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            # dw
            x = self.dwise(x)
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)
            # pw & linear
            x = self.pwise(x)
            x = self.bn3(x, training=training)

        if decType == 'shufflenetv2':
            x = self.shuffle_unit(inputs)

        return x


class shufflenetv2_unit(tf.keras.layers.Layer):
    def __init__(self, out_channel, kernel_size, strides=1, dilation=1, shuffle_group=2):
        super().__init__()
        # arg
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.shuffle_group = shuffle_group

        # others
        self.pwise1 = conv_bn_relu(self.out_channel // 2, 1)
        self.pwise2 = conv_bn_relu(self.out_channel // 2, 1)
        self.pwise3 = conv_bn_relu(self.out_channel // 2, 1)
        self.pwise4 = conv_bn_relu(self.out_channel // 2, 1)
        self.pwise5 = conv_bn_relu(self.out_channel // 2, 1)
        self.dwise1 = depthwise_bn(3)
        self.dwise2 = depthwise_bn(3)
        self.dwise3 = depthwise_bn(3)

    def call(self, inputs, training=False):
        if self.stride == 1 and inputs.shape[-1] == self.out_channel:
            left, right = tf.split(inputs, num_or_size_splits=2, axis=3)

            right = self.pwise1(right, training=training)
            right = self.dwise1(right, training=training)
            right = self.pwise2(right, training=training)

        else:  # stride == 2
            left, right = x, x

            right = self.pwise3(right, training=training)
            right = self.dwise2(right, training=training)
            right = self.pwise4(right, training=training)

            left = self.dwise3(left, training=training)
            left = self.pwise5(left, training=training)

        x = tf.concat([left, right], axis=3)
        x = shuffle_unit(x, self.shuffle_group)
        return x


class conv_bn_relu(tf.keras.layers.Layer):
    def __init__(self, out_channel, kernel_size, strides=1, dilation=1):
        super().__init__()
        # arg
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation

        # others
        self.conv = tf.keras.layers.conv2d(
            self.out_channel, self.kernel_size, self.strides, dilation_rate=self.dilation)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class depthwise_bn(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides=1, dilation=1):
        super().__init__()
        # arg
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation

        # others
        self.dwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size, strides=self.strides, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.dwise(inputs)
        x = self.bn(x, training=training)
        return x


class get_pred(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.cnv = tf.keras.layers.conv2d(1, 3, 1)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = tf.pad(tensor=inputs, paddings=[
                   [0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.cnv(x)
        x = self.bn(x, training=training)
        x = tf.nn.elu(x)
        x = 5 * x + 0.01  # 預測深度值範圍: 0.01 ~ 5
        return x


def shuffle_unit(x, groups=2):
    n, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, shape=tf.convert_to_tensor(
        value=[tf.shape(input=x)[0], h, w, groups, c // groups]))
    x = tf.transpose(a=x, perm=tf.convert_to_tensor(value=[0, 1, 2, 4, 3]))
    x = tf.reshape(x, shape=tf.convert_to_tensor(
        value=[tf.shape(input=x)[0], h, w, c]))
    return x


if __name__ == '__main__':
    net = Depth_net(input_shape=(128, 416, 3), training=True)
    
#     for var in net.trainable_variables:
#         print(var.name)
    
#     for layer in net.layers:
#         print(layer.name)
    net.summary()