from __future__ import division
import functools
import numpy as np
import tensorflow as tf


class Depth_net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = shufflenetv2_encoder(model_scale=1.0, shuffle_group=2)
        self.decoder = decoder(decType='separable')

    def call(self, inputs, training=False):
        x, skip = self.encoder(inputs, training=training)
        x, _ = self.decoder(x, skip, training=training)
        return x


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
