# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from utils.optical_flow_warp_old import transformer_old


class Flow_net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.feature_pyramid = feature_pyramid_flow()
        self.get_flow = get_flow()

    def call(self, src0, tgt, src1):
        self.shape = get_allShape(tgt)

        feature_src0 = self.feature_pyramid(src0)
        feature_tgt = self.feature_pyramid(tgt)
        feature_src1 = self.feature_pyramid(src1)

        # foward warp: |01 |, | 12|, |0 2| ,direction: ->
        flow_fw0 = get_flow(feature_src0, feature_tgt, self.shape)
        flow_fw1 = get_flow(feature_tgt, feature_src1, self.shape)
        flow_fw2 = get_flow(feature_src0, feature_src1, self.shape)

        # backward warp: |01 |, | 12|, |0 2| , direction: <-
        flow_bw0 = get_flow(feature_tgt, feature_src0, self.shape)
        flow_bw1 = get_flow(feature_src1, feature_tgt, self.shape)
        flow_bw2 = get_flow(feature_src1, feature_src0, self.shape)

        return [flow_fw0, flow_fw1, flow_fw2], [flow_bw0, flow_bw1, flow_bw2]


class feature_pyramid_flow(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = tf.keras.layers.conv2d(
            16, 3, strides=2, activation=leaky_relu)
        self.conv2d_2 = tf.keras.layers.conv2d(
            16, 3, strides=1, activation=leaky_relu)
        self.conv2d_3 = tf.keras.layers.conv2d(
            32, 3, strides=2, activation=leaky_relu)
        self.conv2d_4 = tf.keras.layers.conv2d(
            32, 3, strides=1, activation=leaky_relu)
        self.conv2d_5 = tf.keras.layers.conv2d(
            64, 3, strides=2, activation=leaky_relu)
        self.conv2d_6 = tf.keras.layers.conv2d(
            64, 3, strides=1, activation=leaky_relu)
        self.conv2d_7 = tf.keras.layers.conv2d(
            96, 3, strides=2, activation=leaky_relu)
        self.conv2d_8 = tf.keras.layers.conv2d(
            96, 3, strides=1, activation=leaky_relu)
        self.conv2d_9 = tf.keras.layers.conv2d(
            128, 3, strides=2, activation=leaky_relu)
        self.conv2d_10 = tf.keras.layers.conv2d(
            128, 3, strides=1, activation=leaky_relu)
        self.conv2d_11 = tf.keras.layers.conv2d(
            192, 3, strides=2, activation=leaky_relu)
        self.conv2d_12 = tf.keras.layers.conv2d(
            192, 3, strides=1, activation=leaky_relu)

    def call(self, inputs):
        cnv1 = self.conv2d_1(inputs)
        cnv2 = self.conv2d_2(cnv1)
        cnv3 = self.conv2d_3(cnv2)
        cnv4 = self.conv2d_4(cnv3)
        cnv5 = self.conv2d_5(cnv4)
        cnv6 = self.conv2d_6(cnv5)
        cnv7 = self.conv2d_7(cnv6)
        cnv8 = self.conv2d_8(cnv7)
        cnv9 = self.conv2d_9(cnv8)
        cnv10 = self.conv2d_10(cnv9)
        cnv11 = self.conv2d_11(cnv10)
        cnv12 = self.conv2d_12(cnv11)
        return cnv2, cnv4, cnv6, cnv8, cnv10, cnv12


class get_flow(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.cost_volumn = cost_volumn(d=4)
        self.context_net = context_net()
        self.decoder2 = pwc_decoder()
        self.decoder3 = pwc_decoder()
        self.decoder4 = pwc_decoder()
        self.decoder5 = pwc_decoder()
        self.decoder6 = pwc_decoder()

    def call(self, feature1, feature2, shape):
        self.shape = shape
        f11, f12, f13, f14, f15, f16 = feature1
        f21, f22, f23, f24, f25, f26 = feature2

        # Block6
        cv6 = self.cost_volumn(f16, f26)
        flow6, _ = self.decoder6(cv6)

        # Block5
        flow65 = 2.0 * \
            tf.image.resize(
                flow6, self.shape[5], method=tf.image.ResizeMethod.BILINEAR)
        f25_warp = transformer_old(f25, flow65, self.shape[5])
        cv5 = self.cost_volumn(f15, f25_warp)
        flow5, _ = self.decoder5(tf.concat([cv5, f15, flow65], axis=3))
        flow5 = flow5 + flow65

        # Block4
        flow54 = 2.0 * \
            tf.image.resize(
                flow5, self.shape[4], method=tf.image.ResizeMethod.BILINEAR)
        f24_warp = transformer_old(f24, flow54, self.shape[4])
        cv4 = self.cost_volumn(f14, f24_warp)
        flow4, _ = self.decoder4(tf.concat([cv4, f14, flow54], axis=3))
        flow4 = flow4 + flow54

        # Block3
        flow43 = 2.0 * \
            tf.image.resize(
                flow4, self.shape[3], method=tf.image.ResizeMethod.BILINEAR)
        f23_warp = transformer_old(f23, flow43, self.shape[3])
        cv3 = self.cost_volumn(f13, f23_warp)
        flow3, _ = self.decoder3(tf.concat([cv3, f13, flow43], axis=3))
        flow3 = flow3 + flow43

        # Block2
        flow32 = 2.0 * \
            tf.image.resize(
                flow3, self.shape[2], method=tf.image.ResizeMethod.BILINEAR)
        f22_warp = transformer_old(f22, flow32, self.shape[2])
        cv2 = self.cost_volumn(f12, f22_warp)
        flow2, flow2_ = self.decoder2(tf.concat([cv2, f12, flow32], axis=3))
        flow2 = flow2 + flow32

        # context_net
        flow2 = self.context_net(tf.concat([flow2, flow2_], axis=3)) + flow2

        flow0_enlarge = tf.image.resize(
            flow2 * 4.0, self.shape[0], method=tf.image.ResizeMethod.BILINEAR)
        flow1_enlarge = tf.image.resize(
            flow3 * 4.0, self.shape[1], method=tf.image.ResizeMethod.BILINEAR)
        flow2_enlarge = tf.image.resize(
            flow4 * 4.0, self.shape[2], method=tf.image.ResizeMethod.BILINEAR)
        flow3_enlarge = tf.image.resize(
            flow5 * 4.0, self.shape[3], method=tf.image.ResizeMethod.BILINEAR)

        return flow0_enlarge, flow1_enlarge, flow2_enlarge, flow3_enlarge


class cost_volumn(tf.keras.layers.Layer):
    def __init__(self, d=4):
        super().__init__()
        self.d = d

    def call(self, feature1, feature2):
        n, h, w, c = feature1.get_shape().as_list()
        feature2 = tf.pad(tensor=feature2, paddings=[[0, 0], [self.d, self.d], [
                          self.d, self.d], [0, 0]], mode="CONSTANT")
        cv = []
        for i in range(2 * self.d + 1):
            for j in range(2 * self.d + 1):
                cv.append(
                    tf.math.reduce_mean(
                        input_tensor=feature1 *
                        feature2[:, i:(i + h), j:(j + w), :],
                        axis=3,
                        keepdims=True))
        x = tf.concat(cv, axis=3)
        return x


class pwc_decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = tf.keras.layers.conv2d(
            128, 3, strides=1, activation=leaky_relu)
        self.conv2d_2 = tf.keras.layers.conv2d(
            128, 3, strides=1, activation=leaky_relu)
        self.conv2d_3 = tf.keras.layers.conv2d(
            96, 3, strides=1, activation=leaky_relu)
        self.conv2d_4 = tf.keras.layers.conv2d(
            64, 3, strides=1, activation=leaky_relu)
        self.conv2d_5 = tf.keras.layers.conv2d(
            32, 3, strides=1, activation=leaky_relu)
        self.conv2d_6 = tf.keras.layers.conv2d(2, 3, strides=1)

    def call(self, inputs):
        cnv1 = self.conv2d_1(inputs)
        cnv2 = self.conv2d_2(cnv1)
        cnv3 = self.conv2d_3(tf.concat([cnv1, cnv2], axis=3))
        cnv4 = self.conv2d_4(tf.concat([cnv2, cnv3], axis=3))
        cnv5 = self.conv2d_5(tf.concat([cnv3, cnv4], axis=3))
        cnv6 = self.conv2d_6(tf.concat([cnv4, cnv5], axis=3))
        return cnv6, cnv5


class context_net(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = tf.keras.layers.conv2d(
            128, 3, strides=1, dilation_rate=1, activation=leaky_relu)
        self.conv2d_2 = tf.keras.layers.conv2d(
            128, 3, strides=1, dilation_rate=2, activation=leaky_relu)
        self.conv2d_3 = tf.keras.layers.conv2d(
            128, 3, strides=1, dilation_rate=4, activation=leaky_relu)
        self.conv2d_4 = tf.keras.layers.conv2d(
            96, 3, strides=1, dilation_rate=8, activation=leaky_relu)
        self.conv2d_5 = tf.keras.layers.conv2d(
            64, 3, strides=1, dilation_rate=16, activation=leaky_relu)
        self.conv2d_6 = tf.keras.layers.conv2d(
            32, 3, strides=1, dilation_rate=1, activation=leaky_relu)
        self.conv2d_7 = tf.keras.layers.conv2d(
            2, 3, strides=1, dilation_rate=1)

    def call(self, inputs):
        cnv1 = self.conv2d_1(inputs)
        cnv2 = self.conv2d_2(cnv1)
        cnv3 = self.conv2d_3(cnv2)
        cnv4 = self.conv2d_4(cnv3)
        cnv5 = self.conv2d_5(cnv4)
        cnv6 = self.conv2d_6(cnv5)
        cnv7 = self.conv2d_6(cnv6)
        return cnv7


def get_allShape(image):
    ''' 列出6種 size的 hight & width
    list[6]: 0->origin size, 3-> (origin // 2 ** 3), ...'''
    n, h, w, c = image.get_shape().as_list()
    shape = []
    for i in range(6):
        shape.append([h // 2**i, w // 2**i])
    return shape


def leaky_relu(_x, alpha=0.1):
    pos = tf.nn.relu(_x)
    neg = alpha * (_x - abs(_x)) * 0.5

    return pos + neg
