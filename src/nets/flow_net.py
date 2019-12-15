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
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn
from models.common import conv1x1, depthwise_conv3x3, conv1x1_block, conv3x3_block, ChannelShuffle, SEBlock,\
    GluonBatchNormalization, MaxPool2d, get_channel_axis, flatten, dwconv3x3_block
# tf.compat.v1.disable_v2_behavior()

def flow_net(input_shape):
    src0 = tf.keras.Input(shape=input_shape, name='input_src0')
    tgt = tf.keras.Input(shape=input_shape, name='input_tgt')
    src1 = tf.keras.Input(shape=input_shape, name='input_src1')
    
    fpf = feature_pyramid_flow(input_shape)
    fpf.summary()
    
    feature_src0 = fpf(src0)
    feature_tgt = fpf(tgt)
    feature_src1 = fpf(src1)
    
    flow = get_flow(input_shape)
    src0_0, src0_1, src0_2, src0_3, src0_4, src0_5, src0_6 = feature_src0
    tgt_0, tgt_1, tgt_2, tgt_3, tgt_4, tgt_5, tgt_6 = feature_tgt
    src1_0, src1_1, src1_2, src1_3, src1_4, src1_5, src1_6 = feature_src1
    
    
     # foward warp: |01 |, | 12|, |0 2| ,direction: ->
    flow_fw0 = flow(src0_0, src0_1, src0_2, src0_3, src0_4, src0_5, src0_6, 
                    tgt_0, tgt_1, tgt_2, tgt_3, tgt_4, tgt_5, tgt_6)
    flow_fw1 = flow(tgt_0, tgt_1, tgt_2, tgt_3, tgt_4, tgt_5, tgt_6, 
                    src1_0, src1_1, src1_2, src1_3, src1_4, src1_5, src1_6)
    flow_fw2 = flow(src0_0, src0_1, src0_2, src0_3, src0_4, src0_5, src0_6, 
                    src1_0, src1_1, src1_2, src1_3, src1_4, src1_5, src1_6)

    # backward warp: |01 |, | 12|, |0 2| , direction: <-
    flow_bw0 = flow(tgt_0, tgt_1, tgt_2, tgt_3, tgt_4, tgt_5, tgt_6, 
                    src0_0, src0_1, src0_2, src0_3, src0_4, src0_5, src0_6)
    flow_bw1 = flow(src1_0, src1_1, src1_2, src1_3, src1_4, src1_5, src1_6, 
                    tgt_0, tgt_1, tgt_2, tgt_3, tgt_4, tgt_5, tgt_6)
    flow_bw2 = flow(src1_0, src1_1, src1_2, src1_3, src1_4, src1_5, src1_6, 
                    src0_0, src0_1, src0_2, src0_3, src0_4, src0_5, src0_6)
    
    model = tf.keras.Model(inputs=[src0, tgt, src1], 
                           outputs=[flow_fw0, flow_fw1, flow_fw2,
                                    flow_bw0, flow_bw1, flow_bw2], 
                           name='flow_net')
    model.summary()
    
    return model

def get_allShape(input_shape, d=4):
        ''' 列出6種 size的 hight & width
        list[6]: 0->origin size, ..., 5-> (origin // 2 ** 5)'''
        h, w, c = input_shape
        shape = []
        for i in range(6):
            shape.append([h // 2**i, w // 2**i])
        
        cv_channel = (2 * d + 1)**2
        decoder_shape = [(h // 2**2, w // 2**2,  32+cv_channel+2),   
                        (h // 2**3, w // 2**3,  64+cv_channel+2),  
                        (h // 2**4, w // 2**4,  96+cv_channel+2),  
                        (h // 2**5, w // 2**5, 128+cv_channel+2),   
                        (h // 2**6, w // 2**6, cv_channel)]     

        return shape, decoder_shape

def get_flow(input_shape):
    d = 4
    shape, decoder_shape = get_allShape(input_shape, d=d)
    
    f11 = tf.keras.Input(shape=np.asarray(input_shape) // 2**1, name='f11')
    f12 = tf.keras.Input(shape=np.asarray(input_shape) // 2**2, name='f12')
    f13 = tf.keras.Input(shape=np.asarray(input_shape) // 2**3, name='f13')
    f14 = tf.keras.Input(shape=np.asarray(input_shape) // 2**4, name='f14')
    f15 = tf.keras.Input(shape=np.asarray(input_shape) // 2**5, name='f15')
    f16 = tf.keras.Input(shape=np.asarray(input_shape) // 2**6, name='f16')
    f21 = tf.keras.Input(shape=np.asarray(input_shape) // 2**1, name='f21')
    f22 = tf.keras.Input(shape=np.asarray(input_shape) // 2**2, name='f22')
    f23 = tf.keras.Input(shape=np.asarray(input_shape) // 2**3, name='f23')
    f24 = tf.keras.Input(shape=np.asarray(input_shape) // 2**4, name='f24')
    f25 = tf.keras.Input(shape=np.asarray(input_shape) // 2**5, name='f25')
    f26 = tf.keras.Input(shape=np.asarray(input_shape) // 2**6, name='f26')
    
    cost_vol = cost_volumn(d=d)
    context = context_net()
    decoder2 = pwc_decoder(input_shape=decoder_shape[0])
    decoder3 = pwc_decoder(input_shape=decoder_shape[1])
    decoder4 = pwc_decoder(input_shape=decoder_shape[2])
    decoder5 = pwc_decoder(input_shape=decoder_shape[3])
    decoder6 = pwc_decoder(input_shape=decoder_shape[4])
    
    
    # Block6
    cv6 = cost_vol(f16, f26)
    flow6, _ = decoder6(cv6)

    # Block5
    flow65 = tf.scalar_mul(2, tf.image.resize(flow6, shape[5], method=tf.image.ResizeMethod.BILINEAR))
    print('#'*50)
    f25_warp = transformer_old(f25, flow65, shape[5])
    print('#'*50)
    cv5 = cost_vol(f15, f25_warp)
    flow5, _ = decoder5(tf.concat([cv5, f15, flow65], axis=3)) #2
    flow5 = tf.math.add(flow5, flow65)

    # Block4
    flow54 = tf.scalar_mul(2.0, tf.image.resize(flow5, shape[4], method=tf.image.ResizeMethod.BILINEAR))
    f24_warp = transformer_old(f24, flow54, shape[4])
    cv4 = cost_vol(f14, f24_warp)
    flow4, _ = decoder4(tf.concat([cv4, f14, flow54], axis=3)) #2
    flow4 = tf.math.add(flow4, flow54)

    # Block3
    flow43 = tf.scalar_mul(2.0 ,tf.image.resize(flow4, shape[3],
                                             method=tf.image.ResizeMethod.BILINEAR))
    f23_warp = transformer_old(f23, flow43, shape[3])
    cv3 = cost_vol(f13, f23_warp)
    flow3, _ = decoder3(tf.concat([cv3, f13, flow43], axis=3)) #2
    flow3 = tf.math.add(flow3, flow43)
    # Block2
    flow32 = tf.scalar_mul(2.0, tf.image.resize(flow3, shape[2], method=tf.image.ResizeMethod.BILINEAR))
    f22_warp = transformer_old(f22, flow32, shape[2])
    cv2 = cost_vol(f12, f22_warp)
    flow2, flow2_ = decoder2(tf.concat([cv2, f12, flow32], axis=3)) #2
    flow2 = tf.math.add(flow2, flow32) #10
    
    # context_net
    flow2 = context(tf.concat([flow2, flow2_], axis=3))
    flow2 = tf.math.add(flow2, flow2)
    
    flow0_enlarge = tf.image.resize(
            tf.scalar_mul(4.0,flow2), self.shape[0], method=tf.image.ResizeMethod.BILINEAR)
    flow1_enlarge = tf.image.resize(
            tf.scalar_mul(4.0,flow3), self.shape[1], method=tf.image.ResizeMethod.BILINEAR)
    flow2_enlarge = tf.image.resize(
            tf.scalar_mul(4.0,flow4), self.shape[2], method=tf.image.ResizeMethod.BILINEAR)
    flow3_enlarge = tf.image.resize(
            tf.scalar_mul(4.0,flow5), self.shape[3], method=tf.image.ResizeMethod.BILINEAR)
    
    model = tf.keras.Model(inputs=[f11, f12, f13, f14, f15, f16,
                                  f21, f22, f23, f24, f25, f26], 
                           outputs=[flow0_enlarge, flow1_enlarge, flow2_enlarge, flow3_enlarge], 
                           name='flow_net')
    model.summary()
    
    return model
    

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

def feature_pyramid_flow(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name='input_image')
    
    cnv1 = conv3x3_block(3, 16, strides=2, padding=1, use_bn=False, activation='leaky_relu')(inputs)
    cnv2 = conv3x3_block(16, 16, use_bn=False, activation='leaky_relu')(cnv1)
    cnv3 = conv3x3_block(16, 32, strides=2, padding=1, use_bn=False, activation='leaky_relu')(cnv2)
    cnv4 = conv3x3_block(32, 32, use_bn=False, activation='leaky_relu')(cnv3)
    cnv5 = conv3x3_block(32, 64, strides=2, padding=1, use_bn=False, activation='leaky_relu')(cnv4)
    cnv6 = conv3x3_block(64, 64, use_bn=False, activation='leaky_relu')(cnv5)
    cnv7 = conv3x3_block(16, 96, strides=2, padding=1, use_bn=False, activation='leaky_relu')(cnv6)
    cnv8 = conv3x3_block(32, 96, use_bn=False, activation='leaky_relu')(cnv7)
    cnv9 = conv3x3_block(16, 128, strides=2, padding=1, use_bn=False, activation='leaky_relu')(cnv8)
    cnv10 = conv3x3_block(32, 128, use_bn=False, activation='leaky_relu')(cnv9)
    cnv11 = conv3x3_block(16, 192, strides=2, padding=1, use_bn=False, activation='leaky_relu')(cnv10)
    cnv12 = conv3x3_block(32, 192, use_bn=False, activation='leaky_relu')(cnv11)
    
    layers = tf.keras.Model(inputs=inputs, 
                            outputs=[cnv2, cnv4, cnv6, cnv8, cnv10, cnv12],
                            name='feature_pyramid_flow')
    return layers    
    
def pwc_decoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name='input_image')
    
    cnv1 = conv3x3_block(128, 128, use_bn=False, activation='leaky_relu')(inputs)
    cnv2 = conv3x3_block(128, 128, use_bn=False, activation='leaky_relu')(cnv1)
    cnv3 = conv3x3_block(128, 96, use_bn=False, activation='leaky_relu')(nn.concatenate([cnv1, cnv2]))
    cnv4 = conv3x3_block(96, 64, use_bn=False, activation='leaky_relu')(nn.concatenate([cnv2, cnv3]))
    cnv5 = conv3x3_block(64, 32, use_bn=False, activation='leaky_relu')(nn.concatenate([cnv3, cnv4]))
    cnv6 = conv3x3_block(32, 2, use_bn=False)(nn.concatenate([cnv4, cnv5]))
    layers = tf.keras.Model(inputs=inputs, outputs=[cnv6, cnv5], name='pwc_decoder')
    return layers
    
def context_net():
    layers = tf.keras.Sequential(name='context_net')
    layers.add(nn.Conv2D(128, 3, dilation_rate=1, padding='same', use_bias=False, activation=nn.LeakyReLU(alpha=0.1)))
    layers.add(nn.Conv2D(128, 3, dilation_rate=2, padding='same', use_bias=False, activation=nn.LeakyReLU(alpha=0.1)))
    layers.add(nn.Conv2D(128, 3, dilation_rate=4, padding='same', use_bias=False, activation=nn.LeakyReLU(alpha=0.1)))
    layers.add(nn.Conv2D(96, 3, dilation_rate=8, padding='same', use_bias=False, activation=nn.LeakyReLU(alpha=0.1)))
    layers.add(nn.Conv2D(64, 3, dilation_rate=16, padding='same', use_bias=False, activation=nn.LeakyReLU(alpha=0.1)))
    layers.add(nn.Conv2D(32, 3, dilation_rate=1, padding='same', use_bias=False, activation=nn.LeakyReLU(alpha=0.1)))
    layers.add(nn.Conv2D(2, 3, dilation_rate=1, padding='same', use_bias=False, activation=None))
    
    return layers





def leaky_relu(_x, alpha=0.1):
    pos = tf.nn.relu(_x)
    neg = alpha * (_x - abs(_x)) * 0.5

    return pos + neg



###########################################
###################################################
###########################################################
def transformer_old(U, flo, out_size, name='SpatialTransformer', **kwargs):
    """Backward warping layer

    Implements a backward warping layer described in 
    "Unsupervised Deep Learning for Optical Flow Estimation, Zhe Ren et al"

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    flo: float
         The optical flow used to do the backward warping.
         shape is [num_batch, height, width, 2]
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        with tf.compat.v1.variable_scope('_repeat'):
            rep = tf.transpose(
                a=tf.expand_dims(
                    tf.ones(shape=tf.stack([n_repeats, ])), 1), perm=[1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.compat.v1.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(input=im)[0]
            height = tf.shape(input=im)[1]
            width = tf.shape(input=im)[2]
            channels = tf.shape(input=im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(input=im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(input=im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f - 1) / 2.0
            y = (y + 1.0) * (height_f - 1) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0_c = tf.clip_by_value(x0, zero, max_x)
            x1_c = tf.clip_by_value(x1, zero, max_x)
            y0_c = tf.clip_by_value(y0, zero, max_y)
            y1_c = tf.clip_by_value(y1, zero, max_y)

            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)

            base_y0 = base + y0_c * dim2
            base_y1 = base + y1_c * dim2
            idx_a = base_y0 + x0_c
            idx_b = base_y1 + x0_c
            idx_c = base_y0 + x1_c
            idx_d = base_y1 + x1_c

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _meshgrid(height, width):
        with tf.compat.v1.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(
                tf.ones(shape=tf.stack([height, 1])),
                tf.transpose(
                    a=tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0]))
            y_t = tf.matmul(
                tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                tf.ones(shape=tf.stack([1, width])))

            return x_t, y_t

    def _transform(flo, input_dim, out_size):
        with tf.compat.v1.variable_scope('_transform'):
            num_batch = tf.shape(input=input_dim)[0]
            height = tf.shape(input=input_dim)[1]
            width = tf.shape(input=input_dim)[2]
            num_channels = tf.shape(input=input_dim)[3]

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            x_t, y_t = _meshgrid(out_height, out_width)
            x_t = tf.expand_dims(x_t, 0)
            x_t = tf.tile(x_t, [num_batch, 1, 1])

            y_t = tf.expand_dims(y_t, 0)
            y_t = tf.tile(y_t, [num_batch, 1, 1])

            x_s = x_t + flo[:, :, :, 0] / (
                (tf.cast(out_width, tf.float32) - 1.0) / 2.0)
            y_s = y_t + flo[:, :, :, 1] / (
                (tf.cast(out_height, tf.float32) - 1.0) / 2.0)

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat,
                                             out_size)

            output = tf.reshape(
                input_transformed,
                tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.compat.v1.variable_scope(name):
        output = _transform(flo, U, out_size)
        return output


if __name__ == '__main__':
    batch_size = 10
    src0 = tf.random.normal((batch_size, 256, 832, 3))
    tgt = tf.random.normal((batch_size, 256, 832, 3))
    src1 = tf.random.normal((batch_size, 256, 832, 3))
    
    net = flow_net(input_shape=(256, 832, 3))
    
    for var in net.trainable_variables:
        print(var.name)
#     for wei in net.weights:
#         print(wei)
    
    for layer in net.layers:
        print(layer.name)

