# The network design is based on Tinghui Zhou & Clement Godard's works:
# https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
# https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


# Disparity (inverse depth) values range from 0.01 to 10. Note that effectively,
# this is undone if depth normalization is used, which scales the values to
# have a mean of 1.
DISP_SCALING_RESNET50 = 5
DISP_SCALING_RESNET18 = 10
DISP_SCALING_VGG = 10
MIN_DISP = 0.01
WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'


def D_Net(dispnet_inputs, weight_reg=0.0004, is_training=True, reuse=False):
#     return build_resnet50(dispnet_inputs, get_disp_resnet50, is_training, 'depth_net', reuse=reuse)
    # return build_resnet18(dispnet_inputs, is_training, 'depth_net', reuse, weight_reg)
    # return build_vgg(dispnet_inputs, get_disp_vgg, is_training, 'depth_net', reuse, get_feature)
    D_Model = ShuffleNetV2(input_holder=dispnet_inputs, var_scope='depth_net', model_scale=1.0, shuffle_group=2, is_training=True)
    return D_Model.build_model()

######################################
# vgg
######################################
def build_vgg(inputs, get_pred, is_training, var_scope, reuse=False):
    batch_norm_params = {'is_training': is_training}
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    with tf.variable_scope(var_scope) as sc:
        if reuse:
            sc.reuse_variables()
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.elu):
            # ENCODING
            conv1  = slim.conv2d(inputs, 32,  7, 2)
            conv1b = slim.conv2d(conv1,  32,  7, 1)
            conv2  = slim.conv2d(conv1b, 64,  5, 2)
            conv2b = slim.conv2d(conv2,  64,  5, 1)
            conv3  = slim.conv2d(conv2b, 128, 3, 2)
            conv3b = slim.conv2d(conv3,  128, 3, 1)
            conv4  = slim.conv2d(conv3b, 256, 3, 2)
            conv4b = slim.conv2d(conv4,  256, 3, 1)
            conv5  = slim.conv2d(conv4b, 512, 3, 2)
            conv5b = slim.conv2d(conv5,  512, 3, 1)
            conv6  = slim.conv2d(conv5b, 512, 3, 2)
            conv6b = slim.conv2d(conv6,  512, 3, 1)
            conv7  = slim.conv2d(conv6b, 512, 3, 2)
            conv7b = slim.conv2d(conv7,  512, 3, 1)

            # DECODING
            upconv7 = upconv(conv7b, 512, 3, 2)
            # There might be dimension mismatch due to uneven down/up-sampling
            upconv7 = resize_like(upconv7, conv6b)
            i7_in  = tf.concat([upconv7, conv6b], axis=3)
            iconv7  = slim.conv2d(i7_in, 512, 3, 1)
            #
            # print("[upconv7]:", upconv7)
            # print("[i7_in]:", i7_in)
            # print("[iconv7]:", iconv7)

            upconv6 = upconv(iconv7, 512, 3, 2)
            upconv6 = resize_like(upconv6, conv5b)
            i6_in  = tf.concat([upconv6, conv5b], axis=3)
            iconv6  = slim.conv2d(i6_in, 512, 3, 1)

            # print("[upconv6]:", upconv6)
            # print("[i6_in]:", i6_in)
            # print("[iconv6]:", iconv6)

            upconv5 = upconv(iconv6, 256, 3, 2)
            upconv5 = resize_like(upconv5, conv4b)
            i5_in  = tf.concat([upconv5, conv4b], axis=3)
            iconv5  = slim.conv2d(i5_in, 256, 3, 1)
            #
            # print("[upconv5]:", upconv5)
            # print("[i5_in]:", i5_in)
            # print("[iconv5]:", iconv5)

            upconv4 = upconv(iconv5, 128, 3, 2)
            i4_in  = tf.concat([upconv4, conv3b], axis=3)
            iconv4  = slim.conv2d(i4_in, 128, 3, 1)
            pred4  = get_pred(iconv4)
            pred4_up = tf.image.resize_bilinear(pred4, [np.int(H/4), np.int(W/4)])
            #
            # print("[upconv4]:", upconv4)
            # print("[i4_in]:", i4_in)
            # print("[iconv4]:", iconv4)
            # print("[pred4]:", pred4)
            # print("[pred4_up]:", pred4_up)

            upconv3 = upconv(iconv4, 64,  3, 2)
            i3_in  = tf.concat([upconv3, conv2b, pred4_up], axis=3)
            iconv3  = slim.conv2d(i3_in, 64,  3, 1)
            pred3  = get_pred(iconv3)
            pred3_up = tf.image.resize_bilinear(pred3, [np.int(H/2), np.int(W/2)])
            # print("[upconv3]:", upconv3)
            # print("[i3_in]:", i3_in)
            # print("[iconv3]:", iconv3)
            # print("[pred3]:", pred3)
            # print("[pred3_up]:", pred3_up)

            upconv2 = upconv(iconv3, 32,  3, 2)
            i2_in  = tf.concat([upconv2, conv1b, pred3_up], axis=3)
            iconv2  = slim.conv2d(i2_in, 32,  3, 1)
            pred2  = get_pred(iconv2)
            pred2_up = tf.image.resize_bilinear(pred2, [H, W])
            # print("[upconv2]:", upconv2)
            # print("[i2_in]:", i2_in)
            # print("[iconv2]:", iconv2)
            # print("[pred2]:", pred2)
            # print("[pred2_up]:", pred2_up)

            upconv1 = upconv(iconv2, 16,  3, 2)
            i1_in  = tf.concat([upconv1, pred2_up], axis=3)
            iconv1  = slim.conv2d(i1_in, 16,  3, 1)
            pred1  = get_pred(iconv1)

            # print("[upconv1]:", upconv1)
            # print("[i1_in]:", i1_in)
            # print("[iconv1]:", iconv1)
            # print("[pred1]:", pred1)

            return [pred1, pred2, pred3, pred4], iconv1

def get_disp_vgg(x):
    disp = DISP_SCALING_VGG * slim.conv2d(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

######################################
# Resnet18
######################################
def build_resnet18(inputs, is_training, var_scope, reuse=False, weight_reg=0.0004):
    """Defines a ResNet18-based encoding architecture.

    This implementation follows Juyong Kim's implementation of ResNet18 on GitHub:
    https://github.com/dalgu90/resnet-18-tensorflow

    Args:
      target_image: Input tensor with shape [B, h, w, 3] to encode.
      weight_reg: Parameter ignored.
      is_training: Whether the model is being trained or not.

    Returns:
      Tuple of tensors, with the first being the bottleneck layer as tensor of
      size [B, h_hid, w_hid, c_hid], and others being intermediate layers
      for building skip-connections.
    """
    with tf.variable_scope(var_scope) as sc:
        if reuse:
            sc.reuse_variables()
        bottleneck, skip_connections = resnet18_encoder(inputs, is_training)

        # Decode to depth.
        multiscale_disps_i = resnet18_decoder(target_image=inputs,
                                              bottleneck=bottleneck,
                                              use_skip=True,
                                              skip_connections=skip_connections,
                                              weight_reg=weight_reg)
        return multiscale_disps_i, bottleneck

def resnet18_encoder(target_image, is_training):
    encoder_filters = [64, 64, 128, 256, 512]
    stride = 2

    # conv1
    with tf.variable_scope('conv1'):
        x = _conv(target_image, 7, encoder_filters[0], stride)
        x = _bn(x, is_train=is_training)
        econv1 = _relu(x)
        x = tf.nn.max_pool(econv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    # conv2_x
    x = _residual_block(x, is_training, name='conv2_1')
    econv2 = _residual_block(x, is_training, name='conv2_2')

    # conv3_x
    x = _residual_block_first(econv2, is_training, encoder_filters[2], stride,
                              name='conv3_1')
    econv3 = _residual_block(x, is_training, name='conv3_2')

    # conv4_x
    x = _residual_block_first(econv3, is_training, encoder_filters[3], stride,
                              name='conv4_1')
    econv4 = _residual_block(x, is_training, name='conv4_2')

    # conv5_x
    x = _residual_block_first(econv4, is_training, encoder_filters[4], stride,
                              name='conv5_1')
    econv5 = _residual_block(x, is_training, name='conv5_2')
    return econv5, (econv4, econv3, econv2, econv1)

def resnet18_decoder(target_image, bottleneck, use_skip, skip_connections, weight_reg):
    (econv4, econv3, econv2, econv1) = skip_connections
    decoder_filters = [16, 32, 64, 128, 256]
    default_pad = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    reg = slim.l2_regularizer(weight_reg) if weight_reg > 0.0 else None
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=None,
                        normalizer_params=None,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=reg):

        upconv5 = slim.conv2d_transpose(bottleneck, decoder_filters[4], [3, 3],
                                        stride=2, scope='upconv5')
        upconv5 = _resize_like(upconv5, econv4)
        if use_skip:
            i5_in = tf.concat([upconv5, econv4], axis=3)
        else:
            i5_in = upconv5
        i5_in = tf.pad(i5_in, default_pad, mode='REFLECT')
        iconv5 = slim.conv2d(i5_in, decoder_filters[4], [3, 3], stride=1,
                             scope='iconv5', padding='VALID')

        upconv4 = slim.conv2d_transpose(iconv5, decoder_filters[3], [3, 3],
                                        stride=2, scope='upconv4')
        upconv4 = _resize_like(upconv4, econv3)
        if use_skip:
            i4_in = tf.concat([upconv4, econv3], axis=3)
        else:
            i4_in = upconv4
        i4_in = tf.pad(i4_in, default_pad, mode='REFLECT')
        iconv4 = slim.conv2d(i4_in, decoder_filters[3], [3, 3], stride=1,
                             scope='iconv4', padding='VALID')

        disp4_input = tf.pad(iconv4, default_pad, mode='REFLECT')
        disp4 = (slim.conv2d(disp4_input, 1, [3, 3], stride=1,
                             activation_fn=tf.sigmoid, normalizer_fn=None,
                             scope='disp4', padding='VALID')
                 * DISP_SCALING_RESNET18 + MIN_DISP)



        upconv3 = slim.conv2d_transpose(iconv4, decoder_filters[2], [3, 3],
                                        stride=2, scope='upconv3')
        upconv3 = _resize_like(upconv3, econv2)
        if use_skip:
            i3_in = tf.concat([upconv3, econv2], axis=3)
        else:
            i3_in = upconv3
        i3_in = tf.pad(i3_in, default_pad, mode='REFLECT')
        iconv3 = slim.conv2d(i3_in, decoder_filters[2], [3, 3], stride=1,
                             scope='iconv3', padding='VALID')
        disp3_input = tf.pad(iconv3, default_pad, mode='REFLECT')
        disp3 = (slim.conv2d(disp3_input, 1, [3, 3], stride=1,
                             activation_fn=tf.sigmoid, normalizer_fn=None,
                             scope='disp3', padding='VALID')
                 * DISP_SCALING_RESNET18 + MIN_DISP)

        upconv2 = slim.conv2d_transpose(iconv3, decoder_filters[1], [3, 3],
                                        stride=2, scope='upconv2')
        upconv2 = _resize_like(upconv2, econv1)
        if use_skip:
            i2_in = tf.concat([upconv2, econv1], axis=3)
        else:
            i2_in = upconv2
        i2_in = tf.pad(i2_in, default_pad, mode='REFLECT')
        iconv2 = slim.conv2d(i2_in, decoder_filters[1], [3, 3], stride=1,
                             scope='iconv2', padding='VALID')
        disp2_input = tf.pad(iconv2, default_pad, mode='REFLECT')
        disp2 = (slim.conv2d(disp2_input, 1, [3, 3], stride=1,
                             activation_fn=tf.sigmoid, normalizer_fn=None,
                             scope='disp2', padding='VALID')
                 * DISP_SCALING_RESNET18 + MIN_DISP)

        upconv1 = slim.conv2d_transpose(iconv2, decoder_filters[0], [3, 3],
                                        stride=2, scope='upconv1')
        upconv1 = _resize_like(upconv1, target_image)
        upconv1 = tf.pad(upconv1, default_pad, mode='REFLECT')
        iconv1 = slim.conv2d(upconv1, decoder_filters[0], [3, 3], stride=1,
                             scope='iconv1', padding='VALID')
        disp1_input = tf.pad(iconv1, default_pad, mode='REFLECT')
        disp1 = (slim.conv2d(disp1_input, 1, [3, 3], stride=1,
                             activation_fn=tf.sigmoid, normalizer_fn=None,
                             scope='disp1', padding='VALID')
                 * DISP_SCALING_RESNET18 + MIN_DISP)

    return [disp1, disp2, disp3, disp4]

### other utils

def _residual_block_first(x, is_training, out_channel, strides, name='unit'):
  """Helper function for defining ResNet architecture."""
  in_channel = x.get_shape().as_list()[-1]
  with tf.variable_scope(name):
    # Shortcut connection
    if in_channel == out_channel:
      if strides == 1:
        shortcut = tf.identity(x)
      else:
        shortcut = tf.nn.max_pool(x, [1, strides, strides, 1],
                                  [1, strides, strides, 1], 'VALID')
    else:
      shortcut = _conv(x, 1, out_channel, strides, name='shortcut')
    # Residual
    x = _conv(x, 3, out_channel, strides, name='conv_1')
    x = _bn(x, is_train=is_training, name='bn_1')
    x = _relu(x, name='relu_1')
    x = _conv(x, 3, out_channel, 1, name='conv_2')
    x = _bn(x, is_train=is_training, name='bn_2')
    # Merge
    x = x + shortcut
    x = _relu(x, name='relu_2')
  return x

def _residual_block(x, is_training, input_q=None, output_q=None, name='unit'):
    """Helper function for defining ResNet architecture."""
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        shortcut = x  # Shortcut connection
        # Residual
        x = _conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q,
                  name='conv_1')
        x = _bn(x, is_train=is_training, name='bn_1')
        x = _relu(x, name='relu_1')
        x = _conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q,
                  name='conv_2')
        x = _bn(x, is_train=is_training, name='bn_2')
        # Merge
        x = x + shortcut
        x = _relu(x, name='relu_2')
    return x

def _conv(x, filter_size, out_channel, stride, pad='SAME', input_q=None, output_q=None, name='conv'):
    """Helper function for defining ResNet architecture."""
    if (input_q is None) ^ (output_q is None):
        raise ValueError('Input/Output splits are not correctly given.')

    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable(
                'kernel', [filter_size, filter_size, in_shape[3], out_channel],
                tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], pad)
    return conv

def _bn(x, is_train, name='bn'):
    """Helper function for defining ResNet architecture."""
    bn = tf.layers.batch_normalization(x, training=is_train, name=name)
    return bn

def _relu(x, name=None, leakness=0.0):
    """Helper function for defining ResNet architecture."""
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')

def _resize_like(inputs, ref):
    i_h, i_w = inputs.get_shape()[1], inputs.get_shape()[2]
    r_h, r_w = ref.get_shape()[1], ref.get_shape()[2]
    if i_h == r_h and i_w == r_w:
        return inputs
    else:
        # TODO(casser): Other interpolation methods could be explored here.
        return tf.image.resize_bilinear(inputs, [r_h.value, r_w.value],
                                        align_corners=True)

######################################
# Resnet50
######################################
def build_resnet50(inputs, get_pred, is_training, var_scope, weight_reg=0.0001, reuse=False):
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope(var_scope) as sc:
        if reuse:
            sc.reuse_variables()
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_reg),
                            activation_fn=tf.nn.elu):
            conv1 = conv(inputs, 64, 7, 2)      # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4


            # DECODING
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            upconv6 = resize_like(upconv6, skip5)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            upconv5 = resize_like(upconv5, skip4)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            upconv4 = resize_like(upconv4, skip3)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            pred4 = get_pred(iconv4)
            upred4  = upsample_nn(pred4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, upred4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            pred3 = get_pred(iconv3)
            upred3  = upsample_nn(pred3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, upred3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            pred2 = get_pred(iconv2)
            upred2  = upsample_nn(pred2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, upred2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            pred1 = get_pred(iconv1)

            return [pred1, pred2, pred3, pred4], conv5

def get_disp_resnet50(x):
    disp = DISP_SCALING_RESNET50 * conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

def resconv(x, num_layers, stride):
    # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
    # but we preserve it here for consistency with Godard's implementation.
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1)
    conv2 = conv(conv1,     num_layers, 3, stride)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.elu(conv3 + shortcut)

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode='REFLECT')
    return slim.max_pool2d(p_x, kernel_size)

######################################
# ShuffleNetV2
######################################

class ShuffleNetV2():

    first_conv_channel = 24
    
    def __init__(self, input_holder, var_scope, model_scale=1.0, shuffle_group=2, is_training=True):
        self.input = input_holder
        self.output = None
        self.shuffle_group = shuffle_group
        self.channel_sizes = self._select_channel_size(model_scale)
        self.var_scope = var_scope
        self.is_training = is_training

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
        with tf.variable_scope(self.var_scope) as sc:
            with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                skip = []
                with tf.variable_scope('encoding'):
                    with tf.variable_scope('init_block'):
                        out = conv_bn_relu(self.input, self.first_conv_channel, 3, 2)
                        skip.append(out)
                        out = slim.max_pool2d(skip[0], 3, 2, padding='SAME')
                        skip.append(out)
                    for idx, block in enumerate(self.channel_sizes[:-1]):
                        with tf.variable_scope('shuffle_block_{}'.format(idx)):
                            out_channel, repeat = block

                            # First block is downsampling
                            out = shufflenet_v2_block(out, out_channel, 3, 2, shuffle_group=self.shuffle_group)

                            # Rest blocks
                            for i in range(repeat-1):
                                out = shufflenet_v2_block(out, out_channel, 3, shuffle_group=self.shuffle_group)

                            skip.append(out)


                    with tf.variable_scope('end_block'):
                        out = conv_bn_relu(out, self.channel_sizes[-1][0], 1)
                        skip.append(out)

                with tf.variable_scope('decoding'):
                    # DECODING
                    upconv6 = upconv_sep(skip[5],   512, 3, 2) #H/32
                    upconv6 = resize_like(upconv6, skip[4])
                    concat6 = tf.concat([upconv6, skip[4]], 3)
#                     iconv6  = conv(concat6,   512, 3, 1)
                    iconv6  = shufflenet_v2_block(concat6, 512, 3, stride=1, dilation=1, shuffle_group=2)
                    
                    upconv5 = upconv_sep(iconv6, 256, 3, 2) #H/16
                    upconv5 = resize_like(upconv5, skip[3])
                    concat5 = tf.concat([upconv5, skip[3]], 3)
#                     iconv5  = conv(concat5,   256, 3, 1)
                    iconv5  = shufflenet_v2_block(concat5, 256, 3, stride=1, dilation=1, shuffle_group=2)

                    upconv4 = upconv_sep(iconv5,  128, 3, 2) #H/8
                    upconv4 = resize_like(upconv4, skip[2])
                    concat4 = tf.concat([upconv4, skip[2]], 3)
#                     iconv4  = conv(concat4,   128, 3, 1)
                    iconv4  = shufflenet_v2_block(concat4, 128, 3, stride=1, dilation=1, shuffle_group=2)
                    pred4 = get_pred(iconv4)
                    upred4  = upsample_nn(pred4, 2)

                    upconv3 = upconv_sep(iconv4,   64, 3, 2) #H/4
                    concat3 = tf.concat([upconv3, skip[1], upred4], 3)
#                     iconv3  = conv(concat3,    64, 3, 1)
                    iconv3  = shufflenet_v2_block(concat3, 64, 3, stride=1, dilation=1, shuffle_group=2)
                    pred3 = get_pred(iconv3)
                    upred3  = upsample_nn(pred3, 2)

                    upconv2 = upconv_sep(iconv3,   32, 3, 2) #H/2
                    concat2 = tf.concat([upconv2, skip[0], upred3], 3)
#                     iconv2  = conv(concat2,    32, 3, 1)
                    iconv2  = shufflenet_v2_block(concat2, 32, 3, stride=1, dilation=1, shuffle_group=2)
                    pred2 = get_pred(iconv2)
                    upred2  = upsample_nn(pred2, 2)

                    upconv1 = upconv_sep(iconv2,  16, 3, 2) #H
                    concat1 = tf.concat([upconv1, upred2], 3)
#                     iconv1  = conv(concat1,   16, 3, 1)
                    iconv1  = shufflenet_v2_block(concat1, 16, 3, stride=1, dilation=1, shuffle_group=2)
                    pred1 = get_pred(iconv1)

                    return [pred1, pred2, pred3, pred4], skip[5]

                # with tf.variable_scope('prediction'):
                #     out = global_avg_pool2D(out)
                #     out = slim.conv2d(out, self.cls, 1, activation_fn=None, biases_initializer=None)
                #     out = tf.reshape(out, shape=[-1, self.cls])
                #     out = tf.identity(out, name='cls_prediction')
                #     self.output = out

def shuffle_unit(x, groups):
    with tf.variable_scope('shuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
    return x

def conv_bn_relu(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
    return x

def conv_bn(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x

def depthwise_conv_bn(x, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'depthwise_conv_bn'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=stride,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x

def resolve_shape(x):
    with tf.variable_scope(None, 'resolve_shape'):
        n, h, w, c = x.get_shape().as_list()
        if h is None or w is None:
            kernel_size = tf.convert_to_tensor([tf.shape(x)[1], tf.shape(x)[2]])
        else:
            kernel_size = [h, w]
    return kernel_size

def global_avg_pool2D(x):
    with tf.variable_scope(None, 'global_pool2D'):
        kernel_size = resolve_shape(x)
        x = slim.avg_pool2d(x, kernel_size, stride=1)
        x.set_shape([None, 1, 1, None])
    return x

def se_unit(x, bottleneck=2):
    with tf.variable_scope(None, 'SE_module'):
        n, h, w, c = x.get_shape().as_list()

        kernel_size = resolve_shape(x)
        x_pool = slim.avg_pool2d(x, kernel_size, stride=1)
        x_pool = tf.reshape(x_pool, shape=[-1, c])
        fc = slim.fully_connected(x_pool, bottleneck, activation_fn=tf.nn.relu,
                                  biases_initializer=None)
        fc = slim.fully_connected(fc, c, activation_fn=tf.nn.sigmoid,
                                  biases_initializer=None)
        if n is None:
            channel_w = tf.reshape(fc, shape=tf.convert_to_tensor([tf.shape(x)[0], 1, 1, c]))
        else:
            channel_w = tf.reshape(fc, shape=[n, 1, 1, c])

        x = tf.multiply(x, channel_w)
    return x

def shufflenet_v2_block(x, out_channel, kernel_size, stride=1, dilation=1, shuffle_group=2):
    with tf.variable_scope(None, 'shuffle_v2_block'):
        if stride == 1 and x.shape[-1] == out_channel:
            top, bottom = tf.split(x, num_or_size_splits=2, axis=3)

            half_channel = out_channel // 2

            top = conv_bn_relu(top, half_channel, 1)
            top = depthwise_conv_bn(top, kernel_size, stride, dilation)
            top = conv_bn_relu(top, half_channel, 1)

            out = tf.concat([top, bottom], axis=3)
            out = shuffle_unit(out, shuffle_group)

        else:
            half_channel = out_channel // 2
            b0 = conv_bn_relu(x, half_channel, 1)
            b0 = depthwise_conv_bn(b0, kernel_size, stride, dilation)
            b0 = conv_bn_relu(b0, half_channel, 1)

            b1 = depthwise_conv_bn(x, kernel_size, stride, dilation)
            b1 = conv_bn_relu(b1, half_channel, 1)

            out = tf.concat([b0, b1], axis=3)
            out = shuffle_unit(out, shuffle_group)
       
        return out

def get_pred(x):
    disp = 5 * conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

######################################
# other utils
######################################

def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu, normalizer_fn=slim.batch_norm):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode='REFLECT')
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)


def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def upconv(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    cnv = conv(upsample, num_out_layers, kernel_size, 1)
    return cnv

def upconv_sep(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
#     cnv = conv(upsample, num_out_layers, kernel_size, 1)
    cnv  = shufflenet_v2_block(upsample, num_out_layers, kernel_size, stride=1, dilation=1, shuffle_group=2)
    return cnv

def upsample_nn(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])


