from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def P_Net3(image_stack, disp_bottleneck_stack, joint_encoder, weight_reg=0.0004):
    with tf.variable_scope('pose_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                          normalizer_fn=None,
                          weights_regularizer=slim.l2_regularizer(weight_reg),
                          normalizer_params=None,
                          activation_fn=tf.nn.relu,
                          outputs_collections=end_points_collection):
            if not joint_encoder:
                cnv1 = slim.conv2d(image_stack, 16, [7, 7], stride=2, scope='cnv1')
                cnv1b = slim.conv2d(cnv1, 16, [7, 7], stride=1, scope='cnv1b')
                cnv2 = slim.conv2d(cnv1b, 32, [5, 5], stride=2, scope='cnv2')
                cnv2b = slim.conv2d(cnv2, 32, [5, 5], stride=1, scope='cnv2b')
                cnv3 = slim.conv2d(cnv2b, 64, [3, 3], stride=2, scope='cnv3')
                cnv3b = slim.conv2d(cnv3, 64, [3, 3], stride=1, scope='cnv3b')
                cnv4 = slim.conv2d(cnv3b, 128, [3, 3], stride=2, scope='cnv4')
                cnv4b = slim.conv2d(cnv4, 128, [3, 3], stride=1, scope='cnv4b')
                cnv5 = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
                cnv5b = slim.conv2d(cnv5, 256, [3, 3], stride=1, scope='cnv5b')

            inputs = disp_bottleneck_stack if joint_encoder else cnv5b

            # Pose specific layers
            cnv6 = slim.conv2d(inputs, 256, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 256, [3, 3], stride=1, scope='cnv6b')
            cnv7 = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 256, [3, 3], stride=1, scope='cnv7b')

            pose_pred = slim.conv2d(
                cnv7b,
                6*6, [1, 1],
                scope='pred',
                stride=1,
                normalizer_fn=None,
                activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = tf.reshape(pose_avg, [-1, 1, 6*6])

            tran_mag = 0.001 if joint_encoder else 1.0
            rot_mag= 0.01

            pose_final = tf.concat(
                [tran_mag * pose_final[:, :, 0:3],   rot_mag * pose_final[:, :, 3:6],    # 0: src0 -> tgt
                 tran_mag * pose_final[:, :, 6:9],   rot_mag * pose_final[:, :, 9:12],   # 1: tgt -> src1
                 tran_mag * pose_final[:, :, 12:15], rot_mag * pose_final[:, :, 15:18],  # 2: src0 -> src1
                 tran_mag * pose_final[:, :, 18:21], rot_mag * pose_final[:, :, 21:24],  # 3: tgt -> src0
                 tran_mag * pose_final[:, :, 24:27], rot_mag * pose_final[:, :, 27:30],  # 4: src1 -> tgt
                 tran_mag * pose_final[:, :, 30:33], rot_mag * pose_final[:, :, 33:36]], # 5: src1 -> src0
                axis=2)

            return pose_final

def P_Net5(image_stack, disp_bottleneck_stack, joint_encoder, weight_reg=0.0004):
    with tf.variable_scope('pose_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                          normalizer_fn=None,
                          weights_regularizer=slim.l2_regularizer(weight_reg),
                          normalizer_params=None,
                          activation_fn=tf.nn.relu,
                          outputs_collections=end_points_collection):
            if not joint_encoder:
                cnv1 = slim.conv2d(image_stack, 16, [7, 7], stride=2, scope='cnv1')
                cnv1b = slim.conv2d(cnv1, 16, [7, 7], stride=1, scope='cnv1b')
                cnv2 = slim.conv2d(cnv1b, 32, [5, 5], stride=2, scope='cnv2')
                cnv2b = slim.conv2d(cnv2, 32, [5, 5], stride=1, scope='cnv2b')
                cnv3 = slim.conv2d(cnv2b, 64, [3, 3], stride=2, scope='cnv3')
                cnv3b = slim.conv2d(cnv3, 64, [3, 3], stride=1, scope='cnv3b')
                cnv4 = slim.conv2d(cnv3b, 128, [3, 3], stride=2, scope='cnv4')
                cnv4b = slim.conv2d(cnv4, 128, [3, 3], stride=1, scope='cnv4b')
                cnv5 = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
                cnv5b = slim.conv2d(cnv5, 256, [3, 3], stride=1, scope='cnv5b')

            inputs = disp_bottleneck_stack if joint_encoder else cnv5b

            # Pose specific layers
            cnv6 = slim.conv2d(inputs, 256, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 256, [3, 3], stride=1, scope='cnv6b')
            cnv7 = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 256, [3, 3], stride=1, scope='cnv7b')

            pose_pred = slim.conv2d(
                cnv7b,
                6*20, [1, 1],
                scope='pred',
                stride=1,
                normalizer_fn=None,
                activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = tf.reshape(pose_avg, [-1, 1, 6*20])

            tran_mag = 0.001 if joint_encoder else 1.0
            rot_mag= 0.01

            pose_final = tf.concat(
                [tran_mag * pose_final[:, :, 0:3],   rot_mag * pose_final[:, :, 3:6],    # 0: src0 -> tgt
                 tran_mag * pose_final[:, :, 6:9],   rot_mag * pose_final[:, :, 9:12],   # 1: tgt -> src1
                 tran_mag * pose_final[:, :, 12:15], rot_mag * pose_final[:, :, 15:18],  # 2: src0 -> src1
                 tran_mag * pose_final[:, :, 18:21], rot_mag * pose_final[:, :, 21:24],  # 3: tgt -> src0
                 tran_mag * pose_final[:, :, 24:27], rot_mag * pose_final[:, :, 27:30],  # 4: src1 -> tgt
                 tran_mag * pose_final[:, :, 30:33], rot_mag * pose_final[:, :, 33:36],
                 tran_mag * pose_final[:, :, 36:42], rot_mag * pose_final[:, :, 39:42],
                 tran_mag * pose_final[:, :, 42:45], rot_mag * pose_final[:, :, 45:48],
                 tran_mag * pose_final[:, :, 48:51], rot_mag * pose_final[:, :, 51:54],
                 tran_mag * pose_final[:, :, 54:57], rot_mag * pose_final[:, :, 57:60],
                 tran_mag * pose_final[:, :, 60:63], rot_mag * pose_final[:, :, 63:66],
                 tran_mag * pose_final[:, :, 66:69], rot_mag * pose_final[:, :, 69:72],
                 tran_mag * pose_final[:, :, 72:75], rot_mag * pose_final[:, :, 75:78],
                 tran_mag * pose_final[:, :, 78:81], rot_mag * pose_final[:, :, 81:84],
                 tran_mag * pose_final[:, :, 84:87], rot_mag * pose_final[:, :, 87:90],
                 tran_mag * pose_final[:, :, 90:93], rot_mag * pose_final[:, :, 93:96],
                 tran_mag * pose_final[:, :, 96:99], rot_mag * pose_final[:, :, 99:102],
                 tran_mag * pose_final[:, :, 102:105], rot_mag * pose_final[:, :, 105:108],
                 tran_mag * pose_final[:, :, 108:111], rot_mag * pose_final[:, :, 111:114],
                 tran_mag * pose_final[:, :, 114:117], rot_mag * pose_final[:, :, 117:120]],
                axis=2)

            return pose_final
