from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def P_Net3(image_stack, disp_bottleneck_stack, joint_encoder, weight_reg=0.0004):
    with tf.variable_scope('pose_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_regularizer=slim.l2_regularizer(
                                weight_reg),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            if not joint_encoder:
                print("[Info] No Joint encoder")
                print("[Info] Run Pose Net encoder...")
                cnv1 = slim.conv2d(image_stack, 16, [
                                   7, 7], stride=2, scope='cnv1')
                cnv1b = slim.conv2d(cnv1, 16, [7, 7], stride=1, scope='cnv1b')
                cnv2 = slim.conv2d(cnv1b, 32, [5, 5], stride=2, scope='cnv2')
                cnv2b = slim.conv2d(cnv2, 32, [5, 5], stride=1, scope='cnv2b')
                cnv3 = slim.conv2d(cnv2b, 64, [3, 3], stride=2, scope='cnv3')
                cnv3b = slim.conv2d(cnv3, 64, [3, 3], stride=1, scope='cnv3b')
                cnv4 = slim.conv2d(cnv3b, 128, [3, 3], stride=2, scope='cnv4')
                cnv4b = slim.conv2d(cnv4, 128, [3, 3], stride=1, scope='cnv4b')
                cnv5 = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
                cnv5b = slim.conv2d(cnv5, 256, [3, 3], stride=1, scope='cnv5b')
                print("[Info] Flow net decoder input: ", cnv5b.shape)
            else:
                print("[Info] Joint encoder")
                print("[Info] Flow net decoder input: ",
                      disp_bottleneck_stack.shape)

            inputs = cnv5b if not joint_encoder else disp_bottleneck_stack

            # Pose specific layers
#             inputs = slim.conv2d(inputs, 512, 1, activation_fn=None, scope='pwise')
            cnv6 = slim.conv2d(inputs, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
            cnv7 = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 256, [3, 3], stride=1, scope='cnv7b')

            pose_pred = slim.conv2d(cnv7b,
                                    6*6, [1, 1],
                                    scope='pred',
                                    stride=1,
                                    normalizer_fn=None,
                                    activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = tf.reshape(pose_avg, [-1, 1, 6*6])

            tran_mag = 0.001 if joint_encoder else 1.0
            rot_mag = 0.01

            pose_final = tf.concat(
                [  # 0: src0 -> tgt
                    tran_mag * pose_final[:, :, 0:3],
                    rot_mag * pose_final[:, :, 3:6],
                    # 1: tgt -> src1
                    tran_mag * pose_final[:, :, 6:9],
                    rot_mag * pose_final[:, :, 9:12],
                    # 2: src0 -> src1
                    tran_mag * pose_final[:, :, 12:15],
                    rot_mag * pose_final[:, :, 15:18],
                    # 3: tgt -> src0
                    tran_mag * pose_final[:, :, 18:21],
                    rot_mag * pose_final[:, :, 21:24],
                    # 4: src1 -> tgt
                    tran_mag * pose_final[:, :, 24:27],
                    rot_mag * pose_final[:, :, 27:30],
                    # 5: src1 -> src0
                    tran_mag * pose_final[:, :, 30:33],
                    rot_mag * pose_final[:, :, 33:36]
                ], axis=2)

            return pose_final
