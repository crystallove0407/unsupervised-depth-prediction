# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import time
import os

from tensorflow.contrib import slim

from nets.depth_net import D_Net
from data_loader.test_data_loader import BasicDataset
from utils.utils import preprocess_image

"""
# Example:
python3 main.py -c ../config/test_dp_kitti.ini -t kitti_eval \
    --restore_dp_model=../results/KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-140827
"""

def test_kitti_depth(data_list_file, img_dir, height, width, restore_dp_model, save_dir, depth_test_split='eigen', num_input_threads=8):
    print('[Info] Evaluate kitti depth...')
    print('[Info] Reading datalist from:', data_list_file)
    print('[Info] Loading images from:', img_dir)
    print('[Info] Reshaing image to size: ({}, {})'.format(height, width))

    dataset = BasicDataset(img_height=height,
                           img_width=width,
                           data_list_file=data_list_file,
                           img_dir=img_dir)
    iterator = dataset.create_one_shot_iterator_depth_kitti(dataset.data_list, num_parallel_calls=num_input_threads)
    img1 = iterator.get_next()
    img1 = tf.image.resize(img1, [height, width], method=0)
    img1 = preprocess_image(img1)

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()) as vs:
        pred_disp_1, _ = D_Net(img1, is_training=False, reuse=False)
        pred_depth_1 = [1./disp for disp in pred_disp_1]
        pred_depth_1 = pred_depth_1[0]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    restore_vars = tf.compat.v1.trainable_variables()
    # Add batchnorm variables.
    bn_vars = [v for v in tf.compat.v1.global_variables()
               if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name or
               'mu' in v.op.name or 'sigma' in v.op.name]
    restore_vars.extend(bn_vars)

    # restore_vars = [var for var in tf.trainable_variables()]
    print('[Info] Restoring model:', restore_dp_model)
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    # for var in tf.trainable_variables():
    #     print(var.name)

    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(restore_dp_model, restore_vars)

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer)
    sess.run(init_assign_op, init_feed_dict)
    # saver.restore(sess, restore_dp_model)

    depth_path = "%s/depth" % save_dir
    if not os.path.exists(depth_path):
        os.mkdir(depth_path)

    pred_all = []
    print("[Info] Data number:", dataset.data_num)
    start_time = time.time()
    for i in range(dataset.data_num):
        np_pred_depth_1 = sess.run(pred_depth_1)
        np_pred_depth_1 = np_pred_depth_1[0,:,:,0]
        pred_all.append(np_pred_depth_1)

    print("[Info] FPS: %.3f" % (dataset.data_num / (time.time() - start_time)))

    # save_path = save_dir + '/ckpt_' + restore_dp_model.split('/')[-2] + '_' + restore_dp_model.split('/')[-1]
    save_path = save_dir + '/test_kitti'
    print('[Info] Saving to {}.npy'.format(save_path))
    np.save(save_path, pred_all)
