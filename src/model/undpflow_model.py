# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import random
import time
import cv2
import os
import sys
import math
sys.path.insert(0, './kitti_eval/flow_tool/')
from kitti_eval.flow_tool import flowlib as fl

from six.moves import xrange
from scipy import misc, io
from tensorflow.contrib import slim

from utils.optical_flow_warp_fwd import transformerFwd
from utils.optical_flow_warp_old import transformer_old
from utils.loss_utils import SSIM, cal_grad2_error_mask, charbonnier_loss, cal_grad2_error, compute_edge_aware_smooth_loss, ternary_loss
from utils.utils import average_gradients, normalize_depth_for_display, preprocess_image, deprocess_image, inverse_warp, inverse_warp_new, get_imagenet_vars_to_restore

from data_loader.data_loader import DataLoader, AirSim_DataLoader


class Undpflow(object):
    def __init__(self, batch_size=8, iter_steps=1000000, initial_learning_rate=1e-4, decay_steps=2e5,
                 decay_rate=0.5, is_scale=True, num_input_threads=4, buffer_size=5000,
                 beta1=0.9, num_gpus=1, num_scales=4, save_checkpoint_interval=5000, write_summary_interval=200,
                 display_log_interval=50, allow_soft_placement=True, log_device_placement=False,
                 regularizer_scale=1e-4, cpu_device='/cpu:0', save_dir='KITTI', checkpoint_dir='checkpoints',
                 model_name='model', sample_dir='sample', summary_dir='summary',
                 is_restore_model=False, restore_model=None, dataset_config={}):
        self.batch_size = batch_size
        self.iter_steps = iter_steps
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_scale = is_scale
        self.num_input_threads = num_input_threads
        self.buffer_size = buffer_size
        self.beta1 = beta1
        self.num_gpus = num_gpus
        self.save_checkpoint_interval = save_checkpoint_interval
        self.write_summary_interval = write_summary_interval
        self.display_log_interval = display_log_interval
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.regularizer_scale = regularizer_scale
        self.dataset_config = dataset_config
        assert(np.mod(batch_size, num_gpus) == 0)
        self.batch_size_per_gpu = int(batch_size / np.maximum(num_gpus, 1))
        self.num_scales = num_scales  # flow has 5 scales, dp has 4 scales

        self.dataset = dataset_config['dataset']
        self.num_source = int(dataset_config['num_source'])  # do not count the target image
        self.img_height = int(dataset_config['img_height'])
        self.img_width = int(dataset_config['img_width'])

        self.save_dir = '/'.join(["..", "results", save_dir])
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.checkpoint_dir = '/'.join([self.save_dir, checkpoint_dir])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.model_name = model_name
        if not os.path.exists('/'.join([self.checkpoint_dir, model_name])):
            os.makedirs(('/'.join([self.checkpoint_dir, self.model_name])))

        self.summary_dir = '/'.join([self.save_dir, summary_dir])
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists('/'.join([self.summary_dir, 'train'])):
            os.makedirs(('/'.join([self.summary_dir, 'train'])))

    def train(self, train_mode=None, retrain=False, cont_model=None, restore_flow_model=None):
        seed = 8964
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.mode = train_mode

        """ Step 1. Loading the training data """
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            lr_decay = tf.compat.v1.train.exponential_decay(self.initial_learning_rate, global_step, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)
            optim = tf.compat.v1.train.AdamOptimizer(lr_decay, self.beta1)

            tower_grads = []

            if self.dataset == 'airsim':
                loader = AirSim_DataLoader(dataset_dir=self.dataset_config['img_dir'],
                                           img_height=self.img_height,
                                           img_width=self.img_width,
                                           batch_size=self.batch_size,
                                           num_scales=self.num_scales,
                                           num_source=self.num_source,
                                           ext=self.dataset_config['ext'],
                                           mode=train_mode)
            elif self.dataset == 'kitti':
                loader = DataLoader(dataset_dir=self.dataset_config['img_dir'],
                                    img_height=self.img_height,
                                    img_width=self.img_width,
                                    batch_size=self.batch_size,
                                    num_scales=self.num_scales,
                                    num_source=self.num_source,
                                    ext=self.dataset_config['ext'],
                                    mode=train_mode)

            if train_mode == "train_flow":
                self.tgt_image, self.src_image_stack = loader.load_train_batch()

                # print("[!] tgt_image:", self.tgt_image)
                # print("[!] src_image_stack:", self.src_image_stack)
                # Depth inputs: Feed photometric augmented image, [-1, 1]
                self.tgt_image = preprocess_image(self.tgt_image)
                self.src_image_stack = preprocess_image(self.src_image_stack)

                split_tgt_image = tf.split(
                    axis=0, num_or_size_splits=self.num_gpus, value=self.tgt_image)
                split_src_image_stack = tf.split(
                    axis=0, num_or_size_splits=self.num_gpus, value=self.src_image_stack)
                split_tgt_image_norm = [None] * self.num_gpus
                split_src_image_stack_norm = [None] * self.num_gpus
                split_cam2pix = [None] * self.num_gpus
                split_pix2cam = [None] * self.num_gpus
            elif train_mode == "train_dp":
                self.image_stack, self.image_stack_norm, self.proj_cam2pix, self.proj_pix2cam = loader.load_train_batch()

                if self.num_source == 2:
                    """ 3 frames """
                    self.tgt_image = self.image_stack[:, :, :, 3:6]
                    src0_image = self.image_stack[:, :, :, 0:3]
                    src1_image = self.image_stack[:, :, :, 6:9]
                    self.src_image_stack = tf.concat([src0_image, src1_image], axis=3)

                    self.tgt_image_norm = self.image_stack_norm[:, :, :, 3:6]
                    src0_image_norm = self.image_stack_norm[:, :, :, 0:3]
                    src1_image_norm = self.image_stack_norm[:, :, :, 6:9]
                    self.src_image_stack_norm = tf.concat([src0_image_norm, src1_image_norm], axis=3)
                elif self.num_source == 4:
                    """ 5 frames """
                    self.tgt_image = self.image_stack[:, :, :, 6:9]
                    src0_image = self.image_stack[:, :, :, 0:3]
                    src1_image = self.image_stack[:, :, :, 3:6]
                    src2_image = self.image_stack[:, :, :, 9:12]
                    src3_image = self.image_stack[:, :, :, 12:15]
                    self.src_image_stack = tf.concat([src0_image,src1_image, src2_image, src3_image], axis=3)

                    self.tgt_image_norm = self.image_stack_norm[:, :, :, 6:9]
                    src0_image_norm = self.image_stack_norm[:, :, :, 0:3]
                    src1_image_norm = self.image_stack_norm[:, :, :, 3:6]
                    src2_image_norm = self.image_stack_norm[:, :, :, 9:12]
                    src3_image_norm = self.image_stack_norm[:, :, :, 12:15]
                    self.src_image_stack_norm = tf.concat([src0_image_norm, src1_image_norm, src2_image_norm, src3_image_norm], axis=3)

                split_tgt_image = tf.split(
                    axis=0, num_or_size_splits=self.num_gpus, value=self.tgt_image)
                split_src_image_stack = tf.split(
                    axis=0, num_or_size_splits=self.num_gpus, value=self.src_image_stack)
                split_tgt_image_norm = tf.split(
                    axis=0, num_or_size_splits=self.num_gpus, value=self.tgt_image_norm)
                split_src_image_stack_norm = tf.split(
                    axis=0, num_or_size_splits=self.num_gpus, value=self.src_image_stack_norm)
                split_cam2pix = tf.split(
                    axis=0, num_or_size_splits=self.num_gpus, value=self.proj_cam2pix) # K
                split_pix2cam = tf.split(
                    axis=0, num_or_size_splits=self.num_gpus, value=self.proj_pix2cam) # K_inverse

            summaries_cpu = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES, tf.compat.v1.get_variable_scope().name)

            """ Step 2. Building model """
            if self.num_source == 2:
                print("[!] Loading the model for 3 frames...")
                from model.model_3frames import Model
            elif self.num_source == 4:
                print("[!] Loading the model for 5 frames...")
                from model.model_5frames import Model

            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()) as vs:
                print('variable_scope(vs):', vs.name)
                for i in xrange(self.num_gpus): #0 1
                    with tf.device('/gpu:%d' % i):
                        if i == self.num_gpus - 1:  #1
                            scopename = "model"
                        else:
                            scopename = '%s_%d' % ("tower", i)

                        with tf.compat.v1.name_scope(scopename) as ns:
                            if i == 0:
                                # Build models
                                model = Model(split_tgt_image[i],
                                              split_src_image_stack[i],
                                              split_tgt_image_norm[i],
                                              split_src_image_stack_norm[i],
                                              split_cam2pix[i],
                                              split_pix2cam[i],
                                              batch_size=self.batch_size_per_gpu,
                                              img_height=self.img_height,
                                              img_width=self.img_width,
                                              mode=train_mode,
                                              reuse_scope=False,
                                              scope=vs)

                                var_pose = list(
                                    set(
                                        tf.compat.v1.get_collection(
                                            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=".*pose_net.*")))
                                var_depth = list(
                                    set(
                                        tf.compat.v1.get_collection(
                                            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=".*(depth_net|feature_net_disp).*"
                                        )))
                                var_flow = list(
                                    set(
                                        tf.compat.v1.get_collection(
                                            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=".*(flow_net|feature_net_flow).*"
                                        )))

                                if self.mode == 'train_flow':
                                    var_train_list = var_flow
                                elif self.mode == 'train_dp':
                                    var_train_list = var_pose + var_depth
                                elif self.mode == 'train_all':
                                    var_train_list = var_pose + var_depth + var_flow
                            else:
                                model = Model(split_tgt_image[i],
                                              split_src_image_stack[i],
                                              split_tgt_image_norm[i],
                                              split_src_image_stack_norm[i],
                                              split_cam2pix[i],
                                              split_pix2cam[i],
                                              batch_size=self.batch_size_per_gpu,
                                              img_height=self.img_height,
                                              img_width=self.img_width,
                                              mode=train_mode,
                                              reuse_scope=True,
                                              scope=vs)

                            # Parameter Count
                            param_total = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=tf.shape(input=v)) for v in var_train_list])
                            param_depth = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=tf.shape(input=v)) for v in var_depth])
                            param_pose = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=tf.shape(input=v)) for v in var_pose])
                            param_flow = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=tf.shape(input=v)) for v in var_flow])

                            # get loss
                            loss = model.losses

                            # Retain the summaries from the final tower.
                            if i == self.num_gpus - 1:
                                summaries = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES, ns)
                                # eval_model = Model_eval(scope=vs)

                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads = optim.compute_gradients(loss, var_list=var_train_list)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)

            grads = average_gradients(tower_grads)
            # grads = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in grads]

            # Apply the gradients to adjust the shared variables.
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                apply_gradient_op = optim.apply_gradients(grads, global_step=global_step)

            # Create a saver.
            saver = tf.compat.v1.train.Saver(max_to_keep=50)

            # Build the summary operation from the last tower summaries.
            summary_op = tf.compat.v1.summary.merge(summaries + summaries_cpu)

            # Make training session.
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#             config.gpu_options.per_process_gpu_memory_fraction = 0.8
            sess = tf.compat.v1.Session(config=config)

            summary_writer = tf.compat.v1.summary.FileWriter(logdir='/'.join([self.summary_dir, 'train', self.model_name]),
                graph=sess.graph, flush_secs=10)

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            print("[Info] Model size:     {:.5f}M".format(sess.run(param_total)/1000000.0))
            print("[Info] Depth net size: {:.5f}M".format(sess.run(param_depth)/1000000.0))
            print("[Info] Pose net size:  {:.5f}M".format(sess.run(param_pose)/1000000.0))
            print("[Info] Flow net size:  {:.5f}M".format(sess.run(param_flow)/1000000.0))

            if cont_model != None:
                """ Continue training from a checkpoint """
                print("[Info] Continue training. Restoreing:", cont_model)
                saver = tf.compat.v1.train.Saver(max_to_keep=10)
                saver.restore(sess, cont_model)
            else:
                if self.mode == 'train_dp':
                    print("[Info] Restoreing pretrained flow weights from:", restore_flow_model)
                    saver_flow = tf.compat.v1.train.Saver(
                        tf.compat.v1.get_collection(
                            tf.compat.v1.GraphKeys.MODEL_VARIABLES,
                            scope=".*(flow_net|feature_net_flow).*"),
                        max_to_keep=1)
                    saver_flow.restore(sess, restore_flow_model)
                elif self.mode == 'train_all':
                    print("[Info] Restoreing:", restore_flow_model)
                    saver_rest = tf.compat.v1.train.Saver(
                        list(
                            set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)) -
                            set(
                                tf.compat.v1.get_collection(
                                    tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                    scope=".*(Adam_1|Adam).*"))),
                        max_to_keep=1)
                    saver_rest.restore(sess, restore_flow_model)

            if retrain:
                sess.run(global_step.assign(0))

            start_itr = global_step.eval(session=sess)
            tf.compat.v1.train.start_queue_runners(sess)

            """ Step 3. Training """
            steps_per_epoch = loader.steps_per_epoch
            last_summary_time = time.time()
            start_time = time.time()
            for itr in range(start_itr, self.iter_steps):
                fetches = {
                    'train_op': apply_gradient_op,
                    'grads': grads,
                    'global_step': global_step,
                    'lr_decay': lr_decay
                }

                if np.mod(itr, self.write_summary_interval) == 0:
                    fetches['summary_str'] = summary_op
                    fetches['summary_scalar_str'] = model.summ_op

                if np.mod(itr, self.display_log_interval) == 0:
                    fetches['loss'] = loss

                results = sess.run(fetches)
                gs = results['global_step']

                # print(results['valid_src0'])

                if np.mod(itr, self.write_summary_interval) == 0:
                    summary_writer.add_summary(results['summary_scalar_str'], itr)
                    summary_writer.add_summary(results['summary_str'], itr)

                if np.mod(itr, self.display_log_interval) == 0:
                    train_epoch = math.ceil(gs / steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * steps_per_epoch
                    this_cycle = time.time() - last_summary_time
                    last_summary_time += this_cycle
                    print('Epoch: [%2d] [%5d/%5d] total steps:[%6d] lr:[%2.8f] time: %4.2fs (%ds total) loss: %.3f' % \
                        (train_epoch, train_step, steps_per_epoch, gs, results['lr_decay'], this_cycle, time.time() - start_time, results['loss']))

                if np.mod(itr, steps_per_epoch) == 0:
                    print('[Info] Saving checkpoint to %s ...' % self.checkpoint_dir)
                    saver.save(sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=gs)
