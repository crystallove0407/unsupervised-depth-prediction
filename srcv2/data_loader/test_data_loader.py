# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class BasicDataset(object):
    def __init__(self, img_height=375, img_width=1242,
                 data_list_file='path_to_your_data_list_file',
                 img_dir='path_to_your_image_directory'):
        self.img_height = img_height
        self.img_width = img_width
        self.img_dir = img_dir
        self.data_list = np.loadtxt(data_list_file, dtype=np.str)
        self.data_num = self.data_list.shape[0]

    def read_and_decode_depth_kitti(self, filename_queue):
        img1_name = tf.strings.join([self.img_dir, '/', filename_queue])
        img1 = tf.image.decode_png(tf.io.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        return img1

    def preprocess_one_shot_depth_kitti(self, filename_queue):
        img1 = self.read_and_decode_depth_kitti(filename_queue)
        img1 = img1 / 255.

        return img1

    def create_one_shot_iterator_depth_kitti(self, data_list, num_parallel_calls=4):
        data_list = tf.convert_to_tensor(value=data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_one_shot_depth_kitti, num_parallel_calls=num_parallel_calls)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))

        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        return iterator
