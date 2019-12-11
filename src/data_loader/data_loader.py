# Mostly based on the code written by Tinghui Zhou & Clement Godard:
# https://github.com/tinghuiz/SfMLearner/blob/master/data_loader.py
# https://github.com/mrharicot/monodepth/blob/master/monodepth_dataloader.py
from __future__ import division
import os
import random
import tensorflow as tf
from absl import logging


QUEUE_SIZE = 2000
QUEUE_BUFFER = 3
# See nets.encoder_resnet as reference for below input-normalizing constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_SD = (0.229, 0.224, 0.225)
FLIP_RANDOM = 'random'  # Always perform random flipping.
FLIP_ALWAYS = 'always'  # Always flip image input, used for test augmentation.
FLIP_NONE = 'none'  # Always disables flipping.


class DataLoader(object):
    def __init__(self, dataset_dir, img_height, img_width, num_source=2, num_threads=32,
                 batch_size=4, num_scales=4, ext='jpg', random_color=True, imagenet_norm=False,
                 shuffle=True, random_scale_crop=True, mode=None):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.seq_length = num_source + 1
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.num_scales = num_scales
        self.file_extension = ext
        self.flipping_mode = FLIP_RANDOM
        self.random_color = random_color
        self.random_scale_crop = random_scale_crop
        self.shuffle = shuffle
        self.mode = mode
        self.imagenet_norm = imagenet_norm

    def load_train_batch(self):
        """Load a batch of training instances. """
        # Form training batches
        seed = random.randint(0, 2**31 - 1)

        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train', self.file_extension)
        image_paths_queue = tf.compat.v1.train.string_input_producer(
            file_list['image_file_list'], seed=seed,
            shuffle=self.shuffle,
            num_epochs=(1 if not self.shuffle else None))
        cam_paths_queue = tf.compat.v1.train.string_input_producer(
            file_list['cam_file_list'], seed=seed,
            shuffle=self.shuffle,
            num_epochs=(1 if not self.shuffle else None))

        # Load images
        img_reader = tf.compat.v1.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        if self.file_extension == 'jpg':
            image_seq = tf.image.decode_jpeg(image_contents)
        elif self.file_extension == 'png':
            image_seq = tf.image.decode_png(image_contents, channels=3)

        if self.mode == "train_flow":
            tgt_image_o, src_image_stack_o = self.unpack_image_sequence(image_seq, self.img_height, self.img_width, self.num_source)

            if self.shuffle:
                tgt_image_o, src_image_stack_o = tf.compat.v1.train.shuffle_batch(
                    [tgt_image_o, src_image_stack_o],
                    batch_size=self.batch_size,
                    capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
                    min_after_dequeue=QUEUE_SIZE)
            else:
                tgt_image_o, src_image_stack_o = tf.compat.v1.train.batch(
                    [tgt_image_o, src_image_stack_o],
                    batch_size=self.batch_size,
                    num_threads=1,
                    capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size)

            return tgt_image_o, src_image_stack_o
        else:
            """ Data Loading and Augmentation for depth & pose """
            with tf.compat.v1.name_scope('load_intrinsics'):
                cam_reader = tf.compat.v1.TextLineReader()
                _, raw_cam_contents = cam_reader.read(cam_paths_queue)
                rec_def = []
                for i in range(9):
                    rec_def.append([1.0])
                raw_cam_vec = tf.io.decode_csv(records=raw_cam_contents, record_defaults=rec_def)
                raw_cam_vec = tf.stack(raw_cam_vec)
                intrinsics = tf.reshape(raw_cam_vec, [3, 3])

            with tf.compat.v1.name_scope('convert_image'):
                image_seq = self.preprocess_image(image_seq)  # Converts to float.

            if self.random_color:
                with tf.compat.v1.name_scope('image_augmentation'):
                    image_seq = self.augment_image_colorspace(image_seq)

            image_stack = self.unpack_images(image_seq)

            if self.flipping_mode != FLIP_NONE:
                random_flipping = (self.flipping_mode == FLIP_RANDOM)
                with tf.compat.v1.name_scope('image_augmentation_flip'):
                    image_stack, intrinsics = self.augment_images_flip(
                        image_stack, intrinsics, randomized=random_flipping)

            if self.random_scale_crop:
                with tf.compat.v1.name_scope('image_augmentation_scale_crop'):
                    image_stack, intrinsics = self.augment_images_scale_crop(
                        image_stack, intrinsics, self.img_height, self.img_width)

            with tf.compat.v1.name_scope('multi_scale_intrinsics'):
                intrinsic_mat = self.get_multi_scale_intrinsics(intrinsics, self.num_scales)
                intrinsic_mat.set_shape([self.num_scales, 3, 3])
                intrinsic_mat_inv = tf.linalg.inv(intrinsic_mat)
                intrinsic_mat_inv.set_shape([self.num_scales, 3, 3])

            if self.imagenet_norm:
                im_mean = tf.tile(
                    tf.constant(IMAGENET_MEAN), multiples=[self.seq_length])
                im_sd = tf.tile(
                    tf.constant(IMAGENET_SD), multiples=[self.seq_length])
                image_stack_norm = (image_stack - im_mean) / im_sd
            else:
                image_stack_norm = image_stack

            with tf.compat.v1.name_scope('batching'):
                min_after_dequeue = 2048
                capacity = min_after_dequeue + 4 * self.batch_size
                if self.shuffle:
                    logging.info("[Info] Shuffling the batch")
                    (image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv) = tf.compat.v1.train.shuffle_batch(
                        [image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv],
                        batch_size=self.batch_size,
                        capacity=capacity,
                        min_after_dequeue=min_after_dequeue,
                        num_threads=10)
                else:
                    (image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv) = tf.compat.v1.train.batch(
                        [image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv],
                        batch_size=self.batch_size,
                        num_threads=1,
                        capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size)

            return image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv

    def make_batch_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random.uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize(im, [out_h, out_w], method=tf.image.ResizeMethod.AREA)
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_batch_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(input=im))
            offset_y = tf.random.uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random.uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_batch_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random coloring
        def random_coloring(im):
            batch_size, in_h, in_w, in_c = im.get_shape().as_list()
            im_f = tf.image.convert_image_dtype(im, tf.float32)

            # randomly shift gamma
            random_gamma = tf.random.uniform([], 0.8, 1.2)
            im_aug  = im_f  ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random.uniform([], 0.5, 2.0)
            im_aug  =  im_aug * random_brightness

            # randomly shift color
            random_colors = tf.random.uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
            im_aug  *= color_image

            # saturate
            im_aug  = tf.clip_by_value(im_aug,  0, 1)

            im_aug = tf.image.convert_image_dtype(im_aug, tf.uint8)

            return im_aug
        im, intrinsics = random_scaling(im, intrinsics)
        im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        # do_augment  = tf.random_uniform([], 0, 1)
        # im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)

        return im, intrinsics

    def format_file_list(self, data_root, split, ext):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.' + ext) for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        self.steps_per_epoch = len(image_file_list) // self.batch_size
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def get_batch_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(self.make_batch_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

    def get_batch_multi_scale_intrinsics2(self, raw_cam_mat, num_scales):
        proj_cam2pix = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = raw_cam_mat[:, 0, 0] / (2**s)
            fy = raw_cam_mat[:, 1, 1] / (2**s)
            cx = raw_cam_mat[:, 0, 2] / (2**s)
            cy = raw_cam_mat[:, 1, 2] / (2**s)
            # r1 = tf.stack([fx, 0, cx])
            # r2 = tf.stack([0, fy, cy])
            # r3 = tf.constant([0., 0., 1.])
            proj_cam2pix.append(self.make_batch_intrinsics_matrix(fx, fy, cx, cy))
        proj_cam2pix = tf.stack(proj_cam2pix, axis=1)
        proj_pix2cam = tf.linalg.inv(proj_cam2pix)
        proj_cam2pix.set_shape([self.batch_size, num_scales, 3, 3])
        proj_pix2cam.set_shape([self.batch_size, num_scales, 3, 3])
        return proj_cam2pix, proj_pix2cam

    def unpack_images(self, image_seq):
        """[h, w * seq_length, 3] -> [h, w, 3 * seq_length]."""
        with tf.compat.v1.name_scope('unpack_images'):
            image_list = [
                image_seq[:, i * self.img_width:(i + 1) * self.img_width, :]
                for i in range(self.seq_length)
            ]
            image_stack = tf.concat(image_list, axis=2)
            image_stack.set_shape([self.img_height, self.img_width, self.seq_length * 3])
        return image_stack

    @classmethod
    def preprocess_image(cls, image):
        # Convert from uint8 to float.
        return tf.image.convert_image_dtype(image, dtype=tf.float32)

    @classmethod
    def augment_image_colorspace(cls, image_stack):
        """Apply data augmentation to inputs."""
        image_stack_aug = image_stack
        # Randomly shift brightness.
        apply_brightness = tf.less(tf.random.uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            pred=apply_brightness,
            true_fn=lambda: tf.image.random_brightness(image_stack_aug, max_delta=0.1),
            false_fn=lambda: image_stack_aug)

        # Randomly shift contrast.
        apply_contrast = tf.less(tf.random.uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            pred=apply_contrast,
            true_fn=lambda: tf.image.random_contrast(image_stack_aug, 0.85, 1.15),
            false_fn=lambda: image_stack_aug)

        # Randomly change saturation.
        apply_saturation = tf.less(tf.random.uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            pred=apply_saturation,
            true_fn=lambda: tf.image.random_saturation(image_stack_aug, 0.85, 1.15),
            false_fn=lambda: image_stack_aug)

        # Randomly change hue.
        apply_hue = tf.less(tf.random.uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            pred=apply_hue,
            true_fn=lambda: tf.image.random_hue(image_stack_aug, max_delta=0.1),
            false_fn=lambda: image_stack_aug)

        image_stack_aug = tf.clip_by_value(image_stack_aug, 0, 1)
        return image_stack_aug

    @classmethod
    def augment_images_flip(cls, image_stack, intrinsics, randomized=True):
        """Randomly flips the image horizontally."""

        def flip(cls, image_stack, intrinsics):
            _, in_w, _ = image_stack.get_shape().as_list()
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = in_w - intrinsics[0, 2]
            cy = intrinsics[1, 2]
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return (tf.image.flip_left_right(image_stack), intrinsics)

        if randomized:
            prob = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
            predicate = tf.less(prob, 0.5)
            return tf.cond(pred=predicate,
                           true_fn=lambda: flip(cls, image_stack, intrinsics),
                           false_fn=lambda: (image_stack, intrinsics))
        else:
           return flip(cls, image_stack, intrinsics)

    @classmethod
    def augment_images_scale_crop(cls, im, intrinsics, out_h, out_w):
        """Randomly scales and crops image."""

        def scale_randomly(im, intrinsics):
            """Scales image and adjust intrinsics accordingly."""
            in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random.uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            # Add batch.
            im = tf.expand_dims(im, 0)
            im = tf.image.resize(im, [out_h, out_w], method=tf.image.ResizeMethod.AREA)
            im = im[0]
            fx = intrinsics[0, 0] * x_scaling
            fy = intrinsics[1, 1] * y_scaling
            cx = intrinsics[0, 2] * x_scaling
            cy = intrinsics[1, 2] * y_scaling
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def crop_randomly(im, intrinsics, out_h, out_w):
            """Crops image and adjust intrinsics accordingly."""
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            in_h, in_w, _ = tf.unstack(tf.shape(input=im))
            offset_y = tf.random.uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random.uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[1, 2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        im, intrinsics = scale_randomly(im, intrinsics)
        im, intrinsics = crop_randomly(im, intrinsics, out_h, out_w)
        return im, intrinsics

    @classmethod
    def make_intrinsics_matrix(cls, fx, fy, cx, cy):
        r1 = tf.stack([fx, 0, cx])
        r2 = tf.stack([0, fy, cy])
        r3 = tf.constant([0., 0., 1.])
        intrinsics = tf.stack([r1, r2, r3])
        return intrinsics

    @classmethod
    def get_multi_scale_intrinsics(cls, intrinsics, num_scales):
        """Returns multiple intrinsic matrices for different scales."""
        intrinsics_multi_scale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[0, 0] / (2**s)
            fy = intrinsics[1, 1] / (2**s)
            cx = intrinsics[0, 2] / (2**s)
            cy = intrinsics[1, 2] / (2**s)
            intrinsics_multi_scale.append(cls.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_multi_scale = tf.stack(intrinsics_multi_scale)
        return intrinsics_multi_scale

class AirSim_DataLoader(object):
    def __init__(self, dataset_dir, img_height=192, img_width=640, num_source=2, num_threads=32,
                 batch_size=4, num_scales=4, ext='png', random_color=True, imagenet_norm=False,
                 shuffle=True, random_scale_crop=True, mode=None):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.seq_length = num_source + 1
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.num_scales = num_scales
        self.file_extension = ext
        self.flipping_mode = FLIP_RANDOM
        self.random_color = random_color
        self.random_scale_crop = random_scale_crop
        self.shuffle = shuffle
        self.mode = mode
        self.imagenet_norm = imagenet_norm

    def load_train_batch(self):
        """Load a batch of training instances. """
        # Form training batches
        seed = random.randint(0, 2**31 - 1)

        # Load the list of training files into queues
        # file_list = self.format_file_list(self.dataset_dir, 'train', self.file_extension)

        with open(self.dataset_dir + '/train.txt', 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(self.dataset_dir, subfolders[i],
            frame_ids[i] + '.' + self.file_extension) for i in range(len(frames))]
        file_list = {}
        file_list['image_file_list'] = image_file_list
        self.steps_per_epoch = len(image_file_list) // self.batch_size

        image_paths_queue = tf.compat.v1.train.string_input_producer(
            file_list['image_file_list'], seed=seed,
            shuffle=self.shuffle,
            num_epochs=(1 if not self.shuffle else None))

        # Load images
        img_reader = tf.compat.v1.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        if self.file_extension == 'jpg':
            image_seq = tf.image.decode_jpeg(image_contents)
        elif self.file_extension == 'png':
            image_seq = tf.image.decode_png(image_contents, channels=3)

        if self.mode == "train_flow":
            tgt_image_o, src_image_stack_o = self.unpack_image_sequence(image_seq, self.img_height, self.img_width, self.num_source)

            if self.shuffle:
                tgt_image_o, src_image_stack_o = tf.compat.v1.train.shuffle_batch(
                    [tgt_image_o, src_image_stack_o],
                    batch_size=self.batch_size,
                    capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
                    min_after_dequeue=QUEUE_SIZE)
            else:
                tgt_image_o, src_image_stack_o = tf.compat.v1.train.batch(
                    [tgt_image_o, src_image_stack_o],
                    batch_size=self.batch_size,
                    num_threads=1,
                    capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size)

            # Data augmentation
            src0_image_o = src_image_stack_o[:, :, :, 0:3]
            src1_image_o = src_image_stack_o[:, :, :, 3:6]

            # randomly flip images
            # do_flip = tf.random_uniform([], 0, 1)
            # src0_image_o = tf.cond(do_flip > 0.5,
            #                        lambda: tf.image.flip_left_right(src0_image_o),
            #                        lambda: src0_image_o)
            # tgt_image_o = tf.cond(do_flip > 0.5,
            #                       lambda: tf.image.flip_left_right(tgt_image_o),
            #                       lambda: tgt_image_o)
            # src1_image_o = tf.cond(do_flip > 0.5,
            #                        lambda: tf.image.flip_left_right(src1_image_o),
            #                        lambda: src1_image_o)
            #
            # do_flip_fb = tf.random_uniform([], 0, 1)
            # src0_image, tgt_image, src1_image = tf.cond(
            #     do_flip_fb > 0.5,
            #     lambda: (src1_image_o, tgt_image_o, src0_image_o),
            #     lambda: (src0_image_o, tgt_image_o, src1_image_o)
            # )

            # image_all = tf.concat([src0_image, tgt_image, src1_image], axis=3)
            # # image_all, intrinsics = self.data_augmentation(
            # #     image_all, intrinsics, self.img_height, self.img_width)
            # tgt_image = image_all[:, :, :, :3]
            # src_image_stack = image_all[:, :, :, 3:]

            src_image_stack = tf.concat([src0_image_o, src1_image_o], axis=3)
            return tgt_image_o, src_image_stack
        else:
            """ Data Loading and Augmentation for depth & pose """
            with tf.compat.v1.name_scope('load_intrinsics'):
                # cam_reader = tf.TextLineReader()
                # _, raw_cam_contents = cam_reader.read(cam_paths_queue)
                # rec_def = []
                # for i in range(9):
                #     rec_def.append([1.0])
                # raw_cam_vec = tf.decode_csv(raw_cam_contents, record_defaults=rec_def)
                # raw_cam_vec = tf.stack(raw_cam_vec)
                # intrinsics = tf.reshape(raw_cam_vec, [3, 3])
                raw_cam_vec = tf.constant([96, 0, 96, 0, 320, 320, 0, 0, 1], dtype=tf.float32)
                intrinsics = tf.reshape(raw_cam_vec, [3, 3])

            with tf.compat.v1.name_scope('convert_image'):
                image_seq = self.preprocess_image(image_seq)  # Converts to float.

            # if self.random_color:
            #     with tf.name_scope('image_augmentation'):
            #         image_seq = self.augment_image_colorspace(image_seq)

            image_stack = self.unpack_images(image_seq)

            # if self.flipping_mode != FLIP_NONE:
            #     random_flipping = (self.flipping_mode == FLIP_RANDOM)
            #     with tf.name_scope('image_augmentation_flip'):
            #         image_stack, intrinsics = self.augment_images_flip(
            #             image_stack, intrinsics, randomized=random_flipping)

            # if self.random_scale_crop:
            #     with tf.name_scope('image_augmentation_scale_crop'):
            #         image_stack, intrinsics = self.augment_images_scale_crop(
            #             image_stack, intrinsics, self.img_height, self.img_width)

            with tf.compat.v1.name_scope('multi_scale_intrinsics'):
                intrinsic_mat = self.get_multi_scale_intrinsics(intrinsics, self.num_scales)
                intrinsic_mat.set_shape([self.num_scales, 3, 3])
                intrinsic_mat_inv = tf.linalg.inv(intrinsic_mat)
                intrinsic_mat_inv.set_shape([self.num_scales, 3, 3])

            if self.imagenet_norm:
                im_mean = tf.tile(
                    tf.constant(IMAGENET_MEAN), multiples=[self.seq_length])
                im_sd = tf.tile(
                    tf.constant(IMAGENET_SD), multiples=[self.seq_length])
                image_stack_norm = (image_stack - im_mean) / im_sd
            else:
                image_stack_norm = image_stack

            with tf.compat.v1.name_scope('batching'):
                min_after_dequeue = 2048
                capacity = min_after_dequeue + 4 * self.batch_size
                if self.shuffle:
                    logging.info("[Info] Shuffling the batch")
                    (image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv) = tf.compat.v1.train.shuffle_batch(
                        [image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv],
                        batch_size=self.batch_size,
                        capacity=capacity,
                        min_after_dequeue=min_after_dequeue,
                        num_threads=10)
                else:
                    (image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv) = tf.compat.v1.train.batch(
                        [image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv],
                        batch_size=self.batch_size,
                        num_threads=1,
                        capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size)

            return image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv

    def make_batch_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random.uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize(im, [out_h, out_w], method=tf.image.ResizeMethod.AREA)
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_batch_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(input=im))
            offset_y = tf.random.uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random.uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_batch_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random coloring
        def random_coloring(im):
            batch_size, in_h, in_w, in_c = im.get_shape().as_list()
            im_f = tf.image.convert_image_dtype(im, tf.float32)

            # randomly shift gamma
            random_gamma = tf.random.uniform([], 0.8, 1.2)
            im_aug  = im_f  ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random.uniform([], 0.5, 2.0)
            im_aug  =  im_aug * random_brightness

            # randomly shift color
            random_colors = tf.random.uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
            im_aug  *= color_image

            # saturate
            im_aug  = tf.clip_by_value(im_aug,  0, 1)

            im_aug = tf.image.convert_image_dtype(im_aug, tf.uint8)

            return im_aug
        im, intrinsics = random_scaling(im, intrinsics)
        im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        # do_augment  = tf.random_uniform([], 0, 1)
        # im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)

        return im, intrinsics

    def format_file_list(self, data_root, split, ext):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.' + ext) for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        self.steps_per_epoch = len(image_file_list) // self.batch_size
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def get_batch_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(self.make_batch_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

    def get_batch_multi_scale_intrinsics2(self, raw_cam_mat, num_scales):
        proj_cam2pix = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = raw_cam_mat[:, 0, 0] / (2**s)
            fy = raw_cam_mat[:, 1, 1] / (2**s)
            cx = raw_cam_mat[:, 0, 2] / (2**s)
            cy = raw_cam_mat[:, 1, 2] / (2**s)
            # r1 = tf.stack([fx, 0, cx])
            # r2 = tf.stack([0, fy, cy])
            # r3 = tf.constant([0., 0., 1.])
            proj_cam2pix.append(self.make_batch_intrinsics_matrix(fx, fy, cx, cy))
        proj_cam2pix = tf.stack(proj_cam2pix, axis=1)
        proj_pix2cam = tf.linalg.inv(proj_cam2pix)
        proj_cam2pix.set_shape([self.batch_size, num_scales, 3, 3])
        proj_pix2cam.set_shape([self.batch_size, num_scales, 3, 3])
        return proj_cam2pix, proj_pix2cam

    def unpack_images(self, image_seq):
        """[h, w * seq_length, 3] -> [h, w, 3 * seq_length]."""
        with tf.compat.v1.name_scope('unpack_images'):
            image_list = [
                image_seq[:, i * self.img_width:(i + 1) * self.img_width, :]
                for i in range(self.seq_length)
            ]
            image_stack = tf.concat(image_list, axis=2)
            image_stack.set_shape([self.img_height, self.img_width, self.seq_length * 3])
        return image_stack

    @classmethod
    def preprocess_image(cls, image):
        # Convert from uint8 to float.
        return tf.image.convert_image_dtype(image, dtype=tf.float32)

    @classmethod
    def augment_image_colorspace(cls, image_stack):
        """Apply data augmentation to inputs."""
        image_stack_aug = image_stack
        # Randomly shift brightness.
        apply_brightness = tf.less(tf.random.uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            pred=apply_brightness,
            true_fn=lambda: tf.image.random_brightness(image_stack_aug, max_delta=0.1),
            false_fn=lambda: image_stack_aug)

        # Randomly shift contrast.
        apply_contrast = tf.less(tf.random.uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            pred=apply_contrast,
            true_fn=lambda: tf.image.random_contrast(image_stack_aug, 0.85, 1.15),
            false_fn=lambda: image_stack_aug)

        # Randomly change saturation.
        apply_saturation = tf.less(tf.random.uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            pred=apply_saturation,
            true_fn=lambda: tf.image.random_saturation(image_stack_aug, 0.85, 1.15),
            false_fn=lambda: image_stack_aug)

        # Randomly change hue.
        apply_hue = tf.less(tf.random.uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            pred=apply_hue,
            true_fn=lambda: tf.image.random_hue(image_stack_aug, max_delta=0.1),
            false_fn=lambda: image_stack_aug)

        image_stack_aug = tf.clip_by_value(image_stack_aug, 0, 1)
        return image_stack_aug

    @classmethod
    def augment_images_flip(cls, image_stack, intrinsics, randomized=True):
        """Randomly flips the image horizontally."""

        def flip(cls, image_stack, intrinsics):
            _, in_w, _ = image_stack.get_shape().as_list()
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = in_w - intrinsics[0, 2]
            cy = intrinsics[1, 2]
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return (tf.image.flip_left_right(image_stack), intrinsics)

        if randomized:
            prob = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
            predicate = tf.less(prob, 0.5)
            return tf.cond(pred=predicate,
                           true_fn=lambda: flip(cls, image_stack, intrinsics),
                           false_fn=lambda: (image_stack, intrinsics))
        else:
           return flip(cls, image_stack, intrinsics)

    @classmethod
    def augment_images_scale_crop(cls, im, intrinsics, out_h, out_w):
        """Randomly scales and crops image."""

        def scale_randomly(im, intrinsics):
            """Scales image and adjust intrinsics accordingly."""
            in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random.uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            # Add batch.
            im = tf.expand_dims(im, 0)
            im = tf.image.resize(im, [out_h, out_w], method=tf.image.ResizeMethod.AREA)
            im = im[0]
            fx = intrinsics[0, 0] * x_scaling
            fy = intrinsics[1, 1] * y_scaling
            cx = intrinsics[0, 2] * x_scaling
            cy = intrinsics[1, 2] * y_scaling
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def crop_randomly(im, intrinsics, out_h, out_w):
            """Crops image and adjust intrinsics accordingly."""
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            in_h, in_w, _ = tf.unstack(tf.shape(input=im))
            offset_y = tf.random.uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random.uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[1, 2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        im, intrinsics = scale_randomly(im, intrinsics)
        im, intrinsics = crop_randomly(im, intrinsics, out_h, out_w)
        return im, intrinsics

    @classmethod
    def make_intrinsics_matrix(cls, fx, fy, cx, cy):
        r1 = tf.stack([fx, 0, cx])
        r2 = tf.stack([0, fy, cy])
        r3 = tf.constant([0., 0., 1.])
        intrinsics = tf.stack([r1, r2, r3])
        return intrinsics

    @classmethod
    def get_multi_scale_intrinsics(cls, intrinsics, num_scales):
        """Returns multiple intrinsic matrices for different scales."""
        intrinsics_multi_scale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[0, 0] / (2**s)
            fy = intrinsics[1, 1] / (2**s)
            cx = intrinsics[0, 2] / (2**s)
            cy = intrinsics[1, 2] / (2**s)
            intrinsics_multi_scale.append(cls.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_multi_scale = tf.stack(intrinsics_multi_scale)
        return intrinsics_multi_scale
