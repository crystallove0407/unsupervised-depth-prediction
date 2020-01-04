from __future__ import division
import numpy as np
import tensorflow as tf
import time



# class Pose_net(tf.keras.Model):
#     def __init__(self, joint_encoder):
#         super().__init__()
#         self.joint_encoder = joint_encoder
#         self.pose_trunk = pose_trunk()
#         self.cnv1 = tf.keras.layers.Conv2D(
#             256, 3, strides=2, padding='same', activation=tf.nn.relu)
#         self.cnv2 = tf.keras.layers.Conv2D(
#             256, 3, strides=1, padding='same', activation=tf.nn.relu)
#         self.cnv3 = tf.keras.layers.Conv2D(
#             256, 3, strides=2, padding='same', activation=tf.nn.relu)
#         self.cnv4 = tf.keras.layers.Conv2D(
#             256, 3, strides=1, padding='same', activation=tf.nn.relu)
#         self.cnvpredict = tf.keras.layers.Conv2D(
#             6*6, 1, padding='same', strides=1)

#     def call(self, inputs, disp_bottleneck_stack=None):
#         x = disp_bottleneck_stack if self.joint_encoder else self.pose_trunk(
#             inputs)

#         x = self.cnv1(x)
#         x = self.cnv2(x)
#         x = self.cnv3(x)
#         x = self.cnv4(x)
#         x = self.cnvpredict(x)
#         x = tf.math.reduce_mean(input_tensor=x, axis=[1, 2])
#         pose_final = tf.reshape(x, [-1, 1, 6*6])

#         tran_mag = 0.001 if self.joint_encoder else 1.0
#         rot_mag = 0.01
#         pose_final = tf.concat(
#             [  # 0: src0 -> tgt
#                 tran_mag * pose_final[:, :, 0:3],
#                 rot_mag * pose_final[:, :, 3:6],
#                 # 1: tgt -> src1
#                 tran_mag * pose_final[:, :, 6:9],
#                 rot_mag * pose_final[:, :, 9:12],
#                 # 2: src0 -> src1
#                 tran_mag * pose_final[:, :, 12:15],
#                 rot_mag * pose_final[:, :, 15:18],
#                 # 3: tgt -> src0
#                 tran_mag * pose_final[:, :, 18:21],
#                 rot_mag * pose_final[:, :, 21:24],
#                 # 4: src1 -> tgt
#                 tran_mag * pose_final[:, :, 24:27],
#                 rot_mag * pose_final[:, :, 27:30],
#                 # 5: src1 -> src0
#                 tran_mag * pose_final[:, :, 30:33],
#                 rot_mag * pose_final[:, :, 33:36]
#             ], axis=2)

#         return pose_final

def Pose_net(input_shape, disp_bottleneck_stack=None, joint_encoder=False):
    if joint_encoder:
        input_shape = disp_bottleneck_stack.get_shape().as_list()[1:]
        
    inputs = tf.keras.Input(shape=input_shape, name='pose_inputs')
    x = inputs
    
    if not joint_encoder:
        trunk = pose_trunk()
        x = trunk(x)
    
    x = tf.keras.layers.Conv2D(
            256, 3, strides=2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D(
            256, 3, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D(
            256, 3, strides=2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D(
            256, 3, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D(
            6*6, 1, padding='same', strides=1)(x)
    x = tf.math.reduce_mean(input_tensor=x, axis=[1, 2])
    pose_final = tf.reshape(x, [-1, 1, 6*6])
    
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
    
    model = tf.keras.Model(inputs=inputs, 
                           outputs=pose_final, 
                           name='pose_net')
    model.summary()
    
    return model
    
    

class pose_trunk(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(
            16, 7, strides=2, padding='same', activation=tf.nn.relu)
        self.conv2d_2 = tf.keras.layers.Conv2D(
            16, 7, strides=1, padding='same', activation=tf.nn.relu)
        self.conv2d_3 = tf.keras.layers.Conv2D(
            32, 5, strides=2, padding='same', activation=tf.nn.relu)
        self.conv2d_4 = tf.keras.layers.Conv2D(
            32, 5, strides=1, padding='same', activation=tf.nn.relu)
        self.conv2d_5 = tf.keras.layers.Conv2D(
            64, 3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv2d_6 = tf.keras.layers.Conv2D(
            64, 3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv2d_7 = tf.keras.layers.Conv2D(
            128, 3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv2d_8 = tf.keras.layers.Conv2D(
            128, 3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv2d_9 = tf.keras.layers.Conv2D(
            256, 3, strides=2, padding='same', activation=tf.nn.relu)
        self.conv2d_10 = tf.keras.layers.Conv2D(
            256, 3, strides=1, padding='same', activation=tf.nn.relu)

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

        return cnv10

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    
    @tf.function
    def test(inputs, model):
        x = model(inputs)
    
    batch_size = 8
    input_shape = (batch_size, 256, 416, 9)
    
    model = Pose_net(input_shape=input_shape[1:], disp_bottleneck_stack=None, joint_encoder=False)
    
#     for var in model.trainable_variables:
#         print(var.name)
#     for wei in model.weights:
#         print(wei)
    
    for layer in model.layers:
        print(layer.name)
    
        
    #calculate fps
    averageFPS = 0
    for _ in range(10):
        start_time = time.time()
        for num in range(10):
            print('[info] Img', num)
            x = tf.random.normal(input_shape)
            test(x, model)
            
        total_time = time.time() - start_time
        FPS = 10 / total_time
        averageFPS += FPS
    averageFPS /= 10
    print("[Info] Pose Net FPS: {:.3f}".format(averageFPS))
    model.summary()