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
from .models.common import conv1x1, depthwise_conv3x3, conv1x1_block, conv3x3_block, ChannelShuffle, SEBlock,\
    GluonBatchNormalization, MaxPool2d, get_channel_axis, flatten, dwconv3x3_block
import time

import sys
sys.path.append('../kitti_eval/flow_tool/')
sys.path.append('..')
import flowlib as fl


from utils.optical_flow_warp_fwd import TransformerFwd
from utils.optical_flow_warp_old import transformer_old
from utils.loss_utils import SSIM, cal_grad2_error_mask, charbonnier_loss, cal_grad2_error, compute_edge_aware_smooth_loss, ternary_loss, depth_smoothness
from utils.utils import average_gradients, normalize_depth_for_display, preprocess_image, deprocess_image, inverse_warp, inverse_warp_new


def Flow_net(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name='inputs')
    src0 = inputs[:, :, :, :3]
    tgt = inputs[:, :, :, 3:6]
    src1 = inputs[:, :, :, 6:]
#     src0 = tf.keras.Input(shape=input_shape, name='input_src0')
#     tgt = tf.keras.Input(shape=input_shape, name='input_tgt')
#     src1 = tf.keras.Input(shape=input_shape, name='input_src1')
    feature_input_shape = input_shape
    feature_input_shape[-1] = feature_input_shape[-1] // 3
    fpf = feature_pyramid_flow(feature_input_shape)
    fpf.summary()
    
    feature_src0 = fpf(src0)
    feature_tgt = fpf(tgt)
    feature_src1 = fpf(src1)
    
    
    flow = get_flow(input_shape)

     # foward warp: |01 |, | 12|, |0 2| ,direction: ->
    flow_fw0 = flow(feature_src0, feature_tgt)
    flow_fw1 = flow(feature_tgt, feature_src1)
    flow_fw2 = flow(feature_src0, feature_src1)

    # backward warp: |01 |, | 12|, |0 2| , direction: <-
    flow_bw0 = flow(feature_tgt, feature_src0)
    flow_bw1 = flow(feature_src1, feature_tgt)
    flow_bw2 = flow(feature_src1, feature_src0)
    
#     flow_loss = Flow_loss()
#     all_flow = flow_loss([[src0, tgt, src1], [flow_fw0, flow_fw1, flow_fw2,
#                                     flow_bw0, flow_bw1, flow_bw2]])
#     flow_fw0, flow_fw1, flow_fw2, flow_bw0, flow_bw1, flow_bw2 = all_flow

    # 每個 flow_output shape: (n, 256, 832, 2), (n, 128, 416, 2), (n, 64, 208, 2), (n, 32, 104, 2)
    model = tf.keras.Model(inputs=inputs, 
                           outputs=[flow_fw0, flow_fw1, flow_fw2,
                                    flow_bw0, flow_bw1, flow_bw2], 
                           name='flow_net')
    model.summary()
#     tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
    return model

# class Flow_loss(nn.Layer):
#     def __init__(self):
#         super().__init__(self)
#         # Model Input 解析度設定
#         self.batch_size = 8
#         self.img_height = 256
#         self.img_width = 832
#         self.num_scale = 4
#         self.num_source = 2

#         # Loss Hyperparameters
#         self.ssim_weight = 0.85
#         self.flow_reconstruction_weight = 1.0
#         self.flow_smooth_weight = 10.0
#         self.flow_cross_geometry_weight = 0.3
#         self.flow_diff_threshold = 4.0 #useless
#         self.flow_consist_weight = 0.01 #useless
    
#     def occulsion(self, pred_flow, H, W):
#         """
#         Here, we compute the soft occlusion maps proposed in https://arxiv.org/pdf/1711.05890.pdf

#         pred_flow: the estimated forward optical flow
#         """
#         transformerFwd = TransformerFwd()
#         occu_mask = [
#             tf.clip_by_value(
#                 transformerFwd(
#                     tf.ones(
#                         shape=[self.batch_size, H, W, 1],
#                         dtype='float32'),
#                     pred_flow, [H , W]),
#                 clip_value_min=0.0,
#                 clip_value_max=1.0)
#             ]
#         occu_mask = tf.reshape(occu_mask, [self.batch_size, H, W, 1])
#         occu_mask_avg = tf.reduce_mean(input_tensor=occu_mask)

#         return occu_mask, occu_mask_avg
    
#     def build_flow_loss(self):
#         reconstructed_loss = 0
#         cross_reconstructed_loss = 0
#         flow_smooth_loss = 0
#         cross_flow_smooth_loss = 0
#         ssim_loss = 0
#         cross_ssim_loss = 0

#         curr_tgt_image_all = []
#         curr_src_image_stack_all = []
#         occlusion_map_0_all = []
#         occlusion_map_1_all = []
#         occlusion_map_2_all = []
#         occlusion_map_3_all = []
#         occlusion_map_4_all = []
#         occlusion_map_5_all = []

#         # Calculate different scale occulsion maps described in 'Occlusion Aware Unsupervised
#         # Learning of Optical Flow by Yang Wang et al'
#         occu_masks_bw = []
#         occu_masks_bw_avg = []
#         occu_masks_fw = []
#         occu_masks_fw_avg = []

#         for i in range(len(self.pred_bw_flows)):
#             temp_occu_masks_bw = []
#             temp_occu_masks_bw_avg = []
#             temp_occu_masks_fw = []
#             temp_occu_masks_fw_avg = []

#             for s in range(self.num_scale):
#                 H = int(self.img_height / (2**s))
#                 W = int(self.img_width  / (2**s))

#                 mask, mask_avg = self.occulsion(self.pred_bw_flows[i][s], H, W)
#                 temp_occu_masks_bw.append(mask)
#                 temp_occu_masks_bw_avg.append(mask_avg)
#                 # [src0, tgt, src0_1]

#                 mask, mask_avg = self.occulsion(self.pred_fw_flows[i][s], H, W)
#                 temp_occu_masks_fw.append(mask)
#                 temp_occu_masks_fw_avg.append(mask_avg)
#                 # [tgt, src1, src1_1]

#             occu_masks_bw.append(temp_occu_masks_bw)
#             occu_masks_bw_avg.append(temp_occu_masks_bw_avg)
#             occu_masks_fw.append(temp_occu_masks_fw)
#             occu_masks_fw_avg.append(temp_occu_masks_fw_avg)

#         for s in range(self.num_scale):
#             H = int(self.img_height / (2**s))
#             W = int(self.img_width  / (2**s))
#             curr_tgt_image = tf.image.resize(
#                 self.tgt_image, [H, W], method=tf.image.ResizeMethod.AREA)
#             curr_src_image_stack = tf.image.resize(
#                 self.src_image_stack, [H, W], method=tf.image.ResizeMethod.AREA)

#             curr_tgt_image_all.append(curr_tgt_image)
#             curr_src_image_stack_all.append(curr_src_image_stack)

#             # src0
#             curr_proj_image_optical_src0 = transformer_old(curr_tgt_image, self.pred_fw_flows[0][s], [H, W])
#             curr_proj_error_optical_src0 = tf.abs(curr_proj_image_optical_src0 - curr_src_image_stack[:,:,:,0:3])
#             reconstructed_loss += tf.reduce_mean(
#                 input_tensor=curr_proj_error_optical_src0 * occu_masks_bw[0][s]) / occu_masks_bw_avg[0][s]

#             curr_proj_image_optical_src0_1 = transformer_old(curr_src_image_stack[:,:,:,3:6], self.pred_fw_flows[2][s], [H, W])
#             curr_proj_error_optical_src0_1 = tf.abs(curr_proj_image_optical_src0_1 - curr_src_image_stack[:,:,:,0:3])
#             cross_reconstructed_loss += tf.reduce_mean(
#                 input_tensor=curr_proj_error_optical_src0_1 * occu_masks_bw[2][s]) / occu_masks_bw_avg[2][s]

#             # tgt
#             curr_proj_image_optical_tgt = transformer_old(curr_src_image_stack[:,:,:,3:6], self.pred_fw_flows[1][s], [H, W])
#             curr_proj_error_optical_tgt = tf.abs(curr_proj_image_optical_tgt - curr_tgt_image)
#             reconstructed_loss += tf.reduce_mean(
#                 input_tensor=curr_proj_error_optical_tgt * occu_masks_bw[1][s]) / occu_masks_bw_avg[1][s]

#             curr_proj_image_optical_tgt_1 = transformer_old(curr_src_image_stack[:,:,:,0:3], self.pred_bw_flows[0][s], [H, W])
#             curr_proj_error_optical_tgt_1 = tf.abs(curr_proj_image_optical_tgt_1 - curr_tgt_image)
#             reconstructed_loss += tf.reduce_mean(
#                 input_tensor=curr_proj_error_optical_tgt_1 * occu_masks_fw[0][s]) / occu_masks_fw_avg[0][s]

#             # src1
#             curr_proj_image_optical_src1 = transformer_old(curr_tgt_image, self.pred_bw_flows[1][s], [H, W])
#             curr_proj_error_optical_src1 = tf.abs(curr_proj_image_optical_src1 - curr_src_image_stack[:,:,:,3:6])
#             reconstructed_loss += tf.reduce_mean(
#                 input_tensor=curr_proj_error_optical_src1 * occu_masks_fw[1][s]) / occu_masks_fw_avg[1][s]

#             curr_proj_image_optical_src1_1 = transformer_old(curr_src_image_stack[:,:,:,0:3], self.pred_bw_flows[2][s], [H, W])
#             curr_proj_error_optical_src1_1 = tf.abs(curr_proj_image_optical_src1_1 - curr_src_image_stack[:,:,:,3:6])
#             cross_reconstructed_loss += tf.reduce_mean(
#                 input_tensor=curr_proj_error_optical_src1_1 * occu_masks_fw[2][s]) / occu_masks_fw_avg[2][s]

#             if self.ssim_weight > 0:
#                 # src0
#                 ssim_loss += tf.reduce_mean(
#                     input_tensor=SSIM(curr_proj_image_optical_src0 * occu_masks_bw[0][s],
#                          curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[0][s])) / occu_masks_bw_avg[0][s]

#                 cross_ssim_loss += tf.reduce_mean(
#                     input_tensor=SSIM(curr_proj_image_optical_src0_1 * occu_masks_bw[2][s],
#                          curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[2][s])) / occu_masks_bw_avg[2][s]

#                 # tgt
#                 ssim_loss += tf.reduce_mean(
#                     input_tensor=SSIM(curr_proj_image_optical_tgt * occu_masks_bw[1][s],
#                          curr_tgt_image * occu_masks_bw[1][s])) / occu_masks_bw_avg[1][s]

#                 ssim_loss += tf.reduce_mean(
#                     input_tensor=SSIM(curr_proj_image_optical_tgt_1 * occu_masks_fw[0][s],
#                          curr_tgt_image * occu_masks_fw[0][s])) / occu_masks_fw_avg[0][s]

#                 # src1
#                 ssim_loss += tf.reduce_mean(
#                     input_tensor=SSIM(curr_proj_image_optical_src1 * occu_masks_fw[1][s],
#                          curr_src_image_stack[:,:,:,3:6] * occu_masks_fw[1][s])) / occu_masks_fw_avg[1][s]

#                 cross_ssim_loss += tf.reduce_mean(
#                     input_tensor=SSIM(curr_proj_image_optical_src1_1 * occu_masks_fw[2][s],
#                          curr_src_image_stack[:,:,:,3:6] * occu_masks_fw[2][s])) / occu_masks_fw_avg[2][s]

#             # Compute second-order derivatives for flow smoothness loss
#             flow_smooth_loss += cal_grad2_error(
#                 self.pred_fw_flows[0][s] / 20.0, curr_src_image_stack[:,:,:,0:3], 1.0)

#             flow_smooth_loss += cal_grad2_error(
#                 self.pred_fw_flows[1][s] / 20.0, curr_tgt_image, 1.0)

#             cross_flow_smooth_loss += cal_grad2_error(
#                 self.pred_fw_flows[2][s] / 20.0, curr_src_image_stack[:,:,:,0:3], 1.0)

#             flow_smooth_loss += cal_grad2_error(
#                 self.pred_bw_flows[0][s] / 20.0, curr_tgt_image, 1.0)

#             flow_smooth_loss += cal_grad2_error(
#                 self.pred_bw_flows[1][s] / 20.0, curr_src_image_stack[:,:,:,3:6], 1.0)

#             cross_flow_smooth_loss += cal_grad2_error(
#                 self.pred_bw_flows[2][s] / 20.0, curr_src_image_stack[:,:,:,3:6], 1.0)

#             # [TODO] Add first-order derivatives for flow smoothness loss
#             # [TODO] use robust Charbonnier penalty?

#             if s == 0:
#                 occlusion_map_0_all = occu_masks_bw[0][s]
#                 occlusion_map_1_all = occu_masks_bw[1][s]
#                 occlusion_map_2_all = occu_masks_bw[2][s]
#                 occlusion_map_3_all = occu_masks_fw[0][s]
#                 occlusion_map_4_all = occu_masks_fw[1][s]
#                 occlusion_map_5_all = occu_masks_fw[2][s]

#         self.total_losses = self.flow_reconstruction_weight * ((1.0 - self.ssim_weight) * \
#                       (reconstructed_loss + self.flow_cross_geometry_weight*cross_reconstructed_loss) + \
#                         self.ssim_weight*(ssim_loss+self.flow_cross_geometry_weight*cross_ssim_loss)) + \
#                       self.flow_smooth_weight * (flow_smooth_loss + self.flow_cross_geometry_weight*cross_flow_smooth_loss)

# #         summaries = []
# #         summaries.append(tf.compat.v1.summary.scalar("total_losses", self.total_losses))
# #         summaries.append(tf.compat.v1.summary.scalar("reconstructed_loss", reconstructed_loss))
# #         summaries.append(tf.compat.v1.summary.scalar("cross_reconstructed_loss", cross_reconstructed_loss))
# #         summaries.append(tf.compat.v1.summary.scalar("ssim_loss", ssim_loss))
# #         summaries.append(tf.compat.v1.summary.scalar("cross_ssim_loss", cross_ssim_loss))
# #         summaries.append(tf.compat.v1.summary.scalar("flow_smooth_loss", flow_smooth_loss))
# #         summaries.append(tf.compat.v1.summary.scalar("cross_flow_smooth_loss", cross_flow_smooth_loss))

# #         s = 0
# #         tf.compat.v1.summary.image('scale%d_target_image' % s, tf.image.convert_image_dtype(curr_tgt_image_all[0], dtype=tf.uint8))

# #         for i in range(self.num_source):
# #             tf.compat.v1.summary.image('scale%d_src_image_%d' % (s, i), \
# #                             tf.image.convert_image_dtype(curr_src_image_stack_all[0][:, :, :, i*3:(i+1)*3], dtype=tf.uint8))

# #         tf.compat.v1.summary.image('scale%d_flow_src02tgt' % s, fl.flow_to_color(self.pred_fw_flows[0][s], max_flow=256))
# #         tf.compat.v1.summary.image('scale%d_flow_tgt2src1' % s, fl.flow_to_color(self.pred_fw_flows[1][s], max_flow=256))
# #         tf.compat.v1.summary.image('scale%d_flow_src02src1' % s, fl.flow_to_color(self.pred_fw_flows[2][s], max_flow=256))
# #         tf.compat.v1.summary.image('scale%d_flow_tgt2src0' % s, fl.flow_to_color(self.pred_bw_flows[0][s], max_flow=256))
# #         tf.compat.v1.summary.image('scale%d_flow_src12tgt' % s, fl.flow_to_color(self.pred_bw_flows[1][s], max_flow=256))
# #         tf.compat.v1.summary.image('scale%d_flow_src12src0' % s, fl.flow_to_color(self.pred_bw_flows[2][s], max_flow=256))

# #         tf.compat.v1.summary.image('scale_flyout_mask_src0', occlusion_map_0_all)
# #         tf.compat.v1.summary.image('scale_flyout_mask_tgt', occlusion_map_1_all)
# #         tf.compat.v1.summary.image('scale_flyout_mask_src0_1', occlusion_map_2_all)
# #         tf.compat.v1.summary.image('scale_flyout_mask_tgt1', occlusion_map_3_all)
# #         tf.compat.v1.summary.image('scale_flyout_mask_src1', occlusion_map_4_all)
# #         tf.compat.v1.summary.image('scale_flyout_mask_src1_1', occlusion_map_5_all)

# #         self.summ_op = tf.compat.v1.summary.merge(summaries)

    
#     def call(self, inputs):
#         input_images, output_from_model = inputs
#         self.tgt_image = input_images[1]
#         self.src_image_stack = tf.concat([input_images[0], input_images[2]], axis=3)
#         self.pred_fw_flows = output_from_model[:3]
#         self.pred_bw_flows = output_from_model[3:]
        
#         self.build_flow_loss()
        
#         self.add_loss(self.total_losses, inputs=True)
#         self.add_metric(self.total_losses, aggregation='mean', name="flow_loss")
#         return output_from_model

class get_flow(nn.Layer):
    def __init__(self, input_shape):
        super().__init__(self)
        self.d = 4
        self.shape, self.decoder_shape = self.get_allShape(input_shape, d=self.d)
        self.context = context_net()
        self.decoder2 = pwc_decoder(self.decoder_shape[0])
        self.decoder3 = pwc_decoder(self.decoder_shape[1])
        self.decoder4 = pwc_decoder(self.decoder_shape[2])
        self.decoder5 = pwc_decoder(self.decoder_shape[3])
        self.decoder6 = pwc_decoder(self.decoder_shape[4])
    
    def cost_volumn(self, feature1, feature2, d=4):
        n, h, w, c = feature1.get_shape().as_list()
        feature2 = tf.pad(tensor=feature2, paddings=[[0, 0], [d, d], [
                              d, d], [0, 0]], mode="CONSTANT")
        cv = []
        for i in range(2 * d + 1):
            for j in range(2 * d + 1):
                cv.append(
                        tf.math.reduce_mean(
                            input_tensor=feature1 *
                            feature2[:, i:(i + h), j:(j + w), :],
                            axis=3,
                            keepdims=True))
        x = tf.concat(cv, axis=3)
        return x
    
    
    def get_allShape(self, input_shape, d=4):
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
    
    @tf.function
    def call(self, flow1, flow2):
        
        f11, f12, f13, f14, f15, f16 = flow1
        f21, f22, f23, f24, f25, f26 = flow2
        
        # Block6
        cv6 = self.cost_volumn(f16, f26, d=self.d)
        flow6, _ = self.decoder6(cv6)

        # Block5
        flow65 = tf.scalar_mul(2, tf.image.resize(flow6, self.shape[5], method=tf.image.ResizeMethod.BILINEAR))
        f25_warp = transformer_old(f25, flow65, self.shape[5])
        cv5 = self.cost_volumn(f15, f25_warp, d=self.d)
        flow5, _ = self.decoder5(tf.concat([cv5, f15, flow65], axis=3)) #2
        flow5 = tf.math.add(flow5, flow65)

        # Block4
        flow54 = tf.scalar_mul(2.0, tf.image.resize(flow5, self.shape[4], method=tf.image.ResizeMethod.BILINEAR))
        f24_warp = transformer_old(f24, flow54, self.shape[4])
        cv4 = self.cost_volumn(f14, f24_warp, d=self.d)
        flow4, _ = self.decoder4(tf.concat([cv4, f14, flow54], axis=3)) #2
        flow4 = tf.math.add(flow4, flow54)

        # Block3
        flow43 = tf.scalar_mul(2.0 ,tf.image.resize(flow4, self.shape[3],
                                                 method=tf.image.ResizeMethod.BILINEAR))
        f23_warp = transformer_old(f23, flow43, self.shape[3])
        cv3 = self.cost_volumn(f13, f23_warp, d=self.d)
        flow3, _ = self.decoder3(tf.concat([cv3, f13, flow43], axis=3)) #2
        flow3 = tf.math.add(flow3, flow43)
        # Block2
        flow32 = tf.scalar_mul(2.0, tf.image.resize(flow3, self.shape[2], method=tf.image.ResizeMethod.BILINEAR))
        f22_warp = transformer_old(f22, flow32, self.shape[2])
        cv2 = self.cost_volumn(f12, f22_warp, d=self.d)
        flow2_raw, flow2_ = self.decoder2(tf.concat([cv2, f12, flow32], axis=3)) #2
        flow2_raw = tf.math.add(flow2_raw, flow32) #10
        
        
        
        
        # context_net
        flow2 = self.context(tf.concat([flow2_raw, flow2_], axis=3))
        flow2 = tf.math.add(flow2, flow2_raw)

        flow0_enlarge = tf.image.resize(
                tf.scalar_mul(4.0, flow2), self.shape[0], method=tf.image.ResizeMethod.BILINEAR)
        flow1_enlarge = tf.image.resize(
                tf.scalar_mul(4.0, flow3), self.shape[1], method=tf.image.ResizeMethod.BILINEAR)
        flow2_enlarge = tf.image.resize(
                tf.scalar_mul(4.0, flow4), self.shape[2], method=tf.image.ResizeMethod.BILINEAR)
        flow3_enlarge = tf.image.resize(
                tf.scalar_mul(4.0, flow5), self.shape[3], method=tf.image.ResizeMethod.BILINEAR)
        
        #output shape: (n, 256, 832, 2), (n, 128, 416, 2), (n, 64, 208, 2), (n, 32, 104, 2)
        return flow0_enlarge, flow1_enlarge, flow2_enlarge, flow3_enlarge
    



    

def feature_pyramid_flow(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name='input_image')
    
    cnv1 = conv3x3_block(3, 16, strides=2, padding=1, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(inputs)
    cnv2 = conv3x3_block(16, 16, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv1)
    cnv3 = conv3x3_block(16, 32, strides=2, padding=1, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv2)
    cnv4 = conv3x3_block(32, 32, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv3)
    cnv5 = conv3x3_block(32, 64, strides=2, padding=1, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv4)
    cnv6 = conv3x3_block(64, 64, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv5)
    cnv7 = conv3x3_block(16, 96, strides=2, padding=1, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv6)
    cnv8 = conv3x3_block(32, 96, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv7)
    cnv9 = conv3x3_block(16, 128, strides=2, padding=1, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv8)
    cnv10 = conv3x3_block(32, 128, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv9)
    cnv11 = conv3x3_block(16, 192, strides=2, padding=1, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv10)
    cnv12 = conv3x3_block(32, 192, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv11)
    
    layers = tf.keras.Model(inputs=inputs, 
                            outputs=[cnv2, cnv4, cnv6, cnv8, cnv10, cnv12],
                            name='feature_pyramid_flow')
 
    return layers    


def pwc_decoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name='input_image')
    
    cnv1 = conv3x3_block(128, 128, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(inputs)
    cnv2 = conv3x3_block(128, 128, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(cnv1)
    cnv3 = conv3x3_block(128, 96, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(nn.concatenate([cnv1, cnv2]))
    cnv4 = conv3x3_block(96, 64, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(nn.concatenate([cnv2, cnv3]))
    cnv5 = conv3x3_block(64, 32, use_bn=False, activation=nn.LeakyReLU(alpha=0.1))(nn.concatenate([cnv3, cnv4]))
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




# @tf.function
# def leaky_relu(_x, alpha=0.1):
#     pos = tf.nn.relu(_x)
#     neg = alpha * (_x - abs(_x)) * 0.5

#     return pos + neg



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
    tf.compat.v1.disable_eager_execution()
#     if(tf.executing_eagerly()):
#         print('[Info] Eager execution')
#         print('Eager execution is enabled (running operations immediately)\n')
#         print(('Turn eager execution off by running: \n{0}\n{1}').format('' \
#             'from tensorflow.python.framework.ops import disable_eager_execution', \
#             'tf.compat.v1.disable_eager_execution()'))
#     else:
#         print('[Info] Graph execution')
#         print('You are not running eager execution. TensorFlow version >= 2.0.0' \
#               'has eager execution enabled by default.')
#         print(('Turn on eager execution by running: \n\n{0}\n\nOr upgrade '\
#                'your tensorflow version by running:\n\n{1}').format(
#                'tf.compat.v1.enable_eager_execution()',
#                '!pip install --upgrade tensorflow\n' \
#                '!pip install --upgrade tensorflow-gpu'))
    
    
    @tf.function
    def test(inputs, model):
        x = model(inputs)
    
    batch_size = 10
    src0 = tf.random.normal((batch_size, 256, 832, 3))
    tgt = tf.random.normal((batch_size, 256, 832, 3))
    src1 = tf.random.normal((batch_size, 256, 832, 3))
    
    model = Flow_net(input_shape=(256, 832, 3))
    
    
    for var in model.trainable_variables:
        print(var.name)
#     for wei in net.weights:
#         print(wei)
    
    for layer in model.layers:
        print(layer.name)
    model.summary()
        
    #calculate fps
    averageFPS = 0
    for times in range(10):
        start_time = time.time()
        for num in range(10):
            x = tf.random.normal((1, 256, 832, 3))
            y = tf.random.normal((1, 256, 832, 3))
            z = tf.random.normal((1, 256, 832, 3))
            
            test([x, y, z], model)
        total_time = time.time() - start_time
        FPS = 10 / total_time
        averageFPS += FPS
    averageFPS /= 10
    print("[Info] Flow Net FPS: {:.3f}".format(averageFPS))