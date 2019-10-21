import tensorflow as tf
import sys
sys.path.insert(0, '../kitti_eval/flow_tool/')
import flowlib as fl

from nets.depth_net import D_Net
from nets.pose_net import P_Net5
from nets.flow_net import construct_model_pwc_full, feature_pyramid_flow

from utils.optical_flow_warp_fwd import transformerFwd
from utils.optical_flow_warp_old import transformer_old

from utils.loss_utils import SSIM, cal_grad2_error_mask, charbonnier_loss, cal_grad2_error, compute_edge_aware_smooth_loss, ternary_loss, depth_smoothness
from utils.utils import average_gradients, normalize_depth_for_display, preprocess_image, deprocess_image, inverse_warp, inverse_warp_new


class Model(object):
    def __init__(self, tgt_image, src_image_stack, tgt_image_norm, src_image_stack_norm, cam2pix, pix2cam, batch_size, img_height, img_width, mode, scope=None, reuse_scope=None):
        self.tgt_image = tgt_image
        self.src_image_stack = src_image_stack
        self.tgt_image_norm = tgt_image_norm
        self.src_image_stack_norm = src_image_stack_norm
        self.proj_cam2pix = cam2pix
        self.proj_pix2cam = pix2cam

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.num_source = 4
        self.seq_length = self.num_source + 1
        self.num_scales = 4
        self.weight_reg = 0.05

        # Loss Hyperparameters
        self.ssim_weight = 0.85

        self.flow_reconstruction_weight = 1.0
        self.flow_smooth_weight = 10.0
        self.flow_diff_threshold = 4.0
        self.flow_consist_weight = 0.01
        self.flow_cross_geometry_weight = 0.3

        self.dp_reconstruction_weight = 50.0
        self.dp_smooth_weight = 1.0
        self.dp_cross_geometry_weight = 0.8

        self.compute_minimum_loss = False
        self.is_depth_upsampling = False # [!!!!] Cannot work. => cause depth smoothness loss to be 0
        self.joint_encoder = False
        self.equal_weighting = False  # equal weight for depth smoothness loss
        self.depth_normalization = False # depth normalization for depth smoothness loss
        self.scale_normalize = True # depth normalization for all loss

        self.build_model(scale_normalize=self.scale_normalize, scope=scope, reuse_scope=reuse_scope)
        self.build_losses()

    def build_model(self, scale_normalize=True, fix_pose=False, scope=None, reuse_scope=False):
        if self.mode == 'train_flow':
            self.build_flow_model_final(scope=scope, reuse_scope=reuse_scope)
        elif self.mode == 'train_dp':
            self.build_dp_model_final(scale_normalize=scale_normalize, scope=scope, reuse_scope=reuse_scope)

    def build_flow_model_final(self, scope=None, reuse_scope=False):
        print("[Info] Building flow network ...")
        print("[Info] img_height:", self.img_height, "img_width", self.img_width)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            feature0_flow = feature_pyramid_flow(self.src_image_stack[:,:,:,0:3], reuse=False)
            feature1_flow = feature_pyramid_flow(self.src_image_stack[:,:,:,3:6], reuse=True)
            feature2_flow = feature_pyramid_flow(self.tgt_image, reuse=True)
            feature3_flow = feature_pyramid_flow(self.src_image_stack[:,:,:,6:9], reuse=True)
            feature4_flow = feature_pyramid_flow(self.src_image_stack[:,:,:,9:12], reuse=True)

            flow_fw0 = construct_model_pwc_full(self.src_image_stack[:,:,:,3:6], self.tgt_image, feature1_flow, feature2_flow)

        with tf.variable_scope(scope, reuse=True):
            flow_fw1 = construct_model_pwc_full(self.tgt_image, self.src_image_stack[:,:,:,6:9], feature2_flow, feature3_flow)
            flow_fw2 = construct_model_pwc_full(self.src_image_stack[:,:,:,3:6], self.src_image_stack[:,:,:,6:9], feature1_flow, feature3_flow)
            flow_fw3 = construct_model_pwc_full(self.src_image_stack[:,:,:,0:3], self.src_image_stack[:,:,:,3:6], feature0_flow, feature1_flow)
            flow_fw4 = construct_model_pwc_full(self.src_image_stack[:,:,:,6:9], self.src_image_stack[:,:,:,9:12], feature3_flow, feature4_flow)
            flow_fw5 = construct_model_pwc_full(self.src_image_stack[:,:,:,0:3], self.tgt_image, feature0_flow, feature2_flow)
            flow_fw6 = construct_model_pwc_full(self.tgt_image, self.src_image_stack[:,:,:,9:12], feature2_flow, feature4_flow)
            flow_fw7 = construct_model_pwc_full(self.src_image_stack[:,:,:,0:3], self.src_image_stack[:,:,:,6:9], feature0_flow, feature3_flow)
            flow_fw8 = construct_model_pwc_full(self.src_image_stack[:,:,:,3:6], self.src_image_stack[:,:,:,9:12], feature1_flow, feature4_flow)
            flow_fw9 = construct_model_pwc_full(self.src_image_stack[:,:,:,0:3], self.src_image_stack[:,:,:,9:12], feature0_flow, feature4_flow)

            flow_bw0 = construct_model_pwc_full(self.tgt_image, self.src_image_stack[:,:,:,3:6], feature2_flow, feature1_flow)
            flow_bw1 = construct_model_pwc_full(self.src_image_stack[:,:,:,6:9], self.tgt_image, feature3_flow, feature2_flow)
            flow_bw2 = construct_model_pwc_full(self.src_image_stack[:,:,:,6:9], self.src_image_stack[:,:,:,3:6], feature3_flow, feature1_flow)
            flow_bw3 = construct_model_pwc_full(self.src_image_stack[:,:,:,3:6], self.src_image_stack[:,:,:,0:3], feature1_flow, feature0_flow)
            flow_bw4 = construct_model_pwc_full(self.src_image_stack[:,:,:,9:12], self.src_image_stack[:,:,:,6:9], feature4_flow, feature3_flow)
            flow_bw5 = construct_model_pwc_full(self.tgt_image, self.src_image_stack[:,:,:,0:3], feature2_flow, feature0_flow)
            flow_bw6 = construct_model_pwc_full(self.src_image_stack[:,:,:,9:12], self.tgt_image, feature4_flow, feature2_flow)
            flow_bw7 = construct_model_pwc_full(self.src_image_stack[:,:,:,6:9], self.src_image_stack[:,:,:,0:3], feature3_flow, feature0_flow)
            flow_bw8 = construct_model_pwc_full(self.src_image_stack[:,:,:,9:12], self.src_image_stack[:,:,:,3:6], feature4_flow, feature1_flow)
            flow_bw9 = construct_model_pwc_full(self.src_image_stack[:,:,:,9:12], self.src_image_stack[:,:,:,0:3], feature4_flow, feature0_flow)

        self.pred_fw_flows = [flow_fw0, flow_fw1, flow_fw2, flow_fw3, flow_fw4, flow_fw5, flow_fw6, flow_fw7, flow_fw8, flow_fw9]
        self.pred_bw_flows = [flow_bw0, flow_bw1, flow_bw2, flow_bw3, flow_bw4, flow_bw5, flow_bw6, flow_bw7, flow_bw8, flow_bw9]

    def build_dp_model_final(self, scale_normalize=True, scope=None, reuse_scope=False):
        print("[Info] Building depth and pose network ...")
        print("[Info] img_height:", self.img_height, "img_width", self.img_width)

        self.disp = {}
        self.depth = {}
        self.depth_upsampled = {}
        self.disp_bottleneck = {}

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            tgt_pred_disp, tgt_disp_bottlenecks = D_Net(self.tgt_image_norm, weight_reg=self.weight_reg, is_training=True, reuse=False)
            if scale_normalize:
                # As proposed in https://arxiv.org/abs/1712.00175, this can
                # bring improvement in depth estimation, but not included in our paper.
                tgt_pred_disp = [self.spatial_normalize(disp) for disp in tgt_pred_disp]
            tgt_pred_depth = [1. / d for d in tgt_pred_disp]
            self.disp['tgt'] = tgt_pred_disp
            self.depth['tgt'] = tgt_pred_depth
            self.disp_bottleneck['tgt'] = tgt_disp_bottlenecks
            if self.is_depth_upsampling:
                self.depth_upsampled['tgt'] = []
                for s in range(len(tgt_pred_depth)):
                    self.depth_upsampled['tgt'].append(tf.image.resize_bilinear(
                        tgt_pred_depth[s], [self.img_height, self.img_width],
                        align_corners=True))

            for i in range(self.num_source):
                temp_disp, temp_disp_bottlenecks = D_Net(self.src_image_stack_norm[:,:,:,3*i:3*(i+1)], weight_reg=self.weight_reg, is_training=True, reuse=True)
                if scale_normalize:
                    temp_disp = [self.spatial_normalize(disp) for disp in temp_disp]
                temp_depth = [1./d for d in temp_disp]
                name = 'src{}'.format(i)
                self.disp[name] = temp_disp
                self.depth[name] = temp_depth
                self.disp_bottleneck[name] = temp_disp_bottlenecks
                self.depth_upsampled[name] = []
                for s in range(len(temp_depth)):
                    self.depth_upsampled[name].append(tf.image.resize_bilinear(
                        temp_depth[s], [self.img_height, self.img_width],
                        align_corners=True))

            if self.joint_encoder:
                disp_bottleneck_stack = tf.concat([self.disp_bottleneck['src0'], self.disp_bottleneck['tgt'], self.disp_bottleneck['src1']], axis=3)
            else:
                disp_bottleneck_stack = None

            pose_inputs = tf.concat([self.src_image_stack_norm[:,:,:,0:3], self.src_image_stack_norm[:,:,:,3:6], self.tgt_image_norm, self.src_image_stack_norm[:,:,:,6:9], self.src_image_stack_norm[:,:,:,9:12]], axis=3)
            self.pred_poses = P_Net5(pose_inputs, disp_bottleneck_stack, self.joint_encoder, self.weight_reg)

            feature0_flow = feature_pyramid_flow(self.src_image_stack_norm[:,:,:,0:3], reuse=False)
            feature1_flow = feature_pyramid_flow(self.src_image_stack_norm[:,:,:,3:6], reuse=True)
            feature2_flow = feature_pyramid_flow(self.tgt_image_norm, reuse=True)
            feature3_flow = feature_pyramid_flow(self.src_image_stack_norm[:,:,:,6:9], reuse=True)
            feature4_flow = feature_pyramid_flow(self.src_image_stack_norm[:,:,:,9:12], reuse=True)

            flow_fw0 = construct_model_pwc_full(self.src_image_stack_norm[:,:,:,3:6], self.tgt_image_norm, feature1_flow, feature2_flow)
            flow_fw1 = construct_model_pwc_full(self.tgt_image_norm, self.src_image_stack_norm[:,:,:,6:9], feature2_flow, feature3_flow)
            flow_fw2 = construct_model_pwc_full(self.src_image_stack_norm[:,:,:,3:6], self.src_image_stack_norm[:,:,:,6:9], feature1_flow, feature3_flow)
            flow_fw3 = construct_model_pwc_full(self.src_image_stack_norm[:,:,:,0:3], self.src_image_stack_norm[:,:,:,3:6], feature0_flow, feature1_flow)
            flow_fw4 = construct_model_pwc_full(self.src_image_stack_norm[:,:,:,6:9], self.src_image_stack_norm[:,:,:,9:12], feature3_flow, feature4_flow)
            flow_fw5 = construct_model_pwc_full(self.src_image_stack_norm[:,:,:,0:3], self.tgt_image_norm, feature0_flow, feature2_flow)
            flow_fw6 = construct_model_pwc_full(self.tgt_image_norm, self.src_image_stack_norm[:,:,:,9:12], feature2_flow, feature4_flow)
            flow_fw7 = construct_model_pwc_full(self.src_image_stack[:,:,:,0:3], self.src_image_stack[:,:,:,6:9], feature0_flow, feature3_flow)
            flow_fw8 = construct_model_pwc_full(self.src_image_stack[:,:,:,3:6], self.src_image_stack[:,:,:,9:12], feature1_flow, feature4_flow)
            flow_fw9 = construct_model_pwc_full(self.src_image_stack[:,:,:,0:3], self.src_image_stack[:,:,:,9:12], feature0_flow, feature4_flow)

            flow_bw0 = construct_model_pwc_full(self.tgt_image, self.src_image_stack[:,:,:,3:6], feature2_flow, feature1_flow)
            flow_bw1 = construct_model_pwc_full(self.src_image_stack[:,:,:,6:9], self.tgt_image, feature3_flow, feature2_flow)
            flow_bw2 = construct_model_pwc_full(self.src_image_stack[:,:,:,6:9], self.src_image_stack[:,:,:,3:6], feature3_flow, feature1_flow)
            flow_bw3 = construct_model_pwc_full(self.src_image_stack[:,:,:,3:6], self.src_image_stack[:,:,:,0:3], feature1_flow, feature0_flow)
            flow_bw4 = construct_model_pwc_full(self.src_image_stack[:,:,:,9:12], self.src_image_stack[:,:,:,6:9], feature4_flow, feature3_flow)
            flow_bw5 = construct_model_pwc_full(self.tgt_image, self.src_image_stack[:,:,:,0:3], feature2_flow, feature0_flow)
            flow_bw6 = construct_model_pwc_full(self.src_image_stack[:,:,:,9:12], self.tgt_image, feature4_flow, feature2_flow)
            flow_bw7 = construct_model_pwc_full(self.src_image_stack[:,:,:,6:9], self.src_image_stack[:,:,:,0:3], feature3_flow, feature0_flow)
            flow_bw8 = construct_model_pwc_full(self.src_image_stack[:,:,:,9:12], self.src_image_stack[:,:,:,3:6], feature4_flow, feature1_flow)
            flow_bw9 = construct_model_pwc_full(self.src_image_stack[:,:,:,9:12], self.src_image_stack[:,:,:,0:3], feature4_flow, feature0_flow)

            self.pred_fw_flows = [flow_fw0, flow_fw1, flow_fw2, flow_fw3, flow_fw4, flow_fw5, flow_fw6, flow_fw7, flow_fw8, flow_fw9]
            self.pred_bw_flows = [flow_bw0, flow_bw1, flow_bw2, flow_bw3, flow_bw4, flow_bw5, flow_bw6, flow_bw7, flow_bw8, flow_bw9]

    def build_losses(self):
        if self.mode == 'train_flow':
            self.build_flow_loss_final()
        elif self.mode == 'train_dp':
            self.build_dp_loss_mask_allpose()

    def build_flow_loss_final(self):
        reconstructed_loss = 0
        cross_reconstructed_loss = 0
        flow_smooth_loss = 0
        cross_flow_smooth_loss = 0
        ssim_loss = 0
        cross_ssim_loss = 0

        curr_tgt_image_all = []
        curr_src_image_stack_all = []

        occlusion_map_0_all = []
        occlusion_map_1_all = []
        occlusion_map_2_all = []
        occlusion_map_3_all = []
        occlusion_map_4_all = []
        occlusion_map_5_all = []

        # Calculate different scale occulsion maps described in 'Occlusion Aware Unsupervised
        # Learning of Optical Flow by Yang Wang et al'
        occu_masks_bw = []
        occu_masks_bw_avg = []
        occu_masks_fw = []
        occu_masks_fw_avg = []

        for i in range(len(self.pred_bw_flows)):
            temp_occu_masks_bw = []
            temp_occu_masks_bw_avg = []
            temp_occu_masks_fw = []
            temp_occu_masks_fw_avg = []

            for s in range(self.num_scales):
                H = int(self.img_height / (2**s))
                W = int(self.img_width  / (2**s))

                mask, mask_avg = self.occulsion(self.pred_bw_flows[i][s], H, W)
                temp_occu_masks_bw.append(mask)
                temp_occu_masks_bw_avg.append(mask_avg)
                # [src1, tgt, src1_1, src0, src2, src0_1, tgt_1, src0_2, src1_2, src0_3]

                mask, mask_avg = self.occulsion(self.pred_fw_flows[i][s], H, W)
                temp_occu_masks_fw.append(mask)
                temp_occu_masks_fw_avg.append(mask_avg)
                # [tgt, src2, src2_1, src1, src3, tgt_1, src3_1, src2_2, src3_2, src3_3]

            occu_masks_bw.append(temp_occu_masks_bw)
            occu_masks_bw_avg.append(temp_occu_masks_bw_avg)
            occu_masks_fw.append(temp_occu_masks_fw)
            occu_masks_fw_avg.append(temp_occu_masks_fw_avg)

        for s in range(self.num_scales):
            H = int(self.img_height / (2**s))
            W = int(self.img_width  / (2**s))
            curr_tgt_image = tf.image.resize_area(
                self.tgt_image, [H, W])
            curr_src_image_stack = tf.image.resize_area(
                self.src_image_stack, [H, W])

            curr_tgt_image_all.append(curr_tgt_image)
            curr_src_image_stack_all.append(curr_src_image_stack)

            # src0
            curr_proj_image_optical_src0 = transformer_old(curr_src_image_stack[:,:,:,3:6], self.pred_fw_flows[3][s], [H, W])
            curr_proj_error_optical_src0 = tf.abs(curr_proj_image_optical_src0 - curr_src_image_stack[:,:,:,0:3])
            reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src0 * occu_masks_bw[3][s]) / occu_masks_bw_avg[3][s]

            curr_proj_image_optical_src0_1 = transformer_old(curr_tgt_image, self.pred_fw_flows[5][s], [H, W])
            curr_proj_error_optical_src0_1 = tf.abs(curr_proj_image_optical_src0_1 - curr_src_image_stack[:,:,:,0:3])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src0_1 * occu_masks_bw[5][s]) / occu_masks_bw_avg[5][s]

            curr_proj_image_optical_src0_2 = transformer_old(curr_src_image_stack[:,:,:,6:9], self.pred_fw_flows[7][s], [H, W])
            curr_proj_error_optical_src0_2 = tf.abs(curr_proj_image_optical_src0_2 - curr_src_image_stack[:,:,:,0:3])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src0_2 * occu_masks_bw[7][s]) / occu_masks_bw_avg[7][s]

            curr_proj_image_optical_src0_3 = transformer_old(curr_src_image_stack[:,:,:,9:12], self.pred_fw_flows[9][s], [H, W])
            curr_proj_error_optical_src0_3 = tf.abs(curr_proj_image_optical_src0_3 - curr_src_image_stack[:,:,:,0:3])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src0_3 * occu_masks_bw[9][s]) / occu_masks_bw_avg[9][s]

            # src1
            curr_proj_image_optical_src1 = transformer_old(curr_tgt_image, self.pred_fw_flows[0][s], [H, W])
            curr_proj_error_optical_src1 = tf.abs(curr_proj_image_optical_src1 - curr_src_image_stack[:,:,:,3:6])
            reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src1 * occu_masks_bw[0][s]) / occu_masks_bw_avg[0][s]

            curr_proj_image_optical_src1_1 = transformer_old(curr_src_image_stack[:,:,:,6:9], self.pred_fw_flows[2][s], [H, W])
            curr_proj_error_optical_src1_1 = tf.abs(curr_proj_image_optical_src1_1 - curr_src_image_stack[:,:,:,3:6])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src1_1 * occu_masks_bw[2][s]) / occu_masks_bw_avg[2][s]

            curr_proj_image_optical_src1_2 = transformer_old(curr_src_image_stack[:,:,:,0:3], self.pred_bw_flows[3][s], [H, W])
            curr_proj_error_optical_src1_2 = tf.abs(curr_proj_image_optical_src1_2 - curr_src_image_stack[:,:,:,3:6])
            reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src1_2 * occu_masks_fw[3][s]) / occu_masks_fw_avg[3][s]

            curr_proj_image_optical_src1_3 = transformer_old(curr_src_image_stack[:,:,:,9:12], self.pred_fw_flows[8][s], [H, W])
            curr_proj_error_optical_src1_3 = tf.abs(curr_proj_image_optical_src1_3 - curr_src_image_stack[:,:,:,3:6])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src1_3 * occu_masks_bw[8][s]) / occu_masks_bw_avg[8][s]

            # tgt
            curr_proj_image_optical_tgt = transformer_old(curr_src_image_stack[:,:,:,3:6], self.pred_bw_flows[0][s], [H, W])
            curr_proj_error_optical_tgt = tf.abs(curr_proj_image_optical_tgt - curr_tgt_image)
            reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_tgt * occu_masks_fw[0][s]) / occu_masks_fw_avg[0][s]

            curr_proj_image_optical_tgt_1 = transformer_old(curr_src_image_stack[:,:,:,6:9], self.pred_fw_flows[1][s], [H, W])
            curr_proj_error_optical_tgt_1 = tf.abs(curr_proj_image_optical_tgt_1 - curr_tgt_image)
            reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_tgt_1 * occu_masks_bw[1][s]) / occu_masks_bw_avg[1][s]

            curr_proj_image_optical_tgt_2 = transformer_old(curr_src_image_stack[:,:,:,0:3], self.pred_bw_flows[5][s], [H, W])
            curr_proj_error_optical_tgt_2 = tf.abs(curr_proj_image_optical_tgt_2 - curr_tgt_image)
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_tgt_2 * occu_masks_fw[5][s]) / occu_masks_fw_avg[5][s]

            curr_proj_image_optical_tgt_3 = transformer_old(curr_src_image_stack[:,:,:,9:12], self.pred_fw_flows[6][s], [H, W])
            curr_proj_error_optical_tgt_3 = tf.abs(curr_proj_image_optical_tgt_3 - curr_tgt_image)
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_tgt_3 * occu_masks_bw[6][s]) / occu_masks_bw_avg[6][s]

            # src2
            curr_proj_image_optical_src2 = transformer_old(curr_tgt_image, self.pred_bw_flows[1][s], [H, W])
            curr_proj_error_optical_src2 = tf.abs(curr_proj_image_optical_src2 - curr_src_image_stack[:,:,:,6:9])
            reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src2 * occu_masks_fw[1][s]) / occu_masks_fw_avg[1][s]

            curr_proj_image_optical_src2_1 = transformer_old(curr_src_image_stack[:,:,:,9:12], self.pred_fw_flows[4][s], [H, W])
            curr_proj_error_optical_src2_1 = tf.abs(curr_proj_image_optical_src2_1 - curr_src_image_stack[:,:,:,6:9])
            reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src2_1 * occu_masks_bw[4][s]) / occu_masks_bw_avg[4][s]

            curr_proj_image_optical_src2_2 = transformer_old(curr_src_image_stack[:,:,:,3:6], self.pred_bw_flows[2][s], [H, W])
            curr_proj_error_optical_src2_2 = tf.abs(curr_proj_image_optical_src2_2 - curr_src_image_stack[:,:,:,6:9])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src2_2 * occu_masks_fw[2][s]) / occu_masks_fw_avg[2][s]

            curr_proj_image_optical_src2_3 = transformer_old(curr_src_image_stack[:,:,:,0:3], self.pred_bw_flows[7][s], [H, W])
            curr_proj_error_optical_src2_3 = tf.abs(curr_proj_image_optical_src2_3 - curr_src_image_stack[:,:,:,6:9])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src2_3 * occu_masks_fw[7][s]) / occu_masks_fw_avg[7][s]

            # src3
            curr_proj_image_optical_src3 = transformer_old(curr_src_image_stack[:,:,:,6:9], self.pred_bw_flows[4][s], [H, W])
            curr_proj_error_optical_src3 = tf.abs(curr_proj_image_optical_src3 - curr_src_image_stack[:,:,:,9:12])
            reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src3 * occu_masks_fw[4][s]) / occu_masks_fw_avg[4][s]

            curr_proj_image_optical_src3_1 = transformer_old(curr_tgt_image, self.pred_bw_flows[6][s], [H, W])
            curr_proj_error_optical_src3_1 = tf.abs(curr_proj_image_optical_src3_1 - curr_src_image_stack[:,:,:,9:12])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src3_1 * occu_masks_fw[6][s]) / occu_masks_fw_avg[6][s]

            curr_proj_image_optical_src3_2 = transformer_old(curr_src_image_stack[:,:,:,3:6], self.pred_bw_flows[8][s], [H, W])
            curr_proj_error_optical_src3_2 = tf.abs(curr_proj_image_optical_src3_2 - curr_src_image_stack[:,:,:,9:12])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src3_2 * occu_masks_fw[8][s]) / occu_masks_fw_avg[8][s]

            curr_proj_image_optical_src3_3 = transformer_old(curr_src_image_stack[:,:,:,0:3], self.pred_bw_flows[9][s], [H, W])
            curr_proj_error_optical_src3_3 = tf.abs(curr_proj_image_optical_src3_3 - curr_src_image_stack[:,:,:,9:12])
            cross_reconstructed_loss += tf.reduce_mean(
                curr_proj_error_optical_src3_3 * occu_masks_fw[9][s]) / occu_masks_fw_avg[9][s]

            if self.ssim_weight > 0:
                # src0
                ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src0 * occu_masks_bw[3][s],
                         curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[3][s])) / occu_masks_bw_avg[3][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src0_1 * occu_masks_bw[5][s],
                         curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[5][s])) / occu_masks_bw_avg[5][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src0_2 * occu_masks_bw[7][s],
                         curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[7][s])) / occu_masks_bw_avg[7][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src0_3 * occu_masks_bw[9][s],
                         curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[9][s])) / occu_masks_bw_avg[9][s]

                # src1
                ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src1 * occu_masks_bw[0][s],
                         curr_src_image_stack[:,:,:,3:6] * occu_masks_bw[0][s])) / occu_masks_bw_avg[0][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src1_1 * occu_masks_bw[2][s],
                         curr_src_image_stack[:,:,:,3:6] * occu_masks_bw[2][s])) / occu_masks_bw_avg[2][s]

                ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src1_2 * occu_masks_fw[3][s],
                         curr_src_image_stack[:,:,:,3:6] * occu_masks_fw[3][s])) / occu_masks_fw_avg[3][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src1_3 * occu_masks_bw[8][s],
                         curr_src_image_stack[:,:,:,3:6] * occu_masks_bw[8][s])) / occu_masks_bw_avg[8][s]

                # tgt
                ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_tgt * occu_masks_fw[0][s],
                         curr_tgt_image * occu_masks_fw[0][s])) / occu_masks_fw_avg[0][s]

                ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_tgt_1 * occu_masks_bw[1][s],
                         curr_tgt_image * occu_masks_bw[1][s])) / occu_masks_bw_avg[1][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_tgt_2 * occu_masks_fw[5][s],
                         curr_tgt_image * occu_masks_fw[5][s])) / occu_masks_fw_avg[5][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_tgt_3 * occu_masks_bw[6][s],
                         curr_tgt_image * occu_masks_bw[6][s])) / occu_masks_bw_avg[6][s]

                # src2
                ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src2 * occu_masks_fw[1][s],
                         curr_src_image_stack[:,:,:,6:9] * occu_masks_fw[1][s])) / occu_masks_fw_avg[1][s]

                ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src2_1 * occu_masks_bw[4][s],
                         curr_src_image_stack[:,:,:,6:9] * occu_masks_bw[4][s])) / occu_masks_bw_avg[4][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src2_2 * occu_masks_fw[2][s],
                         curr_src_image_stack[:,:,:,6:9] * occu_masks_fw[2][s])) / occu_masks_fw_avg[2][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src2_3 * occu_masks_fw[7][s],
                         curr_src_image_stack[:,:,:,6:9] * occu_masks_fw[7][s])) / occu_masks_fw_avg[7][s]

                # src3
                ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src3 * occu_masks_fw[4][s],
                         curr_src_image_stack[:,:,:,9:12] * occu_masks_fw[4][s])) / occu_masks_fw_avg[4][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src3_1 * occu_masks_fw[6][s],
                         curr_src_image_stack[:,:,:,9:12] * occu_masks_fw[6][s])) / occu_masks_fw_avg[6][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src3_2 * occu_masks_fw[8][s],
                         curr_src_image_stack[:,:,:,9:12] * occu_masks_fw[8][s])) / occu_masks_fw_avg[8][s]

                cross_ssim_loss += tf.reduce_mean(
                    SSIM(curr_proj_image_optical_src3_3 * occu_masks_fw[9][s],
                         curr_src_image_stack[:,:,:,9:12] * occu_masks_fw[9][s])) / occu_masks_fw_avg[9][s]

            # Compute second-order derivatives for flow smoothness loss
            flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[0][s] / 20.0, curr_src_image_stack[:,:,:,3:6], 1.0)

            flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[1][s] / 20.0, curr_tgt_image, 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[2][s] / 20.0, curr_src_image_stack[:,:,:,3:6], 1.0)

            flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[3][s] / 20.0, curr_src_image_stack[:,:,:,0:3], 1.0)

            flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[4][s] / 20.0, curr_src_image_stack[:,:,:,6:9], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[5][s] / 20.0, curr_src_image_stack[:,:,:,0:3], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[6][s] / 20.0, curr_tgt_image, 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[7][s] / 20.0, curr_src_image_stack[:,:,:,0:3], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[8][s] / 20.0, curr_src_image_stack[:,:,:,3:6], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_fw_flows[9][s] / 20.0, curr_src_image_stack[:,:,:,0:3], 1.0)

            flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[0][s] / 20.0, curr_tgt_image, 1.0)

            flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[1][s] / 20.0, curr_src_image_stack[:,:,:,6:9], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[2][s] / 20.0, curr_src_image_stack[:,:,:,6:9], 1.0)

            flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[3][s] / 20.0, curr_src_image_stack[:,:,:,3:6], 1.0)

            flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[4][s] / 20.0, curr_src_image_stack[:,:,:,9:12], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[5][s] / 20.0, curr_tgt_image, 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[6][s] / 20.0, curr_src_image_stack[:,:,:,9:12], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[7][s] / 20.0, curr_src_image_stack[:,:,:,6:9], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[8][s] / 20.0, curr_src_image_stack[:,:,:,9:12], 1.0)

            cross_flow_smooth_loss += cal_grad2_error(
                self.pred_bw_flows[9][s] / 20.0, curr_src_image_stack[:,:,:,9:12], 1.0)

            # [TODO] Add first-order derivatives for flow smoothness loss
            # [TODO] use robust Charbonnier penalty?

            # if s == 0:
            #     occlusion_map_0_all = occu_masks_0[s]
            #     occlusion_map_1_all = occu_masks_1[s]
            #     occlusion_map_2_all = occu_masks_2[s]
            #     occlusion_map_3_all = occu_masks_3[s]
            #     occlusion_map_4_all = occu_masks_4[s]
            #     occlusion_map_5_all = occu_masks_5[s]

        self.losses = self.flow_reconstruction_weight * ((1.0 - self.ssim_weight) * \
                      (reconstructed_loss + self.flow_cross_geometry_weight*cross_reconstructed_loss) + self.ssim_weight*(ssim_loss+self.flow_cross_geometry_weight*cross_ssim_loss)) + \
                      self.flow_smooth_weight * (flow_smooth_loss + self.flow_cross_geometry_weight*cross_flow_smooth_loss)

        summaries = []
        summaries.append(tf.summary.scalar("total_loss", self.losses))
        summaries.append(tf.summary.scalar("reconstructed_loss", reconstructed_loss))
        summaries.append(tf.summary.scalar("cross_reconstructed_loss", cross_reconstructed_loss))
        summaries.append(tf.summary.scalar("ssim_loss", ssim_loss))
        summaries.append(tf.summary.scalar("cross_ssim_loss", cross_ssim_loss))
        summaries.append(tf.summary.scalar("flow_smooth_loss", flow_smooth_loss))
        summaries.append(tf.summary.scalar("cross_flow_smooth_loss", cross_flow_smooth_loss))

        s = 0
        tf.summary.image('scale%d_target_image' % s, tf.image.convert_image_dtype(curr_tgt_image_all[0], dtype=tf.uint8))

        for i in range(self.num_source):
            tf.summary.image('scale%d_src_image_%d' % (s, i), tf.image.convert_image_dtype(curr_src_image_stack_all[0][:, :, :, i*3:(i+1)*3], dtype=tf.uint8))

        tf.summary.image('scale%d_flow_src12tgt' % s, fl.flow_to_color(self.pred_fw_flows[0][s], max_flow=256))
        tf.summary.image('scale%d_flow_tgt2src2' % s, fl.flow_to_color(self.pred_fw_flows[1][s], max_flow=256))
        tf.summary.image('scale%d_flow_src12src2' % s, fl.flow_to_color(self.pred_fw_flows[2][s], max_flow=256))
        tf.summary.image('scale%d_flow_src02src1' % s, fl.flow_to_color(self.pred_fw_flows[3][s], max_flow=256))
        tf.summary.image('scale%d_flow_src22src3' % s, fl.flow_to_color(self.pred_fw_flows[4][s], max_flow=256))
        tf.summary.image('scale%d_flow_src02tgt' % s, fl.flow_to_color(self.pred_fw_flows[5][s], max_flow=256))
        tf.summary.image('scale%d_flow_tgt2src3' % s, fl.flow_to_color(self.pred_fw_flows[6][s], max_flow=256))

        # tf.summary.image('scale%d_flow_tgt2src0', fl.flow_to_color(self.pred_bw_flows[0][s], max_flow=256))
        # tf.summary.image('scale%d_flow_src12tgt', fl.flow_to_color(self.pred_bw_flows[1][s], max_flow=256))
        # tf.summary.image('scale%d_flow_src12src0', fl.flow_to_color(self.pred_bw_flows[2][s], max_flow=256))

        # tf.summary.image('scale_flyout_mask_src0', occlusion_map_0_all)
        # tf.summary.image('scale_flyout_mask_tgt', occlusion_map_1_all)
        # tf.summary.image('scale_flyout_mask_src0_1', occlusion_map_2_all)
        # tf.summary.image('scale_flyout_mask_tgt1', occlusion_map_3_all)
        # tf.summary.image('scale_flyout_mask_src1', occlusion_map_4_all)
        # tf.summary.image('scale_flyout_mask_src1_1', occlusion_map_5_all)

        self.summ_op = tf.summary.merge(summaries)

    def build_dp_loss_mask_allpose(self):
        smooth_loss = 0
        reconstructed_loss = 0
        cross_reconstructed_loss = 0
        ssim_loss = 0
        cross_ssim_loss = 0

        proj_error_depth_all = []
        flyout_map_all_tgt = []
        flyout_map_all_src0 = []
        flyout_map_all_src1 = []
        flyout_map_all_src2 = []
        flyout_map_all_src3 = []
        curr_tgt_image_all = []
        curr_src_image_stack_all = []
        proj_error_src0 = []
        proj_error_src0_1 = []
        proj_error_src1 = []
        proj_error_src1_1 = []
        proj_error_tgt = []
        proj_error_tgt1 = []
        upsampled_tgt_depth_all = []
        summaries = []

        """ Generating occlusion map from FlowNet
        Calculate different scale occulsion maps described in 'Occlusion Aware Unsupervised
        Learning of Optical Flow by Yang Wang et al'
        """
        occu_masks_bw = []
        occu_masks_bw_avg = []
        occu_masks_fw = []
        occu_masks_fw_avg = []

        for i in range(len(self.pred_bw_flows)):
            temp_occu_masks_bw = []
            temp_occu_masks_bw_avg = []
            temp_occu_masks_fw = []
            temp_occu_masks_fw_avg = []

            for s in range(self.num_scales):
                H = int(self.img_height / (2**s))
                W = int(self.img_width  / (2**s))

                mask, mask_avg = self.occulsion(self.pred_bw_flows[i][s], H, W)
                temp_occu_masks_bw.append(mask)
                temp_occu_masks_bw_avg.append(mask_avg)
                # [src1, tgt, src1_1, src0, src2, src0_1, tgt_1, src0_2, src1_2, src0_3]

                mask, mask_avg = self.occulsion(self.pred_fw_flows[i][s], H, W)
                temp_occu_masks_fw.append(mask)
                temp_occu_masks_fw_avg.append(mask_avg)
                # [tgt, src2, src2_1, src1, src3, tgt_1, src3_1, src2_2, src3_2, src3_3]

            occu_masks_bw.append(temp_occu_masks_bw)
            occu_masks_bw_avg.append(temp_occu_masks_bw_avg)
            occu_masks_fw.append(temp_occu_masks_fw)
            occu_masks_fw_avg.append(temp_occu_masks_fw_avg)

        self.scaled_tgt_images = [None for _ in range(self.num_scales)]
        self.scaled_src_images_stack = [None for _ in range(self.num_scales)]
        for s in range(self.num_scales):
            if s == 0: # Just as a precaution. TF often has interpolation bugs.
                self.scaled_tgt_images[s] = self.tgt_image
                self.scaled_src_images_stack[s] = self.src_image_stack
            else:
                self.scaled_tgt_images[s] = tf.image.resize_bilinear(
                    self.tgt_image, [int(self.img_height/(2**s)), int(self.img_width/(2**s))], align_corners=True)
                self.scaled_src_images_stack[s] = tf.image.resize_bilinear(
                    self.src_image_stack, [int(self.img_height/(2**s)), int(self.img_width/(2**s))], align_corners=True)

            selected_scale = 0 if self.is_depth_upsampling else s
            H = int(self.img_height / (2**selected_scale))
            W = int(self.img_width  / (2**selected_scale))
            curr_tgt_image = self.scaled_tgt_images[selected_scale]
            curr_src_image_stack = self.scaled_src_images_stack[selected_scale]

            curr_tgt_image_all.append(curr_tgt_image)
            curr_src_image_stack_all.append(curr_src_image_stack)

            if self.is_depth_upsampling:
                tgt_depth = self.depth_upsampled['tgt'][s]
                src0_depth = self.depth_upsampled['src0'][s]
                src1_depth = self.depth_upsampled['src1'][s]
                src2_depth = self.depth_upsampled['src2'][s]
                src3_depth = self.depth_upsampled['src3'][s]
            else:
                tgt_depth = self.depth['tgt'][s]
                src0_depth = self.depth['src0'][s]
                src1_depth = self.depth['src1'][s]
                src2_depth = self.depth['src2'][s]
                src3_depth = self.depth['src3'][s]

            # src0
            depth_flow_src02src1, _ = inverse_warp(
                src0_depth,
                self.pred_poses[:, 0, 0:6], # src0 -> src1 (fw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src12src0 = transformer_old(curr_src_image_stack[:,:,:,3:6], depth_flow_src02src1, [H, W])
            curr_proj_error_src0 = tf.abs(curr_proj_image_src12src0 - curr_src_image_stack[:,:,:,0:3])

            depth_flow_src02tgt, _ = inverse_warp(
                src0_depth,
                self.pred_poses[:, 0, 6:12], # src0 -> tgt (fw2)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_tgt2src0 = transformer_old(curr_tgt_image, depth_flow_src02tgt, [H, W])
            curr_proj_error_src0_1 = tf.abs(curr_proj_image_tgt2src0 - curr_src_image_stack[:,:,:,0:3])

            depth_flow_src02src2, _ = inverse_warp(
                src0_depth,
                self.pred_poses[:, 0, 12:18], # src0 -> src1 (fw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src22src0 = transformer_old(curr_src_image_stack[:,:,:,6:9], depth_flow_src02src2, [H, W])
            curr_proj_error_src0_2 = tf.abs(curr_proj_image_src22src0 - curr_src_image_stack[:,:,:,0:3])

            depth_flow_src02src3, _ = inverse_warp(
                src0_depth,
                self.pred_poses[:, 0, 18:24], # src0 -> src1 (fw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src32src0 = transformer_old(curr_src_image_stack[:,:,:,9:12], depth_flow_src02src3, [H, W])
            curr_proj_error_src0_3 = tf.abs(curr_proj_image_src32src0 - curr_src_image_stack[:,:,:,0:3])

            # src1
            depth_flow_src12tgt, _ = inverse_warp(
                src1_depth,
                self.pred_poses[:, 0, 24:30], # src1 -> tgt (fw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_tgt2src1 = transformer_old(curr_tgt_image, depth_flow_src12tgt, [H, W])
            curr_proj_error_src1 = tf.abs(curr_proj_image_tgt2src1 - curr_src_image_stack[:,:,:,3:6])

            depth_flow_src12src2, _ = inverse_warp(
                src1_depth,
                self.pred_poses[:, 0, 30:36], # src1 -> src2 (fw2)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src22src1 = transformer_old(curr_src_image_stack[:,:,:,6:9], depth_flow_src12src2, [H, W])
            curr_proj_error_src1_1 = tf.abs(curr_proj_image_src22src1 - curr_src_image_stack[:,:,:,3:6])

            depth_flow_src12src0, _ = inverse_warp(
                src1_depth,
                self.pred_poses[:, 0, 36:42], # src1 -> src0 (bw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src02src1 = transformer_old(curr_src_image_stack[:,:,:,0:3], depth_flow_src12src0, [H, W])
            curr_proj_error_src1_2 = tf.abs(curr_proj_image_src02src1 - curr_src_image_stack[:,:,:,3:6])

            depth_flow_src12src3, _ = inverse_warp(
                src1_depth,
                self.pred_poses[:, 0, 42:48], # src1 -> src3 (fw3)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src32src1 = transformer_old(curr_src_image_stack[:,:,:,9:12], depth_flow_src12src3, [H, W])
            curr_proj_error_src1_3 = tf.abs(curr_proj_image_src32src1 - curr_src_image_stack[:,:,:,3:6])

            # tgt
            depth_flow_tgt2src2, _ = inverse_warp(
                tgt_depth,
                self.pred_poses[:, 0, 48:54], # tgt -> src2 (fw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src22tgt = transformer_old(curr_src_image_stack[:,:,:,6:9], depth_flow_tgt2src2, [H, W])
            curr_proj_error_tgt = tf.abs(curr_proj_image_src22tgt - curr_tgt_image)

            depth_flow_tgt2src3, _ = inverse_warp(
                tgt_depth,
                self.pred_poses[:, 0, 54:60], # tgt -> src3 (fw2)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src32tgt = transformer_old(curr_src_image_stack[:,:,:,9:12], depth_flow_tgt2src3, [H, W])
            curr_proj_error_tgt_1 = tf.abs(curr_proj_image_src32tgt - curr_tgt_image)

            depth_flow_tgt2src1, _ = inverse_warp(
                tgt_depth,
                self.pred_poses[:, 0, 60:66], # tgt -> src1 (bw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src12tgt = transformer_old(curr_src_image_stack[:,:,:,3:6], depth_flow_tgt2src1, [H, W])
            curr_proj_error_tgt_2 = tf.abs(curr_proj_image_src12tgt - curr_tgt_image)

            depth_flow_tgt2src0, _ = inverse_warp(
                tgt_depth,
                self.pred_poses[:, 0, 66:72], # tgt -> src0 (bw2)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src02tgt = transformer_old(curr_src_image_stack[:,:,:,0:3], depth_flow_tgt2src0, [H, W])
            curr_proj_error_tgt_3 = tf.abs(curr_proj_image_src02tgt - curr_tgt_image)

            # src2
            depth_flow_src22src3, _ = inverse_warp(
                src2_depth,
                self.pred_poses[:, 0, 72:78], # src2 -> src3 (fw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src32src2 = transformer_old(curr_src_image_stack[:,:,:,9:12], depth_flow_src22src3, [H, W])
            curr_proj_error_src2 = tf.abs(curr_proj_image_src32src2 - curr_src_image_stack[:,:,:,6:9])

            depth_flow_src22tgt, _ = inverse_warp(
                src2_depth,
                self.pred_poses[:, 0, 78:84], # src2 -> tgt (bw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_tgt2src2 = transformer_old(curr_tgt_image, depth_flow_src22tgt, [H, W])
            curr_proj_error_src2_1 = tf.abs(curr_proj_image_tgt2src2 - curr_src_image_stack[:,:,:,6:9])

            depth_flow_src22src1, _ = inverse_warp(
                src2_depth,
                self.pred_poses[:, 0, 84:90], # src2 -> src1 (bw2)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src12src2 = transformer_old(curr_src_image_stack[:,:,:,3:6], depth_flow_src22src1, [H, W])
            curr_proj_error_src2_2 = tf.abs(curr_proj_image_src12src2 - curr_src_image_stack[:,:,:,6:9])

            depth_flow_src22src0, _ = inverse_warp(
                src2_depth,
                self.pred_poses[:, 0, 90:96], # src2 -> src0 (bw3)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src02src2 = transformer_old(curr_src_image_stack[:,:,:,0:3], depth_flow_src22src0, [H, W])
            curr_proj_error_src2_3 = tf.abs(curr_proj_image_src02src2 - curr_src_image_stack[:,:,:,6:9])

            # src3
            depth_flow_src32src2, _ = inverse_warp(
                src3_depth,
                self.pred_poses[:, 0, 96:102], # src3 -> src2 (bw1)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src22src3 = transformer_old(curr_src_image_stack[:,:,:,6:9], depth_flow_src32src2, [H, W])
            curr_proj_error_src3 = tf.abs(curr_proj_image_src22src3 - curr_src_image_stack[:,:,:,9:12])

            depth_flow_src32tgt, _ = inverse_warp(
                src3_depth,
                self.pred_poses[:, 0, 102:108], # src3 -> tgt (bw2)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_tgt2src3 = transformer_old(curr_tgt_image, depth_flow_src32tgt, [H, W])
            curr_proj_error_src3_1 = tf.abs(curr_proj_image_tgt2src3 - curr_src_image_stack[:,:,:,9:12])

            depth_flow_src32src1, _ = inverse_warp(
                src3_depth,
                self.pred_poses[:, 0, 108:114], # src3 -> src1 (bw3)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src12src3 = transformer_old(curr_src_image_stack[:,:,:,3:6], depth_flow_src32src1, [H, W])
            curr_proj_error_src3_2 = tf.abs(curr_proj_image_src12src3 - curr_src_image_stack[:,:,:,9:12])

            depth_flow_src32src0, _ = inverse_warp(
                src3_depth,
                self.pred_poses[:, 0, 114:120], # src3 -> src0 (bw4)
                self.proj_cam2pix[:, selected_scale, :, :],
                self.proj_pix2cam[:, selected_scale, :, :])
            curr_proj_image_src02src3 = transformer_old(curr_src_image_stack[:,:,:,0:3], depth_flow_src32src0, [H, W])
            curr_proj_error_src3_3 = tf.abs(curr_proj_image_src02src3 - curr_src_image_stack[:,:,:,9:12])

            if not self.compute_minimum_loss:
                # src0
                reconstructed_loss +=       tf.reduce_mean(curr_proj_error_src0 * occu_masks_bw[3][selected_scale]) / occu_masks_bw_avg[3][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src0_1 * occu_masks_bw[5][selected_scale]) / occu_masks_bw_avg[5][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src0_2 * occu_masks_bw[7][selected_scale]) / occu_masks_bw_avg[7][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src0_3 * occu_masks_bw[9][selected_scale]) / occu_masks_bw_avg[9][selected_scale]
                # src1
                reconstructed_loss +=       tf.reduce_mean(curr_proj_error_src1 * occu_masks_bw[0][selected_scale]) / occu_masks_bw_avg[0][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src1_1 * occu_masks_bw[2][selected_scale]) / occu_masks_bw_avg[2][selected_scale]
                reconstructed_loss +=       tf.reduce_mean(curr_proj_error_src1_2 * occu_masks_fw[3][selected_scale]) / occu_masks_fw_avg[3][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src1_3 * occu_masks_bw[8][selected_scale]) / occu_masks_bw_avg[8][selected_scale]
                # tgt
                reconstructed_loss +=       tf.reduce_mean(curr_proj_error_tgt * occu_masks_bw[1][selected_scale]) / occu_masks_bw_avg[1][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_tgt_1 * occu_masks_bw[6][selected_scale]) / occu_masks_bw_avg[6][selected_scale]
                reconstructed_loss +=       tf.reduce_mean(curr_proj_error_tgt_2 * occu_masks_fw[0][selected_scale]) / occu_masks_fw_avg[0][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_tgt_3 * occu_masks_fw[5][selected_scale]) / occu_masks_fw_avg[5][selected_scale]
                # src2
                reconstructed_loss +=       tf.reduce_mean(curr_proj_error_src2 * occu_masks_bw[4][selected_scale]) / occu_masks_bw_avg[4][selected_scale]
                reconstructed_loss +=       tf.reduce_mean(curr_proj_error_src2_1 * occu_masks_fw[1][selected_scale]) / occu_masks_fw_avg[1][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src2_2 * occu_masks_fw[2][selected_scale]) / occu_masks_fw_avg[2][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src2_3 * occu_masks_fw[7][selected_scale]) / occu_masks_fw_avg[7][selected_scale]
                # src3
                reconstructed_loss +=       tf.reduce_mean(curr_proj_error_src3 * occu_masks_fw[4][selected_scale]) / occu_masks_fw_avg[4][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src3_1 * occu_masks_fw[6][selected_scale]) / occu_masks_fw_avg[6][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src3_2 * occu_masks_fw[8][selected_scale]) / occu_masks_fw_avg[8][selected_scale]
                cross_reconstructed_loss += tf.reduce_mean(curr_proj_error_src3_3 * occu_masks_fw[9][selected_scale]) / occu_masks_fw_avg[9][selected_scale]

                if self.ssim_weight > 0:
                    # src0
                    ssim_loss +=       tf.reduce_mean(SSIM(curr_proj_image_src12src0 * occu_masks_bw[3][selected_scale], curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[3][selected_scale])) / occu_masks_bw_avg[3][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_tgt2src0 * occu_masks_bw[5][selected_scale], curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[5][selected_scale])) / occu_masks_bw_avg[5][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src22src0 * occu_masks_bw[7][selected_scale], curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[7][selected_scale])) / occu_masks_bw_avg[7][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src32src0 * occu_masks_bw[9][selected_scale], curr_src_image_stack[:,:,:,0:3] * occu_masks_bw[9][selected_scale])) / occu_masks_bw_avg[9][selected_scale]
                    # src1
                    ssim_loss +=       tf.reduce_mean(SSIM(curr_proj_image_tgt2src1 * occu_masks_bw[0][selected_scale], curr_src_image_stack[:,:,:,3:6] * occu_masks_bw[0][selected_scale])) / occu_masks_bw_avg[0][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src22src1 * occu_masks_bw[2][selected_scale], curr_src_image_stack[:,:,:,3:6] * occu_masks_bw[2][selected_scale])) / occu_masks_bw_avg[2][selected_scale]
                    ssim_loss +=       tf.reduce_mean(SSIM(curr_proj_image_src02src1 * occu_masks_fw[3][selected_scale], curr_src_image_stack[:,:,:,3:6] * occu_masks_fw[3][selected_scale])) / occu_masks_fw_avg[3][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src32src1 * occu_masks_bw[8][selected_scale], curr_src_image_stack[:,:,:,3:6] * occu_masks_bw[8][selected_scale])) / occu_masks_bw_avg[8][selected_scale]
                    # tgt
                    ssim_loss +=       tf.reduce_mean(SSIM(curr_proj_image_src22tgt * occu_masks_bw[1][selected_scale], curr_tgt_image * occu_masks_bw[1][selected_scale])) / occu_masks_bw_avg[1][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src32tgt * occu_masks_bw[6][selected_scale], curr_tgt_image * occu_masks_bw[6][selected_scale])) / occu_masks_bw_avg[6][selected_scale]
                    ssim_loss +=       tf.reduce_mean(SSIM(curr_proj_image_src12tgt * occu_masks_fw[0][selected_scale], curr_tgt_image * occu_masks_fw[0][selected_scale])) / occu_masks_fw_avg[0][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src02tgt * occu_masks_fw[5][selected_scale], curr_tgt_image * occu_masks_fw[5][selected_scale])) / occu_masks_fw_avg[5][selected_scale]
                    # src2
                    ssim_loss +=       tf.reduce_mean(SSIM(curr_proj_image_src32src2 * occu_masks_bw[4][selected_scale], curr_src_image_stack[:,:,:,6:9] * occu_masks_bw[4][selected_scale])) / occu_masks_bw_avg[4][selected_scale]
                    ssim_loss +=       tf.reduce_mean(SSIM(curr_proj_image_tgt2src2 * occu_masks_fw[1][selected_scale], curr_src_image_stack[:,:,:,6:9] * occu_masks_fw[1][selected_scale])) / occu_masks_fw_avg[1][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src12src2 * occu_masks_fw[2][selected_scale], curr_src_image_stack[:,:,:,6:9] * occu_masks_fw[2][selected_scale])) / occu_masks_fw_avg[2][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src02src2 * occu_masks_fw[7][selected_scale], curr_src_image_stack[:,:,:,6:9] * occu_masks_fw[7][selected_scale])) / occu_masks_fw_avg[7][selected_scale]
                    # src3
                    ssim_loss +=       tf.reduce_mean(SSIM(curr_proj_image_src22src3 * occu_masks_fw[4][selected_scale], curr_src_image_stack[:,:,:,9:12] * occu_masks_fw[4][selected_scale])) / occu_masks_fw_avg[4][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_tgt2src3 * occu_masks_fw[6][selected_scale], curr_src_image_stack[:,:,:,9:12] * occu_masks_fw[6][selected_scale])) / occu_masks_fw_avg[6][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src12src3 * occu_masks_fw[8][selected_scale], curr_src_image_stack[:,:,:,9:12] * occu_masks_fw[8][selected_scale])) / occu_masks_fw_avg[8][selected_scale]
                    cross_ssim_loss += tf.reduce_mean(SSIM(curr_proj_image_src02src3 * occu_masks_fw[9][selected_scale], curr_src_image_stack[:,:,:,9:12] * occu_masks_fw[9][selected_scale])) / occu_masks_fw_avg[9][selected_scale]

            if self.dp_smooth_weight > 0:
                if self.depth_normalization:
                    # Perform depth normalization, dividing by the mean.
                    mean_tgt_disp = tf.reduce_mean(self.disp['tgt'][s], axis=[1, 2, 3], keepdims=True)
                    tgt_disp_input = self.disp['tgt'][s] / mean_tgt_disp
                    mean_src0_disp = tf.reduce_mean(self.disp['src0'][s], axis=[1, 2, 3], keepdims=True)
                    src0_disp_input = self.disp['src0'][s] / mean_src0_disp
                    mean_src1_disp = tf.reduce_mean(self.disp['src1'][s], axis=[1, 2, 3], keepdims=True)
                    src1_disp_input = self.disp['src1'][s] / mean_src1_disp
                    mean_src2_disp = tf.reduce_mean(self.disp['src2'][s], axis=[1, 2, 3], keepdims=True)
                    src2_disp_input = self.disp['src2'][s] / mean_src2_disp
                    mean_src3_disp = tf.reduce_mean(self.disp['src3'][s], axis=[1, 2, 3], keepdims=True)
                    src3_disp_input = self.disp['src3'][s] / mean_src3_disp
                else:
                    tgt_disp_input = self.disp['tgt'][s]
                    src0_disp_input = self.disp['src0'][s]
                    src1_disp_input = self.disp['src1'][s]
                    src2_disp_input = self.disp['src2'][s]
                    src3_disp_input = self.disp['src3'][s]

                scaling_f = (1.0 if self.equal_weighting else 1.0 / (2**s))
                # Edge-aware first-order
                smooth_loss += scaling_f * depth_smoothness(tgt_disp_input, self.scaled_tgt_images[s])
                smooth_loss += scaling_f * depth_smoothness(src0_disp_input, self.scaled_src_images_stack[s][:,:,:,0:3])
                smooth_loss += scaling_f * depth_smoothness(src1_disp_input, self.scaled_src_images_stack[s][:,:,:,3:6])
                smooth_loss += scaling_f * depth_smoothness(src2_disp_input, self.scaled_src_images_stack[s][:,:,:,6:9])
                smooth_loss += scaling_f * depth_smoothness(src3_disp_input, self.scaled_src_images_stack[s][:,:,:,9:12])

            if s == 0:
                if self.compute_minimum_loss:
                    flyout_map_all_tgt.append(min_mask_tgt)
                    flyout_map_all_src0.append(min_mask_src0)
                    flyout_map_all_src1.append(min_mask_src1)
                else:
                    flyout_map_all_tgt.append(occu_masks_bw[1][selected_scale])
                    flyout_map_all_src0.append(occu_masks_bw[3][selected_scale])
                    flyout_map_all_src1.append(occu_masks_bw[0][selected_scale])
                    flyout_map_all_src2.append(occu_masks_bw[4][selected_scale])
                    flyout_map_all_src3.append(occu_masks_fw[4][selected_scale])

            upsampled_tgt_depth_all.append(tgt_depth)

        # self.losses = (self.pixel_loss_weight * pixel_loss_depth + self.smooth_weight * dp_smooth_loss)
        self.losses = self.dp_reconstruction_weight*((1.0 - self.ssim_weight)*(reconstructed_loss + self.dp_cross_geometry_weight*cross_reconstructed_loss) + self.ssim_weight*(ssim_loss+self.dp_cross_geometry_weight*cross_ssim_loss)) + \
                      self.dp_smooth_weight * smooth_loss

        summaries.append(tf.summary.scalar("total_loss", self.losses))
        summaries.append(tf.summary.scalar("reconstruction_loss", reconstructed_loss))
        summaries.append(tf.summary.scalar("cross_reconstruction_loss", cross_reconstructed_loss))
        summaries.append(tf.summary.scalar("ssim_loss", ssim_loss))
        summaries.append(tf.summary.scalar("cross_ssim_loss", cross_ssim_loss))
        summaries.append(tf.summary.scalar("dp_smooth_loss", smooth_loss))

        s = 0
        tf.summary.image('scale%d_target_image' % s, tf.image.convert_image_dtype(curr_tgt_image_all[0], dtype=tf.uint8))

        for i in range(self.num_source):
            tf.summary.image('scale%d_src_image_%d' % (s, i), tf.image.convert_image_dtype(curr_src_image_stack_all[0][:, :, :, i*3:(i+1)*3], dtype=tf.uint8))

        tf.summary.image('scale%d_tgt_pred_disp' % s, self.disp['tgt'][s])
        tf.summary.image('scale%d_src0_pred_disp' % s, self.disp['src0'][s])
        tf.summary.image('scale%d_src1_pred_disp' % s, self.disp['src1'][s])
        tf.summary.image('scale%d_src2_pred_disp' % s, self.disp['src2'][s])
        tf.summary.image('scale%d_src3_pred_disp' % s, self.disp['src3'][s])

        tf.summary.image('scale%d_flow_src12tgt' % s, fl.flow_to_color(self.pred_fw_flows[0][s], max_flow=256))
        tf.summary.image('scale%d_flow_tgt2src2' % s, fl.flow_to_color(self.pred_fw_flows[1][s], max_flow=256))
        tf.summary.image('scale%d_flow_src12src2' % s, fl.flow_to_color(self.pred_fw_flows[2][s], max_flow=256))
        tf.summary.image('scale%d_flow_src02src1' % s, fl.flow_to_color(self.pred_fw_flows[3][s], max_flow=256))
        tf.summary.image('scale%d_flow_src22src3' % s, fl.flow_to_color(self.pred_fw_flows[4][s], max_flow=256))
        tf.summary.image('scale%d_flow_src02tgt' % s, fl.flow_to_color(self.pred_fw_flows[5][s], max_flow=256))
        tf.summary.image('scale%d_flow_tgt2src3' % s, fl.flow_to_color(self.pred_fw_flows[6][s], max_flow=256))

        if self.is_depth_upsampling:
            for k in range(self.num_scales):
                tf.summary.image('scale%d_tgt_upsampled_pred_depth' % k, upsampled_tgt_depth_all[k])

        tf.summary.image('occlusion_tgt', flyout_map_all_tgt[0])
        tf.summary.image('occlusion_src0', flyout_map_all_src0[0])
        tf.summary.image('occlusion_src1', flyout_map_all_src1[0])
        tf.summary.image('occlusion_src2', flyout_map_all_src2[0])
        tf.summary.image('occlusion_src3', flyout_map_all_src3[0])

        self.summ_op = tf.summary.merge(summaries)

    def spatial_normalize(self, disp):
        # Credit: https://github.com/yzcjtr/GeoNet/blob/master/geonet_model.py
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keepdims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean

    def occulsion(self, pred_flow, H, W):
        """
        Here, we compute the soft occlusion maps proposed in https://arxiv.org/pdf/1711.05890.pdf

        pred_flow: the estimated forward optical flow
        """
        occu_mask = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[self.batch_size, H, W, 1],
                        dtype='float32'),
                    pred_flow, [H , W]),
                clip_value_min=0.0,
                clip_value_max=1.0)
            ]
        occu_mask = tf.reshape(occu_mask, [self.batch_size, H, W, 1])
        occu_mask_avg = tf.reduce_mean(occu_mask)

        return occu_mask, occu_mask_avg
