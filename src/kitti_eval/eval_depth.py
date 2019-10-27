# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
from __future__ import division
import sys
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import misc
from depth_evaluation_utils import *
from PIL import Image, ImageDraw
import imageio

"""
# How to evaluate the predicted depth maps 
python kitti_eval/eval_depth.py \
       --split=eigen \
       --kitti_dir=/home/waterman/dataset/KITTI/ \
       --pred_file=../results/kitti/test_kitti.npy \
       --depth_results=../results/kitti_depths
"""

CMAP = 'plasma'
# CMAP = 'gray'

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default='eigen', help='eigen or stereo split')
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument("--depth_results", type=str, help="Path to save the predicted depth maps")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
args = parser.parse_args()


def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth

def convert_disps_to_depths_stereo(gt_disparities, pred_depths):
    gt_depths = []
    pred_depths_resized = []
    pred_disparities_resized = []

    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(1./pred_depth)

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        #pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths_resized.append(pred_depth)
    return gt_depths, pred_depths_resized, pred_disparities_resized


def main():
    pred_depths = np.load(args.pred_file)
    args.test_file_list = './kitti_eval/kitti/test_files_%s.txt' % args.split

    print('evaluating ' + args.pred_file + '...')

    if args.split == 'eigen':
        test_files = read_text_lines(args.test_file_list)
        assert len(pred_depths) == len(test_files)

        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.kitti_dir)
        num_test = len(im_files)

        depth_path = args.depth_results
        if not os.path.exists(depth_path):
            os.mkdir(depth_path)
        print("[Info] Saving depth resutls to: " + depth_path)

        print('[NUM TEST]: %d' % num_test)

        gt_depths = []
        pred_depths_resized = []
        for t_id in range(num_test):


            camera_id = cams[t_id]  # 2 is left, 3 is right
            # print("[%d] resize to %d %d" % (t_id, im_sizes[t_id][1], im_sizes[t_id][0]))
            pred_depth_resized = cv2.resize(pred_depths[t_id],
                                    (im_sizes[t_id][1], im_sizes[t_id][0]),
                                    interpolation=cv2.INTER_LINEAR)


            pred_depths_resized.append(pred_depth_resized)
            depth = generate_depth_map(gt_calib[t_id],
                                       gt_files[t_id],
                                       im_sizes[t_id],
                                       camera_id,
                                       False,
                                       True)
            gt_depths.append(depth.astype(np.float32))




            filename = '_'.join(test_files[t_id].split('.')[0].split('/'))
            colored_map = normalize_depth_for_display(pred_depth_resized, cmap=CMAP)
            misc.imsave("%s/%s_pred_depth.png" % (depth_path, filename), colored_map)



            # im_file = imageio.imread(im_files[t_id])
            # imageio.imwrite('%s_input/%s.png' % (depth_path, filename), im_file)
        pred_depths = pred_depths_resized
    else:
        num_test = 200
        gt_disparities = load_gt_disp_kitti(args.kitti_dir)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_stereo(gt_disparities, pred_depths)

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    for i in range(num_test):
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])


        if args.split == 'eigen':

            mask = np.logical_and(gt_depth > args.min_depth,
                                  gt_depth < args.max_depth)
            # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            gt_height, gt_width = gt_depth.shape
            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                             0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        if args.split == 'stereo':
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        # Scale matching
        scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])



    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))

main()
