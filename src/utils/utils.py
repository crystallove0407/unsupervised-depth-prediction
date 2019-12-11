import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from absl import logging
from utils.optical_flow_warp_old import transformer_old

def get_shape(x, train=True):
    if train:
        x_shape = x.get_shape().as_list()
    else:
        x_shape = tf.shape(input=x)
    return x_shape

def resize_like(inputs, ref, type='nearest'):
    inputs_shape = get_shape(inputs, train=True)
    iH = inputs_shape[1]
    iW = inputs_shape[2]

    ref_shape = get_shape(ref, train=True)
    rH = ref_shape[1]
    rW = ref_shape[2]

    if iH == rH and iW == rW:
        return inputs
    if type == 'nearest':
        return tf.image.resize(inputs, [rH, rW], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    elif type == 'bilinear':
        return tf.image.resize(inputs, [rH, rW], method=tf.image.ResizeMethod.BILINEAR)
        # [TODO]: find the effect of "align_corners"
        # return tf.image.resize_bilinear(inputs, [rH, rW], align_corners=True)

def lrelu(x, leak=0.2, name='leaky_relu'):
    return tf.maximum(x, leak*x)

def imshow(img, re_normalize=False):
    if re_normalize:
        min_value = np.min(img)
        max_value = np.max(img)
        img = (img - min_value) / (max_value - min_value)
        img = img * 255
    elif np.max(img) <= 1.:
        img = img * 255
    img = img.astype('uint8')
    shape = img.shape
    if len(shape) == 2:
        img = np.repeat(np.expand_dims(img, -1), 3, -1)
    elif shape[2] == 1:
        img = np.repeat(img, 3, -1)
    plt.imshow(img)
    plt.show()

def unpack_image_sequence(image_seq, img_height, img_width, num_source):
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

def rgb_bgr(img):
    tmp = np.copy(img[:, :, 0])
    img[:, :, 0] = np.copy(img[:, :, 2])
    img[:, :, 2] = np.copy(tmp)
    return img

def compute_Fl(flow_gt, flow_est, mask):
    # F1 measure
    err = tf.multiply(flow_gt - flow_est, mask)
    err_norm = tf.norm(tensor=err, axis=-1)

    flow_gt_norm = tf.maximum(tf.norm(tensor=flow_gt, axis=-1), 1e-12)
    F1_logic = tf.logical_and(err_norm > 3, tf.divide(err_norm, flow_gt_norm) > 0.05)
    F1_logic = tf.cast(tf.logical_and(tf.expand_dims(F1_logic, -1), mask > 0), tf.float32)
    F1 = tf.reduce_sum(input_tensor=F1_logic) / (tf.reduce_sum(input_tensor=mask) + 1e-6)
    return F1

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        if grads != []:
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(input_tensor=grad, axis=0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def length_sq(x):
    return tf.reduce_sum(input_tensor=tf.square(x), axis=3, keepdims=True)

# def occlusion(flow_fw, flow_bw):
#     x_shape = tf.shape(flow_fw)
#     H = x_shape[1]
#     W = x_shape[2]
#     flow_bw_warped = tf_warp(flow_bw, flow_fw, H, W)
#     flow_fw_warped = tf_warp(flow_fw, flow_bw, H, W)
#     flow_diff_fw = flow_fw + flow_bw_warped
#     flow_diff_bw = flow_bw + flow_fw_warped
#     mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
#     mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
#     occ_thresh_fw =  0.01 * mag_sq_fw + 0.5
#     occ_thresh_bw =  0.01 * mag_sq_bw + 0.5
#     occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh_fw, tf.float32)
#     occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh_bw, tf.float32)
#
#     return occ_fw, occ_bw, flow_bw_warped, flow_fw_warped

def compute_rigid_flow(depth, pose, intrinsics, reverse_pose=False):
    """Compute the rigid flow from target image plane to source image

    Args:
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source (or source to target if reverse_pose=True)
          camera transformation matrix [batch, 6], in the order of
          tx, ty, tz, rx, ry, rz;
    intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
    Rigid flow from target image to source image [batch, height_t, width_t, 2]
    """
    batch, height, width = depth.get_shape().as_list()
    # Convert pose vector to matrix
    pose = pose_vec2mat(pose)
    if reverse_pose:
        pose = tf.linalg.inv(pose)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    tgt_pixel_coords = tf.transpose(a=pixel_coords[:,:2,:,:], perm=[0, 2, 3, 1])
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    rigid_flow = src_pixel_coords - tgt_pixel_coords
    return rigid_flow

def rgb_bgr(img):
    tmp = np.copy(img[:, :, 0])
    img[:, :, 0] = np.copy(img[:, :, 2])
    img[:, :, 2] = np.copy(tmp)
    return img

# Add inverse_pose flag
def euler2mat(z, y, x, inverse_pose=False):
    """Converts euler angles to rotation matrix
    TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.shape(input=z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)
    if inverse_pose:
        z = -z
        y = -y
        x = -x

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones  = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat


# Add inverse_pose flag
def pose_vec2mat(vec, inverse_pose=False):
    """Converts 6DoF parameters to transformation matrix
    Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
      A transformation matrix -- [B, 4, 4]
    """
    batch_size, _ = vec.get_shape().as_list()
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    if inverse_pose:
        translation = -translation
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx, inverse_pose)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
    Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch, height, width = depth.get_shape().as_list()
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.linalg.inv(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = tf.ones([batch, 1, height*width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords

def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
    Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.get_shape().as_list()
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.transpose(a=pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
    Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                   tf.transpose(a=tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords

# Add inverse_pose flag
def projective_inverse_warp(img, depth, pose, intrinsics, inverse_pose=False):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
        img: the source image [batch, height_s, width_s, 3]
        depth: depth map of the target image [batch, height_t, width_t]
        pose: target to source camera transformation matrix [batch, 6], in the
              order of tx, ty, tz, rx, ry, rz
        intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
        Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    # Convert pose vector to matrix
    pose = pose_vec2mat(pose, inverse_pose)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    output_img = bilinear_sampler(img, src_pixel_coords)
    return output_img, src_pixel_coords

def flow_warp(src_img, flow):
    """ inverse warp a source image to the target image plane based on flow field
    Args:
    src_img: the source  image [batch, height_s, width_s, 3]
    flow: target image to source image flow [batch, height_t, width_t, 2]
    Returns:
    Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
    """
    batch, height, width, _ = src_img.get_shape().as_list()
    tgt_pixel_coords = tf.transpose(a=meshgrid(batch, height, width, False), perm=[0, 2, 3, 1])
    src_pixel_coords = tgt_pixel_coords + flow
    output_img = bilinear_sampler(src_img, src_pixel_coords)
    return output_img


def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
    Returns:
    A new sampled image [batch, height_t, width_t, channels]
    """
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            a=tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), perm=[1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.compat.v1.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(input=imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(input=imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
                [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return output

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

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def make_intrinsics_offset_matrix(fx, fy, cx, cy):
    """
    Without batch
    """
    r1 = tf.stack([fx, 0., cx], axis=0)
    r2 = tf.stack([0., fy, cy], axis=0)
    r3 = tf.constant([0., 0., 0.])
    intrinsics = tf.stack([r1, r2, r3])
    return intrinsics

def make_intrinsics_scale_matrix(fx, fy, cx, cy):
    """
    Without batch
    """
    r1 = tf.stack([fx, 1., cx], axis=0)
    r2 = tf.stack([1., fy, cy], axis=0)
    r3 = tf.constant([1., 1., 1.])
    intrinsics = tf.stack([r1, r2, r3])
    return intrinsics

def make_intrinsics_matrix(fx, fy, cx, cy):
    r1 = tf.stack([fx, 0., cx], axis=0)
    r2 = tf.stack([0., fy, cy], axis=0)
    r3 = tf.constant([0., 0., 1.])
    intrinsics = tf.stack([r1, r2, r3])
    return intrinsics

def make_batch_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics

def get_multi_scale_intrinsics(intrinsics, num_scales):
    intrinsics_mscale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[:,0,0]/(2 ** s)
        fy = intrinsics[:,1,1]/(2 ** s)
        cx = intrinsics[:,0,2]/(2 ** s)
        cy = intrinsics[:,1,2]/(2 ** s)
        intrinsics_mscale.append(make_batch_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
    return intrinsics_mscale

# def preprocess_image(image):
#     # Assuming input image is uint8
#     if image == None:
#         return None
#     else:
#         image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#         return image * 2. - 1.

# def preprocess_image(image, is_dp=True):
#     # Assuming input image is uint8
#     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#     # if is_dp:
#     #     image = image * 2. - 1.
#     #     return image
#     # else:
#     #     mean = [104.920005, 110.1753, 114.785955]
#     #     out = []
#     #     for i in range(0, int(image.shape[-1]), 3):
#     #         r = image[:,:,:,i] - mean[0]/255.
#     #         g = image[:,:,:,i+1] - mean[1]/255.
#     #         b = image[:,:,:,i+2] - mean[2]/255.
#     #         out += [r, g, b]
#     #     return tf.stack(out, axis=-1)
#     return image

# def deprocess_image(image):
#     # Assuming input image is float32
#     image = (image + 1.)/2.
#     return tf.image.convert_image_dtype(image, dtype=tf.uint8)

def deprocess_image(image):
    # Assuming input image is float32
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def preprocess_image(image):
    # Assuming input image is uint8
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def inverse_warp_new(depth1,
                     depth2,
                     pose,
                     intrinsics,
                     intrinsics_inv,
                     flow_input,
                     occu_mask,
                     pose_mat_inverse=False):
    """
    Inverse warp a source image to the target image plane after refining the
    pose by rigid alignment described in
    'Joint Unsupervised Learning of Optical Flow and Depth by Watching
    Stereo Videos by Yang Wang et al.'
    Args:
        depth1: depth map of the target image -- [B, H, W]
        depth2: depth map of the source image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
        flow_input: flow between target and source image -- [B, H, W, 2]
        occu_mask: occlusion mask of target image -- [B, H, W, 1]
    Returns:
        [optical flow induced by refined pose,
         refined pose matrix,
         disparity of the target frame transformed by refined pose,
         the mask for areas used for rigid alignment]
    """

    def _pixel2cam(depth, pixel_coords, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame"""
        cam_coords = tf.matmul(intrinsics_inv, pixel_coords) * depth
        return cam_coords

    def _repeat(x, n_repeats):
        with tf.compat.v1.variable_scope('_repeat'):
            rep = tf.transpose(
                a=tf.expand_dims(
                    tf.ones(shape=tf.stack([n_repeats, ])), 1), perm=[1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _cam2pixel(cam_coords, proj_c2p):
        """Transform coordinates in the camera frame to the pixel frame"""
        pcoords = tf.matmul(proj_c2p, cam_coords)
        X = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
        Y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
        Z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
        # Not tested if adding a small number is necessary
        X_norm = X / (Z + 1e-10)
        Y_norm = Y / (Z + 1e-10)
        pixel_coords = tf.concat([X_norm, Y_norm], axis=1)
        return pixel_coords

    def _meshgrid_abs(height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                a=tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    def _meshgrid_abs_xy(batch, height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                a=tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        return tf.tile(tf.expand_dims(x_t, 0), [batch, 1, 1]), tf.tile(
            tf.expand_dims(y_t, 0), [batch, 1, 1])

    def _euler2mat(z, y, x):
        """Converts euler angles to rotation matrix
         TODO: remove the dimension for 'N' (deprecated for converting all source
               poses altogether)
         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

        Args:
            z: rotation angle along z axis (in radians) -- size = [B, N]
            y: rotation angle along y axis (in radians) -- size = [B, N]
            x: rotation angle along x axis (in radians) -- size = [B, N]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
        """
        B = tf.shape(input=z)[0]
        N = 1
        z = tf.clip_by_value(z, -np.pi, np.pi)
        y = tf.clip_by_value(y, -np.pi, np.pi)
        x = tf.clip_by_value(x, -np.pi, np.pi)

        # Expand to B x N x 1 x 1
        z = tf.expand_dims(tf.expand_dims(z, -1), -1)
        y = tf.expand_dims(tf.expand_dims(y, -1), -1)
        x = tf.expand_dims(tf.expand_dims(x, -1), -1)

        zeros = tf.zeros([B, N, 1, 1])
        ones = tf.ones([B, N, 1, 1])

        cosz = tf.cos(z)
        sinz = tf.sin(z)
        rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
        rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
        rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
        zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

        cosy = tf.cos(y)
        siny = tf.sin(y)
        roty_1 = tf.concat([cosy, zeros, siny], axis=3)
        roty_2 = tf.concat([zeros, ones, zeros], axis=3)
        roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
        ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

        cosx = tf.cos(x)
        sinx = tf.sin(x)
        rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
        rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
        rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
        xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

        rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
        return rotMat

    def _pose_vec2mat(vec):
        """Converts 6DoF parameters to transformation matrix
        Args:
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 4, 4]
        """
        translation = tf.slice(vec, [0, 0], [-1, 3])
        translation = tf.expand_dims(translation, -1)
        rx = tf.slice(vec, [0, 3], [-1, 1])
        ry = tf.slice(vec, [0, 4], [-1, 1])
        rz = tf.slice(vec, [0, 5], [-1, 1])
        rot_mat = _euler2mat(rz, ry, rx)
        rot_mat = tf.squeeze(rot_mat, axis=[1])
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])
        transform_mat = tf.concat([rot_mat, translation], axis=2)
        transform_mat = tf.concat([transform_mat, filler], axis=1)
        return transform_mat

    dims = tf.shape(input=depth1)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth1 = tf.reshape(depth1, [batch_size, 1, img_height * img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    # Point Cloud Q_1
    cam_coords1 = _pixel2cam(depth1, grid, intrinsics_inv)
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords1_hom = tf.concat([cam_coords1, ones], axis=1)
    if len(pose.get_shape().as_list()) == 3:
        pose_mat = pose
    else:
        pose_mat = _pose_vec2mat(pose)

    if pose_mat_inverse:
        pose_mat = tf.linalg.inv(pose_mat)
    # Point Cloud \hat{Q_1}
    cam_coords1_trans = tf.matmul(pose_mat, cam_coords1_hom)[:, 0:3, :]

    depth2 = tf.reshape(depth2, [batch_size, 1, img_height * img_width])
    # Point Cloud Q_2
    cam_coords2 = _pixel2cam(depth2, grid, intrinsics_inv)
    cam_coords2 = tf.reshape(cam_coords2,
                             [batch_size, 3, img_height, img_width])
    cam_coords2 = tf.transpose(a=cam_coords2, perm=[0, 2, 3, 1])
    cam_coords2_trans = transformer_old(cam_coords2, flow_input,
                                        [img_height, img_width])
    # Point Cloud \tilda{Q_1}
    cam_coords2_trans = tf.reshape(
        tf.transpose(a=cam_coords2_trans, perm=[0, 3, 1, 2]), [batch_size, 3, -1])

    occu_mask = tf.reshape(occu_mask, [batch_size, 1, -1])
    # To eliminate occluded area from the small_mask
    occu_mask = tf.compat.v1.where(occu_mask < 0.75,
                         tf.ones_like(occu_mask) * 10000.0,
                         tf.ones_like(occu_mask))

    diff2 = tf.sqrt(
        tf.reduce_sum(
            input_tensor=tf.square(cam_coords1_trans - cam_coords2_trans),
            axis=1,
            keepdims=True)) * occu_mask
    small_mask = tf.compat.v1.where(
        diff2 < tfp.stats.percentile(
            diff2, 25.0, axis=2, keepdims=True),
        tf.ones_like(diff2),
        tf.zeros_like(diff2))

    # Delta T
    rigid_pose_mat = calculate_pose_basis(cam_coords1_trans, cam_coords2_trans,
                                          small_mask, batch_size)
    # T' = deltaT x T
    pose_mat2 = tf.matmul(rigid_pose_mat, pose_mat)

    # Get projection matrix for tgt camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, hom_filler], axis=1)
    proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat2)
    src_pixel_coords = _cam2pixel(cam_coords1_hom, proj_cam_to_src_pixel)
    src_pixel_coords = tf.reshape(src_pixel_coords,
                                  [batch_size, 2, img_height, img_width])
    src_pixel_coords = tf.transpose(a=src_pixel_coords, perm=[0, 2, 3, 1])

    tgt_pixel_coords_x, tgt_pixel_coords_y = _meshgrid_abs_xy(
        batch_size, img_height, img_width)
    flow_x = src_pixel_coords[:, :, :, 0] - tgt_pixel_coords_x
    flow_y = src_pixel_coords[:, :, :, 1] - tgt_pixel_coords_y
    flow = tf.concat(
        [tf.expand_dims(flow_x, -1), tf.expand_dims(flow_y, -1)], axis=-1)

    cam_coords1_trans_z = tf.matmul(pose_mat2, cam_coords1_hom)[:, 2:3, :]
    cam_coords1_trans_z = tf.reshape(cam_coords1_trans_z,
                                     [batch_size, img_height, img_width, 1])
    disp1_trans = 1.0 / cam_coords1_trans_z

    return flow, pose_mat2, disp1_trans, tf.reshape(
        small_mask, [batch_size, img_height, img_width, 1])

def inverse_warp(depth,
                 pose,
                 intrinsics,
                 intrinsics_inv,
                 pose_mat_inverse=False):
    """Inverse warp a source image to the target image plane
       Part of the code modified from
       https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
    Args:
        depth: depth map of the "target" image -- [B, H, W]
        pose: 6DoF pose parameters from "target" to "source" -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        optical flow induced by the given depth and pose,
        pose matrix

    """

    def _pixel2cam(depth, pixel_coords, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame"""
        cam_coords = tf.matmul(intrinsics_inv, pixel_coords) * depth
        return cam_coords

    def _cam2pixel(cam_coords, proj_c2p):
        """Transform coordinates in the camera frame to the pixel frame"""
        pcoords = tf.matmul(proj_c2p, cam_coords)
        X = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
        Y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
        Z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
        # Not tested if adding a small number is necessary
        X_norm = X / (Z + 1e-10)
        Y_norm = Y / (Z + 1e-10)
        pixel_coords = tf.concat([X_norm, Y_norm], axis=1)
        return pixel_coords

    def _meshgrid_abs(height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                a=tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    def _meshgrid_abs_xy(batch, height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                a=tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        return tf.tile(tf.expand_dims(x_t, 0), [batch, 1, 1]), tf.tile(
            tf.expand_dims(y_t, 0), [batch, 1, 1])

    def _euler2mat(z, y, x):
        """Converts euler angles to rotation matrix
         TODO: remove the dimension for 'N' (deprecated for converting all source
               poses altogether)
         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

        Args:
            z: rotation angle along z axis (in radians) -- size = [B, N]
            y: rotation angle along y axis (in radians) -- size = [B, N]
            x: rotation angle along x axis (in radians) -- size = [B, N]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
        """
        B = tf.shape(input=z)[0]
        N = 1
        z = tf.clip_by_value(z, -np.pi, np.pi)
        y = tf.clip_by_value(y, -np.pi, np.pi)
        x = tf.clip_by_value(x, -np.pi, np.pi)

        # Expand to B x N x 1 x 1
        z = tf.expand_dims(tf.expand_dims(z, -1), -1)
        y = tf.expand_dims(tf.expand_dims(y, -1), -1)
        x = tf.expand_dims(tf.expand_dims(x, -1), -1)

        zeros = tf.zeros([B, N, 1, 1])
        ones = tf.ones([B, N, 1, 1])

        cosz = tf.cos(z)
        sinz = tf.sin(z)
        rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
        rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
        rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
        zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

        cosy = tf.cos(y)
        siny = tf.sin(y)
        roty_1 = tf.concat([cosy, zeros, siny], axis=3)
        roty_2 = tf.concat([zeros, ones, zeros], axis=3)
        roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
        ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

        cosx = tf.cos(x)
        sinx = tf.sin(x)
        rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
        rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
        rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
        xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

        rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
        return rotMat

    def _pose_vec2mat(vec):
        """Converts 6DoF parameters to transformation matrix
        Args:
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 4, 4]
        """
        translation = tf.slice(vec, [0, 0], [-1, 3])
        translation = tf.expand_dims(translation, -1)
        rx = tf.slice(vec, [0, 3], [-1, 1])
        ry = tf.slice(vec, [0, 4], [-1, 1])
        rz = tf.slice(vec, [0, 5], [-1, 1])
        rot_mat = _euler2mat(rz, ry, rx)
        rot_mat = tf.squeeze(rot_mat, axis=[1])
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])
        transform_mat = tf.concat([rot_mat, translation], axis=2)
        transform_mat = tf.concat([transform_mat, filler], axis=1)
        return transform_mat

    dims = tf.shape(input=depth)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    cam_coords = _pixel2cam(depth, grid, intrinsics_inv)
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)
    if len(pose.get_shape().as_list()) == 3:
        pose_mat = pose
    else:
        pose_mat = _pose_vec2mat(pose)

    if pose_mat_inverse:
        pose_mat = tf.linalg.inv(pose_mat)

    # Get projection matrix for tgt camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, hom_filler], axis=1)
    proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat)
    src_pixel_coords = _cam2pixel(cam_coords_hom, proj_cam_to_src_pixel)
    src_pixel_coords = tf.reshape(src_pixel_coords,
                                  [batch_size, 2, img_height, img_width])
    src_pixel_coords = tf.transpose(a=src_pixel_coords, perm=[0, 2, 3, 1])

    tgt_pixel_coords_x, tgt_pixel_coords_y = _meshgrid_abs_xy(
        batch_size, img_height, img_width)
    flow_x = src_pixel_coords[:, :, :, 0] - tgt_pixel_coords_x
    flow_y = src_pixel_coords[:, :, :, 1] - tgt_pixel_coords_y
    flow = tf.concat(
        [tf.expand_dims(flow_x, -1), tf.expand_dims(flow_y, -1)], axis=-1)
    return flow, pose_mat

def calculate_pose_basis(cam_coords1, cam_coords2, weights, batch_size):
    '''
    Given two point clouds and weights, find the transformation that
    minimizes the distance between the two clouds
    Args:
        cam_coords1: point cloud 1 -- [B, 3, -1]
        cam_coords2: point cloud 2 -- [B, 3, -1]
        weights: weights to specify which points in the point cloud are
                 used for alignment -- [B, 1, -1]
    return:
        transformation matrix -- [B, 4, 4]
    '''
    centroids1 = tf.reduce_mean(
        input_tensor=cam_coords1 * weights, axis=2, keepdims=True) / tf.reduce_mean(
            input_tensor=weights, axis=2, keepdims=True)
    centroids2 = tf.reduce_mean(
        input_tensor=cam_coords2 * weights, axis=2, keepdims=True) / tf.reduce_mean(
            input_tensor=weights, axis=2, keepdims=True)

    cam_coords1_shifted = tf.expand_dims(
        tf.transpose(a=cam_coords1 - centroids1, perm=[0, 2, 1]), -1)
    cam_coords2_shifted = tf.expand_dims(
        tf.transpose(a=cam_coords2 - centroids2, perm=[0, 2, 1]), -2)

    weights_trans = tf.expand_dims(tf.transpose(a=weights, perm=[0, 2, 1]), -1)
    H = tf.reduce_sum(
        input_tensor=tf.matmul(cam_coords1_shifted, cam_coords2_shifted) * weights_trans,
        axis=1,
        keepdims=False)
    S, U, V = tf.linalg.svd(H)
    R = tf.matmul(V, U, transpose_a=False, transpose_b=True)

    T = -tf.matmul(R, centroids1) + centroids2

    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    rigid_pose_mat = tf.concat([tf.concat([R, T], axis=2), filler], axis=1)

    return rigid_pose_mat

def get_imagenet_vars_to_restore(imagenet_ckpt):
    """Returns dict of variables to restore from ImageNet-checkpoint."""
    vars_to_restore_imagenet = {}
    ckpt_var_names = tf.train.list_variables(imagenet_ckpt)
    ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
    model_vars = tf.compat.v1.global_variables()

    for v in model_vars:
        if 'global_step' in v.op.name: continue
        mvname_noprefix = v.op.name.replace('depth_prediction/', '')
        mvname_noprefix = mvname_noprefix.replace('moving_mean', 'mu')
        mvname_noprefix = mvname_noprefix.replace('moving_variance', 'sigma')
        if mvname_noprefix in ckpt_var_names:
            vars_to_restore_imagenet[mvname_noprefix] = v
        else:
            logging.info('The following variable will not be restored from '
                         'pretrained ImageNet-checkpoint: %s', mvname_noprefix)

    return vars_to_restore_imagenet

def get_vars_to_save_and_restore(ckpt=None):
    """Returns list of variables that should be saved/restored.

    Args:
    ckpt: Path to existing checkpoint.  If present, returns only the subset of
        variables that exist in given checkpoint.

    Returns:
    List of all variables that need to be saved/restored.
    """
    model_vars = tf.compat.v1.trainable_variables()
    # Add batchnorm variables.
    bn_vars = [v for v in tf.compat.v1.global_variables()
               if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name or
               'mu' in v.op.name or 'sigma' in v.op.name]
    model_vars.extend(bn_vars)
    model_vars = sorted(model_vars, key=lambda x: x.op.name)
    mapping = {}
    if ckpt is not None:
        ckpt_var = tf.train.list_variables(ckpt)
        ckpt_var_names = [name for (name, unused_shape) in ckpt_var]
        ckpt_var_shapes = [shape for (unused_name, shape) in ckpt_var]
        not_loaded = list(ckpt_var_names)
        for v in model_vars:
            if v.op.name not in ckpt_var_names:
                # For backward compatibility, try additional matching.
                v_additional_name = v.op.name.replace('egomotion_prediction/', '')
                if v_additional_name in ckpt_var_names:
                    # Check if shapes match.
                    ind = ckpt_var_names.index(v_additional_name)
                    if ckpt_var_shapes[ind] == v.get_shape():
                        mapping[v_additional_name] = v
                        not_loaded.remove(v_additional_name)
                        continue
                    else:
                        print('Shape mismatch, will not restore %s.' % v.op.name)
                print('Did not find var %s in checkpoint: %s' % v.op.name,
                     os.path.basename(ckpt))
            else:
                # Check if shapes match.
                ind = ckpt_var_names.index(v.op.name)
                if ckpt_var_shapes[ind] == v.get_shape():
                    mapping[v.op.name] = v
                    not_loaded.remove(v.op.name)
                else:
                    print('Shape mismatch, will not restore %s.' % v.op.name)
        if not_loaded:
            print('The following variables in the checkpoint were not loaded:')
            for varname_not_loaded in not_loaded:
                print('%s' % varname_not_loaded)
    else:  # just get model vars.
        for v in model_vars:
            mapping[v.op.name] = v
    return mapping
