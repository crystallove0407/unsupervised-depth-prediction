import tensorflow as tf
import numpy as np
import cv2


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
    return tf.reduce_sum(input_tensor=tf.square(true - pred)) / tf.cast(tf.size(input=pred), dtype=tf.float32)

# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py


def ternary_loss(im1, im2_warped, valid_mask, max_distance=1):
    patch_size = 2*max_distance+1
    with tf.compat.v1.variable_scope('ternary_loss'):
        def _ternary_transform(image):
            intensities = tf.image.rgb_to_grayscale(image) * 255
            out_channels = patch_size * patch_size
            w = np.eye(out_channels).reshape(
                (patch_size, patch_size, 1, out_channels))
            weights = tf.constant(w, dtype=tf.float32)
            patches = tf.nn.conv2d(input=intensities, filters=weights, strides=[
                                   1, 1, 1, 1], padding='SAME')

            transf = patches - intensities
            transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
            return transf_norm

        def _hamming_distance(t1, t2):
            dist = tf.square(t1 - t2)
            dist_norm = dist / (0.1 + dist)
            dist_sum = tf.reduce_sum(
                input_tensor=dist_norm, axis=3, keepdims=True)
            return dist_sum

    t1 = _ternary_transform(im1)
    t2 = _ternary_transform(im2_warped)
    dist = _hamming_distance(t1, t2)

    transform_mask = create_mask(
        valid_mask, [[max_distance, max_distance], [max_distance, max_distance]])
    return charbonnier_loss(dist, valid_mask * transform_mask), dist


def create_mask(tensor, paddings):
    with tf.compat.v1.variable_scope('create_mask'):
        shape = tf.shape(input=tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(tensor=inner, paddings=paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return tf.stop_gradient(mask4d)


def weighted_mean_squared_error(true, pred, weight):
    """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """

    tmp = tf.reduce_sum(
        input_tensor=weight * tf.square(true - pred), axis=[1, 2],
        keepdims=True) / tf.reduce_sum(
            input_tensor=weight, axis=[1, 2], keepdims=True)
    return tf.reduce_mean(input_tensor=tmp)


def mean_L1_error(true, pred):
    """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
    return tf.reduce_sum(input_tensor=tf.abs(true - pred)) / tf.cast(tf.size(input=pred), dtype=tf.float32)


def weighted_mean_L1_error(true, pred, weight):
    """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
    return tf.reduce_sum(input_tensor=tf.abs(true - pred) *
                         weight) / tf.cast(tf.size(input=pred), dtype=tf.float32)


def cal_grad2_error(flo, image, beta):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo
    """

    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy

    img_grad_x, img_grad_y = gradient(image)
    weights_x = tf.exp(-10.0 * tf.reduce_mean(
        input_tensor=tf.abs(img_grad_x), axis=3, keepdims=True))
    weights_y = tf.exp(-10.0 * tf.reduce_mean(
        input_tensor=tf.abs(img_grad_y), axis=3, keepdims=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (tf.reduce_mean(input_tensor=beta*weights_x[:, :, 1:, :]*tf.abs(dx2)) +
            tf.reduce_mean(input_tensor=beta*weights_y[:, 1:, :, :]*tf.abs(dy2))) / 2.0


def cal_grad2_error_mask(flo, image, beta, mask):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo
    within the given mask
    """

    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy

    img_grad_x, img_grad_y = gradient(image)
    weights_x = tf.exp(-10.0 * tf.reduce_mean(
        input_tensor=tf.abs(img_grad_x), axis=3, keepdims=True))
    weights_y = tf.exp(-10.0 * tf.reduce_mean(
        input_tensor=tf.abs(img_grad_y), axis=3, keepdims=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (tf.reduce_mean(input_tensor=beta*weights_x[:, :, 1:, :]*tf.abs(dx2) * mask[:, :, 1:-1, :]) +
            tf.reduce_mean(input_tensor=beta*weights_y[:, 1:, :, :]*tf.abs(dy2) * mask[:, 1:-1, :, :])) / 2.0


def compute_edge_aware_smooth_loss(pred_disp, img):
    """
    Edge-aware L1-norm on first-order gradient
    """
    def gradient(pred):
        D_dx = -pred[:, :, 1:, :] + pred[:, :, :-1, :]
        D_dy = -pred[:, 1:, :, :] + pred[:, :-1, :, :]
        return D_dx, D_dy
    img_dx, img_dy = gradient(img)
    disp_dx, disp_dy = gradient(pred_disp)

    weight_x = tf.exp(-tf.reduce_mean(input_tensor=tf.abs(img_dx),
                                      axis=3, keepdims=True))
    weight_y = tf.exp(-tf.reduce_mean(input_tensor=tf.abs(img_dy),
                                      axis=3, keepdims=True))

    loss = tf.reduce_mean(input_tensor=weight_x*tf.abs(disp_dx)) + \
        tf.reduce_mean(input_tensor=weight_y*tf.abs(disp_dy))
    return loss


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]


def depth_smoothness(depth, img):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    # depth_dx = tf.Print(depth_dx, ["[!] depth_dx: ", depth_dx], summarize=100)

    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    # image_dx = tf.Print(image_dx, ["[!] image_dx: ", image_dx], summarize=100)

    weights_x = tf.exp(-tf.reduce_mean(input_tensor=tf.abs(image_dx),
                                       axis=3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(input_tensor=tf.abs(image_dy),
                                       axis=3, keepdims=True))
    # weights_x = tf.Print(weights_x, ["[!] weights_x: ", weights_x], summarize=100)
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return tf.reduce_mean(input_tensor=abs(smoothness_x)) + tf.reduce_mean(input_tensor=abs(smoothness_y))


def SSIM(x, y):
    C1 = 0.01**2
    C2 = 0.03**2

    default_pad = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    x = tf.pad(tensor=x, paddings=default_pad, mode='REFLECT')
    y = tf.pad(tensor=y, paddings=default_pad, mode='REFLECT')

    mu_x = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1)(x)
    mu_y = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1)(y)

    sigma_x = tf.keras.layers.AveragePooling2D(
        pool_size=3, strides=1)(x**2) - mu_x**2
    sigma_y = tf.keras.layers.AveragePooling2D(
        pool_size=3, strides=1)(y**2) - mu_y**2
    sigma_xy = tf.keras.layers.AveragePooling2D(
        pool_size=3, strides=1)(x*y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def charbonnier_loss(x,
                     mask=None,
                     truncate=None,
                     alpha=0.45,
                     beta=1.0,
                     epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.
    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    batch, height, width, channels = tf.unstack(tf.shape(input=x))
    normalization = tf.cast(batch * height * width * channels, tf.float32)

    error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

    if mask is not None:
        error = tf.multiply(mask, error)

    if truncate is not None:
        error = tf.minimum(error, truncate)

    return tf.reduce_sum(input_tensor=error) / normalization
