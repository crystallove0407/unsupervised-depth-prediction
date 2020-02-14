import tensorflow as tf

def transformer(pred_flow, input_image=None):
    if input_image is None: # Get occlusion
        n, h, w, c = pred_flow.get_shape().as_list()
        n=8
        ones = tf.ones(shape=[n, h, w, 1], dtype='float32')
        occu_mask = _transform(pred_flow, ones, type='new', backprop=True)
        occu_mask = tf.clip_by_value(occu_mask, 0.0, 1.0)
        occu_mask_avg = tf.reduce_mean(occu_mask)
        return occu_mask, occu_mask_avg
        
    
    else: # Get transform
        return _transform(pred_flow, input_image)


def _repeat(x, n_repeats):
    rep = tf.transpose(
        a=tf.expand_dims(
            tf.ones(shape=[n_repeats, ]), 1), perm=[1, 0])
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, [-1, 1]), rep)
    return tf.reshape(x, [-1])

def _meshgrid(height, width):
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

def _interpolate(im, x, y, type='old', backprop=True):
    # constants
    n, h, w, c = im.get_shape().as_list()
    n=8
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    height_f = tf.cast(h, 'float32')
    width_f = tf.cast(w, 'float32')
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

    dim2 = w
    dim1 = w * h
    base = _repeat(tf.range(n) * dim1, h * w)

    base_y0 = base + y0_c * dim2
    base_y1 = base + y1_c * dim2
    idx_a = base_y0 + x0_c
    idx_b = base_y1 + x0_c
    idx_c = base_y0 + x1_c
    idx_d = base_y1 + x1_c

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    
    im_flat = tf.reshape(im, [-1, c])
    im_flat = tf.cast(im_flat, 'float32')
    

    # and finally calculate interpolated values
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
    wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
    wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
    wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)

    if type == 'old':
        
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        return output

    if type == 'new':
        zerof = tf.zeros_like(wa)

        wa = tf.compat.v1.where(
            tf.logical_and(tf.equal(x0_c, x0), tf.equal(y0_c, y0)), wa,
            zerof)
        wb = tf.compat.v1.where(
            tf.logical_and(tf.equal(x0_c, x0), tf.equal(y1_c, y1)), wb,
            zerof)
        wc = tf.compat.v1.where(
            tf.logical_and(tf.equal(x1_c, x1), tf.equal(y0_c, y0)), wc,
            zerof)
        wd = tf.compat.v1.where(
            tf.logical_and(tf.equal(x1_c, x1), tf.equal(y1_c, y1)), wd,
            zerof)

        if not backprop:
            output = tf.zeros(shape=[n*h*w, c], dtype='float32')
            output = tf.Variable(zeros, trainable=False)
            tf.compat.v1.assign(output, zeros)

            output = tf.tensor_scatter_nd_add(output, idx_a, im_flat * wa)
            output = tf.tensor_scatter_nd_add(output, idx_b, im_flat * wb)
            output = tf.tensor_scatter_nd_add(output, idx_c, im_flat * wc)
            output = tf.tensor_scatter_nd_add(output, idx_d, im_flat * wd)
        else:
            shape = [n*h*w, c]
            output = tf.scatter_nd(tf.expand_dims(idx_a, -1), im_flat*wa, shape) + \
                        tf.scatter_nd(tf.expand_dims(idx_b, -1), im_flat*wb, shape) + \
                        tf.scatter_nd(tf.expand_dims(idx_c, -1), im_flat*wc, shape) + \
                        tf.scatter_nd(tf.expand_dims(idx_d, -1), im_flat*wd, shape)
            output = tf.stop_gradient(output)

        return output

def _transform(flo, input_image, type='old', backprop=True):
    n, h, w, c = input_image.get_shape().as_list()
    n=8
    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    x_s, y_s = _meshgrid(h, w)
    x_s = tf.expand_dims(x_s, 0)
    x_s = tf.tile(x_s, [n, 1, 1])

    y_s = tf.expand_dims(y_s, 0)
    y_s = tf.tile(y_s, [n, 1, 1])

    x_t = x_s + flo[:, :, :, 0] / ((w - 1.0) / 2.0)
    y_t = y_s + flo[:, :, :, 1] / ((h - 1.0) / 2.0)

    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    

    input_transformed = _interpolate(input_image, x_t_flat, y_t_flat, type=type, backprop=backprop)
    output = tf.reshape(input_transformed, [n, h, w, c])

    return output