import tensorflow as tf
from tensorflow.contrib import slim


def Upsampling(inputs, scale):
    return tf.image.resize_nearest_neighbor(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])


def Unpooling(inputs, scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])


def ResidualUnit(inputs, n_filters=48, filter_size=3):
   

    net = slim.conv2d(inputs, n_filters, filter_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, filter_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)

    return net


def FullResolutionResidualUnit(pool_stream, res_stream, n_filters_3, n_filters_1, pool_scale):
    

    G = tf.concat([pool_stream, slim.pool(res_stream, [pool_scale, pool_scale], stride=[pool_scale, pool_scale],
                                          pooling_type='MAX')], axis=-1)

    net = slim.conv2d(G, n_filters_3, kernel_size=3, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters_3, kernel_size=3, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    pool_stream_out = tf.nn.relu(net)

    net = slim.conv2d(pool_stream_out, n_filters_1, kernel_size=1, activation_fn=None)
    net = Upsampling(net, scale=pool_scale)
    res_stream_out = tf.add(res_stream, net)

    return pool_stream_out, res_stream_out


def build_frrn(inputs, num_classes, preset_model='FRRN-A'):
    


      
        net = slim.conv2d(inputs, 48, kernel_size=5, activation_fn=None)
        net = slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)

        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)

        pool_stream = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
        res_stream = slim.conv2d(net, 32, kernel_size=1, activation_fn=None)

        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=384, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=384, n_filters_1=32, pool_scale=8)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=384, n_filters_1=32, pool_scale=16)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=384, n_filters_1=32, pool_scale=16)

      
        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=192, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=192, n_filters_1=32, pool_scale=8)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream,
                                                             n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = Unpooling(pool_stream, 2)

        
        net = tf.concat([pool_stream, res_stream], axis=-1)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
        return net


