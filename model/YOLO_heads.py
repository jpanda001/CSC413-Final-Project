import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model.layer_helpers as L
import util.helper as helper


def singleLayerHead(input_features, num_anchors, num_classes, grid_depth):
    """ YOLO HEAD for one specific feature stage (after FPN) """

    # 7 items because [x_center, y_center, z_Center, w, h ,d, objectness]
    output_channels = grid_depth * num_anchors * (num_classes + 7))
    output_shape = [-1] + list(input_features.shape[1:-1]) + \
                        [int(grid_depth), int(num_anchors) * (num_classes + 7)]
    
    # Upsampling just like in YOLOv4
    convOut = L.convolution2D(input_features, feature_map.shape[-1]*2, \
            3, strides, activation=True, biases=use_bias, batchNormalization=True \
            if_regularization=False)

    convOut = L.convolution2D(convOut, output_channels, \
            3, strides, activation=True, biases=use_bias, batchNormalization=True \
            if_regularization=False)

    convOut = tf.reshape(convOut, output_shape)
    
    return convOut


def boxDecoder(YOLO_heads, input_features_size, anchor_layers, num_classes, scale=1.):
    """ Getting bouding boxes from YOLO output"""

    grid_sizes = YOLO_heads.shape[1:4]

    reshape = tuple([tf.shape(YOLO_heads)[0]] + list(grid_size) + [len(anchor_layers), 7+num_classes])
    
    raw_predictions = tf.reshape(YOLO_heads, reshape)

    centers_raw, sizes_raw, confidence_raw, probability_raw = tf.split(raw_predictions, (3,3,1,num_classes), axis=-1)

    sizes_raw = tf.clip_by_value(sizes_raw, 1e-12, 1e12)

    grid = tf.expand_dims(tf.stack(tf.meshgrid(tf.range(grid_sizes[0]), tf.range(grid_sizes[1]), \
            tf.range(grid_sizes[2])), axis=-1), axis=3)

    grid = tf.tile(tf.expand_dims(tf.transpose(grid, perm=[1,0,2,3,4]), axis=0), \
            [tf.shape(YOLO_heads)[0], 1, 1, 1,  len(anchors_layer), 1])

    grid = tf.cast(grid, tf.float32)

    # Applying formula to derive bounding boxes
    strides = np.array(input_features_size) / np.array(list(grid_sizes))

    pred_centers = ((tf.sigmoid(centers_raw) * scale) - 0.5 * (scale - 1) + grid) * \
                strides

    pred_sizes = tf.exp(sizes_raw) * anchors_layer

    pred_combined = tf.concat([pred_centers, pred_sizes], axis=-1)

    conf_pred = tf.sigmoid(confidence_raw)
    prob_pred = tf.sigmoid(probability_raw)

    return pred_raw, tf.concat([pred_combined, conf_pred, prob_pred], axis=-1)


def yoloHead(input_features, all_anchors, num_classes):
    """Running YOLO heads 
    """
    grid_depth = int(input_features.shape[1]/4)
    raw_yolo_out = singleLayerHead(input_features, len(all_anchors), num_classes, grid_depth)
    
    return raw_yolo_out
