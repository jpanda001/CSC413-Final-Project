import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model.layer_helpers as L
import util.helper as helper


def getYoloInfo(yolo_output):
    """ Extract information about box, objectness and class """
    box_info = yolo_output[..., :6]
    conf_info = yolo_output[..., 6:7]
    category_info = yolo_output[..., 7:]
    
    return box_info, conf_info, category_info
        
def yoloBoxLoss(box_prediction, box_ground_truth, conf_ground_truth, input, loss_box_scale=True):
    """ loss function for yolo box regression """
    if loss_box_scale:
        scale = 2.0 - box_ground_truth[..., 3:4] * box_ground_truth[..., 4:5] * box_ground_truth[..., 5:6] /\
                                    (input[0] * input[1] * input[2])
    else:
        scale = 1.0
    
    loss = conf_ground_truth * scale * (tf.square(box_prediction[..., :3] - conf_ground_truth[..., :3]) + \
                    tf.square(tf.sqrt(box_prediction[..., 3:]) - tf.sqrt(conf_ground_truth[..., 3:])))
    
    return loss

def lossFocal(confidence_raw, confidence_prediction, confidence_groundTruth, box_prediction, boxes_raw_pred, input_size, \
            iou_loss_threshold=0.5):
    """ Focal loss estimation for objectness from YOLO"""
    iou_value = helper.find_iou3d(box_prediction[:, :, :, :, :, tf.newaxis, :], \
                    boxes_raw_pred[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :],\
                    input_size) 

    max_iou = tf.expand_dims(tf.reduce_max(iou_value, axis=-1), axis=-1)
    
    loss = tf.pow(confidence_groundTruth - confidence_prediction, 2) * (\
            confidence_groundTruth * tf.nn.sigmoid_cross_entropy_with_logits(labels=confidence_groundTruth, \
                                                            logits=confidence_raw) \
            + \
            0.01 * (1.0 - confidence_groundTruth) * tf.cast(max_iou < iou_loss_threshold, tf.float32) * \
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=confidence_groundTruth, \
                                                            logits=confidence_raw) \
            )
    
    return loss

def lossCategory(category_raw, category_prediction, category_groundTruth, confidence_groundTruth):
    """ Category Cross Entropy Loss """
    loss = confidence_groundTruth * tf.nn.sigmoid_cross_entropy_with_logits(labels=category_groundTruth, logits=category_raw)
    
    return loss

def YOLOLoss(raw_predictions, processed_predictions, label_groundTruth, boxes_raw, input_dim, loss_focal_threshold_iou):
    """ Calculate loss function of YOLO HEAD 
    Args:
        feature_stages      ->      3 different feature stages after YOLO HEAD
                                    with shape [None, r, a, d, num_anchors, 7+num_class]
        gt_stages           ->      3 different ground truth stages 
                                    with shape [None, r, a, d, num_anchors, 7+num_class]"""
                                    
    box_raw, conf_raw, category_raw = getYoloInfo(raw_predictions)

    box_pred, conf_pred, category_pred = getYoloInfo(processed_predictions)
    
    box_groundTruth, conf_groundTruth, category_groundTruth = getYoloInfo(label_groundTruth)
    
    loss_giou = yoloBoxLoss(box_pred, box_groundTruth, conf_groundTruth, input_dim, \
                            if_box_loss_scale=False)
    
    loss_focal = lossFocal(conf_raw, conf_pred, conf_groundTruth, box_pred, boxes_raw, \
                            input_dim, loss_focal_threshold_iou)
    
    loss_category = lossCategory(category_raw, category_pred, category_groundTruth, conf_groundTruth)
    
    total_loss_giou = tf.reduce_mean(tf.reduce_sum(loss_giou, axis=[1,2,3,4]))
    
    total_loss_conf = tf.reduce_mean(tf.reduce_sum(loss_focal, axis=[1,2,3,4]))
    
    total_loss_category = tf.reduce_mean(tf.reduce_sum(loss_category, axis=[1,2,3,4]))
    
    return total_loss_giou, total_loss_conf, total_loss_category

