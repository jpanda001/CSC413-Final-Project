import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model.layer_helpers as L

import model.ResNet_backbone as ResNetBack
import model.YOLO_heads as yolohead
import model.loss_helpers as losses

class RadarYOLO(K.Model):
    def __init__(self, model_config, dataset_config, training_config, anchors):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables 
        could be initialized """
        super(RadarYOLO, self).__init__()

        self.input_dims = list(model_config["input_shape"])
        self.input_channels = self.input_dims[-1]
        self.num_classes = len(dataset_config["all_classes"])
        self.anchors = anchors
        self.yolo_scales = model_config["yolohead_xyz_scales"]
        self.loss_iou_threshold = training_config["focal_loss_iou_threshold"]
        self.model = self.model()

    def model(self,):
        """ Constructing the RadarYOLO model"""

        RAV_input = K.layers.Input(self.input_dims)

        # Backbone 
        features = ResNetBack.ResNet(RAV_input)

        # Detection heads
        yolo_out = yolohead.yoloHead(features, self.anchors, self.num_classes)

        # defining and compiling overall model
        model = K.Model(RAV_input, yolo_out)

        return model

    def YOLODecoder(self, yolo_raw):
        raw_predictions, pred_processed = yolohead.boxDecoder(yolo_raw, self.input_dims, self.anchors, self.num_classes, self.yolo_scales[0])
        return raw_predictions, pred_processed

    def loss(self, raw_predictions, prediction_processed, ground_truth, raw_boxes):
        box_loss, confidence_loss, category_loss = losses.YOLOLoss(raw_predictions, prediction_processed, \
                            ground_truth, raw_boxes, self.input_dims, self.loss_iou_threshold)

        total_loss = box_loss/10 + confidence_loss + category_loss
        return total_loss, box_loss/10, confidence_loss, category_loss

    def model_call(self, input_features):
        out = self.model(input_features)
        return out
