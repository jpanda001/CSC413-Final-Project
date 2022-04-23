import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import shutil
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

import model.model as M
from data.batch_generator import DataGenerator
import metric.AP as mAP

import utilities.loader as loader
import utilities.helper as helper
import utilities.drawer as drawer


def main():
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) > 0:
        for hardware in gpu:
            tf.config.experimental.set_memory_growth(hardware, True)
        logical_gpu = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpu), "Physical GPUs,", len(logical_gpu), "Logical GPUs")

    config_setup = loader.loadConfig()
    data_configuration = config_setup["DATA"]
    radar_configuration = config_setup["RADAR_CONFIGURATION"]
    model_configuration = config_setup["MODEL"]
    training_configuration = config_setup["TRAIN"]

    anchor_boxes = loader.loadAnchorBoxes() # load anchor boxes with order
    num_classes = len(data_configuration["all_classes"])

    # Using the yolo head shape out from model for data generator
    model = M.RadarYOLO(model_configuration, data_configuration, training_configuration, anchor_boxes)
    model.build([None] + model_configuration["input_shape"])
    model.summary()

    # Preparing data 
    data_generator = DataGenerator(data_configuration, training_configuration, model_configuration, \
                                model.features_shape, anchor_boxes)
    train_generator = data_generator.trainGenerator()
    validate_generator = data_generator.validateGenerator()

    # Training settings #
    logdir = os.path.join(training_configuration["log_dir"], \
                        "b_" + str(training_configuration["batch_size"]) + \
                        "lr_" + str(training_configuration["learningrate_init"]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    optimizer = K.optimizers.Adam(learning_rate=training_configuration["learningrate_init"])
    writer = tf.summary.create_file_writer(logdir)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model, step=global_steps)
    log_specific_dir = os.path.join(logdir, "ckpt")
    ckpt_manager = tf.train.CheckpointManager(ckpt, log_specific_dir, max_to_keep=3)

    # Restore from last checkpoint
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        global_steps.assign(ckpt.step.numpy())

    # Define training step
    @tf.function
    def train_step(data, label):
        """ define train step for training """
        with tf.GradientTape() as tape:
            feature = model(data)
            pred_raw, pred = model.YOLODecoder(feature)
            total_loss, box_loss, conf_loss, category_loss = \
                model.loss(pred_raw, pred, label, raw_boxes[..., :6])
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss, box_loss, conf_loss, category_loss
 
    # Define validate step
    # @tf.function
    def validate_step():
        mean_ap_test = 0.0
        ap_all_class_test = []
        ap_all_class = []
        total_losstest = []
        box_losstest = []
        conf_losstest = []
        category_losstest = []
        for class_id in range(num_classes):
            ap_all_class.append([])
        for data, label, raw_boxes in validate_generator.\
            batch(data_generator.batch_size).take(data_generator.total_validate_batches):
            feature = model(data)
            pred_raw, pred = model.decodeYolo(feature)
            total_loss_b, box_loss_b, conf_loss_b, category_loss_b = \
                model.loss(pred_raw, pred, label, raw_boxes[..., :6])
            total_losstest.append(total_loss_b)
            box_losstest.append(box_loss_b)
            conf_losstest.append(conf_loss_b)
            category_losstest.append(category_loss_b)
            for batch_id in range(raw_boxes.shape[0]):
                raw_boxes_frame = raw_boxes[batch_id]
                pred_frame = pred[batch_id]
                predicitons = helper.yoloheadPredictor(pred_frame, \
                                    conf_threshold=model_configuration["confidence_threshold"])
                nms_pred = helper.nms(predicitons, model_configuration["nms_iou3d_threshold"], \
                                model_configuration["input_shape"], sigma=0.3, method="nms")
                mean_ap, ap_all_class = mAP.mAP(nms_pred, raw_boxes_frame.numpy(), \
                                        model_configuration["input_shape"], ap_all_class, \
                                        tp_iou_threshold=model_configuration["mAP_iou3d_threshold"])
                mean_ap_test += mean_ap
        for ap_class_i in ap_all_class:
            if len(ap_class_i) == 0:
                class_ap = 0.
            else:
                class_ap = np.mean(ap_class_i)
            ap_all_class_test.append(class_ap)
        mean_ap_test /= data_generator.batch_size*data_generator.total_validate_batches
        tf.print("-------> ap: %.6f"%(mean_ap_test))
        # writing summary data 
        with writer.as_default():
            tf.summary.scalar("ap/ap_all", mean_ap_test, step=global_steps)
            tf.summary.scalar("ap/ap_person", ap_all_class_test[0], step=global_steps)
            tf.summary.scalar("ap/ap_bicycle", ap_all_class_test[1], step=global_steps)
            tf.summary.scalar("ap/ap_car", ap_all_class_test[2], step=global_steps)
            tf.summary.scalar("ap/ap_motorcycle", ap_all_class_test[3], step=global_steps)
            tf.summary.scalar("ap/ap_bus", ap_all_class_test[4], step=global_steps)
            tf.summary.scalar("ap/ap_truck", ap_all_class_test[5], step=global_steps)
            # Validate loss 
            tf.summary.scalar("validate_loss/total_loss", \
                    np.mean(total_losstest), step=global_steps)
            tf.summary.scalar("validate_loss/box_loss", \
                    np.mean(box_losstest), step=global_steps)
            tf.summary.scalar("validate_loss/conf_loss", \
                    np.mean(conf_losstest), step=global_steps)
            tf.summary.scalar("validate_loss/category_loss", \
                    np.mean(category_losstest), step=global_steps)
        writer.flush()

    ###---------------------------- TRAIN SET -------------------------###
    for data, label, raw_boxes in train_generator.repeat().\
            batch(data_generator.batch_size).take(data_generator.total_train_batches):
        total_loss, box_loss, conf_loss, category_loss = train_step(data, label)
        tf.print("=======> train step: %4d, lr: %.6f, total_loss: %4.2f,  \
                box_loss: %4.2f, conf_loss: %4.2f, category_loss: %4.2f" % \
                (global_steps, optimizer.lr.numpy(), total_loss, box_loss, \
                conf_loss, category_loss))
        ### NOTE: learning rate decay ###
        global_steps.assign_add(1)
        if global_steps < training_configuration["warmup_steps"]:
            if global_steps < training_configuration["startup_steps"]:
                lr = training_configuration["learningrate_startup"]
            else:
                lr = training_configuration["learningrate_init"]
            optimizer.lr.assign(lr)
        elif global_steps % training_configuration["learningrate_decay_gap"] == 0:
            lr = optimizer.lr.numpy()
            lr = training_configuration["learningrate_end"] + \
                    training_configuration["learningrate_decay"] * \
                    (lr - training_configuration["learningrate_end"])
            optimizer.lr.assign(lr)
 
        ###---------------------------- VALIDATE SET -------------------------###
        if global_steps.numpy() >= training_configuration["validate_start_steps"] and \
                global_steps.numpy() % training_configuration["validate_gap"] == 0:
            validate_step()
            save_path = ckpt_manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))


if __name__ == "__main__":
    main()

