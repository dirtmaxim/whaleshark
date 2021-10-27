import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coeff(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.clip(K.batch_flatten(y_true), K.epsilon(), 1.0)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def loss(y_true, y_pred):
    b = tf.keras.losses.binary_crossentropy(K.clip(y_true, K.epsilon(), 1.0), K.clip(y_pred, K.epsilon(), 1.0))
    d = dice_coeff(y_true, y_pred)

    return 1 - K.log(d) + b


def iou(y_true, y_pred):
    mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)
    mean_iou.update_state(y_true, y_pred)

    return mean_iou.result().numpy()
