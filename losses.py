# from tensorflow.keras.losses import binary_crossentropy
# import tensorflow.keras.backend as K
import tensorflow as tf

epsilon = 1e-5
smooth = 1.

def tversky(y_true, y_pred):
    smooth = 1.
    y_true_pos = tf.reshape(y_true, [1, -1])
    y_pred_pos = tf.reshape(y_pred, [1, -1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1.0-y_pred_pos))
    false_pos = tf.reduce_sum((1.0-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1.0-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1.0 - tversky(y_true, y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.pow((1.0-pt_1), gamma)
#
# def dsc(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return score
#
# def dice_loss(y_true, y_pred):
#     loss = 1 - dsc(y_true, y_pred)
#     return loss
#
# def bce_dice_loss(y_true, y_pred):
#     loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#     return loss
#
# def confusion(y_true, y_pred):
#     smooth=1
#     y_pred_pos = K.clip(y_pred, 0, 1)
#     y_pred_neg = 1 - y_pred_pos
#     y_pos = K.clip(y_true, 0, 1)
#     y_neg = 1 - y_pos
#     tp = K.sum(y_pos * y_pred_pos)
#     fp = K.sum(y_neg * y_pred_pos)
#     fn = K.sum(y_pos * y_pred_neg)
#     prec = (tp + smooth)/(tp+fp+smooth)
#     recall = (tp+smooth)/(tp+fn+smooth)
#     return prec, recall
#
# def tp(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
#     return tp
#
# def tn(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pred_neg = 1 - y_pred_pos
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     y_neg = 1 - y_pos
#     tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
#     return tn
