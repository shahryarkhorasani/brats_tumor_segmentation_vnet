import tensorflow as tf
from tensorflow.keras import backend as K

def generalised_dice_loss_3D(y_true, y_pred):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    """

    shapes = K.shape(y_pred)

    # flatten everything except channels (i.e. class probabilities)
    y_true = K.reshape(y_true, (shapes[0], shapes[1] * shapes[2] * shapes[3] , shapes[4]))
    y_pred = K.reshape(y_pred, (shapes[0], shapes[1] * shapes[2] * shapes[3] , shapes[4]))

    # equal weights for each class:
    # weights = 1. - ((K.sum(y_true, 2)+1.) / vol)

    # weights like in paper:
    weights = 1. / K.square((K.sum(y_true, 1) + 1.))

    overlaps = K.sum(y_pred * y_true, axis=1)
    total = K.sum(y_pred + y_true, axis=1)

    numerator = -2. * (weights * overlaps)
    denominator = total * weights

    return 1. + K.sum(numerator, -1) / K.sum(denominator, -1)


def brats_wt_metric(y_true, y_pred):
    # whole tumor
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    gt_wt = tf.cast(tf.identity(y_true), tf.int32)
    gt_wt = tf.where(tf.equal(2, gt_wt), 1 * tf.ones_like(gt_wt), gt_wt)  # ground_truth_wt[ground_truth_wt == 2] = 1
    gt_wt = tf.where(tf.equal(3, gt_wt), 1 * tf.ones_like(gt_wt), gt_wt)  # ground_truth_wt[ground_truth_wt == 3] = 1
    pd_wt = tf.cast(tf.round(tf.identity(y_pred)), tf.int32)
    pd_wt = tf.where(tf.equal(2, pd_wt), 1 * tf.ones_like(pd_wt), pd_wt)  # predictions_wt[predictions_wt == 2] = 1
    pd_wt = tf.where(tf.equal(3, pd_wt), 1 * tf.ones_like(pd_wt), pd_wt)  # predictions_wt[predictions_wt == 3] = 1
    return _dice_hard_coe(gt_wt, pd_wt)


def brats_tc_metric(y_true, y_pred):
    # tumor core
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    gt_tc = tf.cast(tf.identity(y_true), tf.int32)
    gt_tc = tf.where(tf.equal(2, gt_tc), 0 * tf.ones_like(gt_tc), gt_tc)  # ground_truth_tc[ground_truth_tc == 2] = 0
    gt_tc = tf.where(tf.equal(3, gt_tc), 1 * tf.ones_like(gt_tc), gt_tc)  # ground_truth_tc[ground_truth_tc == 3] = 1
    pd_tc = tf.cast(tf.round(tf.identity(y_pred)), tf.int32)
    pd_tc = tf.where(tf.equal(2, pd_tc), 0 * tf.ones_like(pd_tc), pd_tc)  # predictions_tc[predictions_tc == 2] = 0
    pd_tc = tf.where(tf.equal(3, pd_tc), 1 * tf.ones_like(pd_tc), pd_tc)  # predictions_tc[predictions_tc == 3] = 1
    return _dice_hard_coe(gt_tc, pd_tc)


def brats_et_metric(y_true, y_pred):
    # enhancing tumor
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    gt_et = tf.cast(tf.identity(y_true), tf.int32)
    gt_et = tf.where(tf.equal(1, gt_et), 0 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 1] = 0
    gt_et = tf.where(tf.equal(2, gt_et), 0 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 2] = 0
    gt_et = tf.where(tf.equal(3, gt_et), 1 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 3] = 1
    pd_et = tf.cast(tf.round(tf.identity(y_pred)), tf.int32)
    pd_et = tf.where(tf.equal(1, pd_et), 0 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 1] = 0
    pd_et = tf.where(tf.equal(2, pd_et), 0 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 2] = 0
    pd_et = tf.where(tf.equal(3, pd_et), 1 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 3] = 1
    return _dice_hard_coe(gt_et, pd_et)


def _dice_hard_coe(target, output, smooth=1e-5):
    output = tf.cast(output, dtype=tf.float32)
    target = tf.cast(target, dtype=tf.float32)

    inse = tf.reduce_sum(tf.multiply(output, target))
    l = tf.reduce_sum(output)
    r = tf.reduce_sum(target)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    return tf.reduce_mean(hard_dice)