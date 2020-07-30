import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam, adadelta, RMSprop
import datetime
import time
from sklearn.model_selection import KFold
from losses import *

from data_loading import BRATS_DataGenerator
from architecture import vnet


model = vnet(192,192,192)
rand_weights = model.get_weights()
#read csv file from the directory:
brats = pd.read_csv('./survival_data.csv')
all_subjects = brats.BraTS18ID
#choosing the first 130 subjects for kfold, leaving the remaining for final evaluation
train_test_subjects = all_subjects[:130]


#Kfold
subject_list_splits = train_test_subjects
kf = KFold(n_splits=5, shuffle=True)
dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H%M%S')
#choose where you want to save the model
stemdir = './{}'.format(dt)
## training and evaluating on the k-folds
fold_number = 1
for train_index, test_index in kf.split(subject_list_splits):
    training_indices_splits, testing_indices_splits = subject_list_splits[train_index], subject_list_splits[test_index]
    #choosing (train, test) sets
    gen_train = BRATS_DataGenerator(list(training_indices_splits),n_labels=4, augment_flip=True, seed = 2020)
    gen_valid = BRATS_DataGenerator(list(testing_indices_splits), n_labels=4, augment_flip=False)
    #tracking the model
    csv_dir = stemdir + '/vnet_brats_{}_fold.csv'.format(fold_number)
    checkpoint_dir = stemdir + '/vnet_brats_{}_fold.h5'.format(fold_number)
    if not os.path.exists(stemdir):
        os.makedirs(stemdir)
    model_checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csv_dir)
    tb_logger = TensorBoard(log_dir=stemdir, write_grads=True, batch_size=1, update_freq='batch')
    
    model.compile(optimizer=RMSprop(lr=0.0000125, decay=0.0005), loss=generalised_dice_loss_3D, metrics=['accuracy', brats_wt_metric, brats_tc_metric, brats_et_metric])
    model.fit_generator(gen_train, validation_data=gen_valid, epochs=500, verbose=1, callbacks=[model_checkpoint, csv_logger, tb_logger], max_queue_size=10,
                        workers=8, use_multiprocessing=True)
    fold_number += 1
    model.set_weights(rand_weights)

