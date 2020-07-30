import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from keras.optimizers import RMSprop
from data_loading import BRATS_DataGenerator
from losses import *
from keras.models import load_model
from fast_mri_view import *
# read csv file from directory
brats = pd.read_csv('./survival_data.csv')
all_subjects = brats.BraTS18ID
# choose the part of the data that model has never seen:
evaluation_subjects = all_subjects[130:]
gen_eval = BRATS_DataGenerator(list(evaluation_subjects), n_labels=4, augment_flip=False)
#load the trained model from directory
loaded_model = load_model('./vnet_brats_2_fold.h5', compile=False)
#compile model using dice loss
loaded_model.compile(optimizer=RMSprop(lr=0), loss=generalised_dice_loss_3D, metrics=['accuracy', brats_wt_metric, brats_tc_metric, brats_et_metric])
#evaluate model
evaluations = loaded_model.evaluate_generator(gen_eval, workers=8, use_multiprocessing=True, verbose=1)
# order of the evaluation output:  total dice loss, total accuracy, whole tumor, tumor core, enhancing tumor
print(evaluations)