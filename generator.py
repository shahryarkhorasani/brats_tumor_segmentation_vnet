import os
import numpy as np
import pandas as pd
import nibabel as nib
import keras 

def padwidth(wid):
    wid = np.max((0, wid))
    wid /= 2
    return int(np.ceil(wid)), int(np.floor(wid))


def cropwidth(wid):
    wid = np.min((0, wid))
    wid = np.abs(wid)
    wid /= 2
    return int(np.ceil(wid)), int(np.floor(wid))


def padcrop(img, dim):
    '''
    pads or crops a rescaled scan to given target dimensions x,y,z
    '''
    new_img = np.zeros(dim)
    target_dim = np.array(dim)
    difs = target_dim - np.array(img.shape)
    cropped_img = None
    if np.any(difs < 0):
        crop_x = cropwidth(difs[0])
        crop_y = cropwidth(difs[1])
        crop_z = cropwidth(difs[2])
        cropped_img = img[crop_x[0]:(img.shape[0] - crop_x[1]), crop_y[0]:(img.shape[1] - crop_y[1]),
                      crop_z[0]:(img.shape[2] - crop_z[1])]
        # print(cropped_img.shape)
    else:
        cropped_img = img
    if np.any(difs > 0):
        new_img[:, :, :] = np.pad(cropped_img, (padwidth(difs[0]), padwidth(difs[1]), padwidth(difs[2])),
                                  mode='constant')
    else:
        new_img[:, :, :] = cropped_img
    return new_img


class BRATS_DataGenerator(keras.utils.Sequence):
    'Generates data BRATS scans for Keras with a flip augmentation option'
    def __init__(self, patient_IDs, batch_size=1, dim=(192,192,192), shuffle=False, n_channels=4, augment_flip=False, seed=2020, n_labels=3, segment=True, structure=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.patient_IDs = patient_IDs                 #os.listdir('/mnt/dsets/brats/train/')
        self.segment = segment
        self.augment_flip = augment_flip
        self.seed = seed
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.n_labels = n_labels
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.patient_IDs) / self.batch_size))
    

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.patient_IDs[k] for k in indexes]
        

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.patient_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_labels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
                
            np.random.seed = self.seed
            flip_prob = np.random.uniform()
            if self.augment_flip == False:
                flip_prob = 0
            
            t1 = nib.load('/mnt/dsets/brats/train/' + ID + '/' + ID + '_t1.nii.gz').get_data()
            t1 =  padcrop(t1, self.dim) 
            t1ce = nib.load('/mnt/dsets/brats/train/' + ID + '/' + ID + '_t1ce.nii.gz').get_data()
            t1ce = padcrop(t1ce, self.dim)
            t2 = nib.load('/mnt/dsets/brats/train/' + ID + '/' + ID + '_t2.nii.gz').get_data()
            t2 = padcrop(t2, self.dim)
            flair = nib.load('/mnt/dsets/brats/train/' + ID + '/' + ID + '_flair.nii.gz').get_data()
            flair = padcrop(flair, self.dim)            
            scan = np.stack((t1, t1ce, t2, flair))
            
            if flip_prob >= .5:
                scan = scan[:,::-1,:,:]
            X[i,] = np.moveaxis(scan, 0, -1) 
            
            if self.segment:
                seg = nib.load('/mnt/dsets/brats/train/' + ID + '/' + ID + '_seg.nii.gz').get_data().astype(np.int)
                seg = padcrop(seg, self.dim)   
                seg1 = np.isin(seg, 1.)
                seg2 = np.isin(seg, 2.)
                seg4 = np.isin(seg, 4.)
                segs = np.stack((seg1, seg2, seg4))
                segs = segs.astype(np.int8)
                
                if flip_prob >= .5:
                    segs = segs[:,::-1,:,:]
                y[i,] = np.moveaxis(segs, 0, -1)
            
            else:    
                y[i] = X[i,] #if segment is None target is reconsruction
            
            
        return X, y