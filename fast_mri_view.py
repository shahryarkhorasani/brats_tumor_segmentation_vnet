import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def show_slices(scan, x=None, y=None, z=None, size=30): #scan: directory to a nifti file
    """ Function to display the middle slice in each axis of a scan """
    fig, axes = plt.subplots(1, 3, figsize=(size,size))
    if type(scan) == str: 
        scan = nib.load(scan)
        scan = scan.get_fdata()
    elif type(scan) == np.array:
        scan = scan
    if len(scan.shape) == 3:
        if x:
            slice_x = scan[x,:,:]
        else:    
            slice_x = scan[int(scan.shape[0]/2),:,:]
        if y:
            slice_y = scan[:,y,:]
        else:
            slice_y = scan[:,int(scan.shape[1]/2),:]
        if z:
            slice_z = scan[:,:,z]
        else:
            slice_z = scan[:,:,int(scan.shape[2]/2)]
        slices = [slice_x, slice_y, slice_z]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, origin="lower")
    elif len(scan.shape) == 4: 
        if x:
            slice_x = scan[x,:,:, 0]
        else:    
            slice_x = scan[int(scan.shape[0]/2),:,:, 0]
        if y:
            slice_y = scan[:,y,:, 0]
        else:
            slice_y = scan[:,int(scan.shape[1]/2),:, 0]
        if z:
            slice_z = scan[:,:,z, 0]
        else:
            slice_z = scan[:,:,int(scan.shape[2]/2), 0]
        slices = [slice_x, slice_y, slice_z]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, origin="lower")
            
def show_scan(scan, x=None, y=None, z=None, size=30, save_to=None): #scan: directory to a nifti file
    """ Function to display the middle slice in each axis of a scan """
    fig, axes = plt.subplots(1, 3, figsize=(size,size),sharey=True)
    if type(scan) == str: 
        scan = nib.load(scan)
        scan = scan.get_fdata()
    elif type(scan) == np.array:
        scan = scan
    if x:
        slice_x = scan[x,:,:].T
    else:    
        slice_x = scan[int(scan.shape[0]/2),:,:]
    if y:
        slice_y = scan[:,y,:]
    else:
        slice_y = scan[:,int(scan.shape[1]/2),:]
    if z:
        slice_z = scan[:,:,z]
    else:
        slice_z = scan[:,:,int(scan.shape[2]/2)]
    slices = [slice_x, slice_y, slice_z]
    
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, origin="lower")
        axes[i].axis('off')
        plt.tight_layout(pad=0)
        plt.subplots_adjust(wspace=0.0,hspace=0.0)
        if save_to:
            plt.savefig(fname=save_to, quality =90)
            
def show_mask(scan, mask, mask_valu=None, x=None, y=None, z=None, size=30, alpha=.2): 
    """ Function to display a scan and its mask - optional: specific mask_value and/or slices"""
    fig, axes = plt.subplots(1, 3, figsize=(size,size))
    #reading the file
    if type(scan) == str: 
        scan = nib.load(scan)
        scan = scan.get_fdata()
    elif type(scan) == np.array:
        scan = scan
        
    if type(mask) == str: 
        mask = nib.load(mask)
        mask = mask.get_fdata()
    elif type(mask) == np.array:
        mask = mask
    ###    
    #indexing the scan    
    if len(scan.shape) == 3:
        if x:
            slice_x = scan[x,:,:]
        else:    
            slice_x = scan[int(scan.shape[0]/2),:,:]
        if y:
            slice_y = scan[:,y,:]
        else:
            slice_y = scan[:,int(scan.shape[1]/2),:]
        if z:
            slice_z = scan[:,:,z]
        else:
            slice_z = scan[:,:,int(scan.shape[2]/2)]
        slices = [slice_x, slice_y, slice_z]

    elif len(scan.shape) == 4: 
        if x:
            slice_x = scan[x,:,:, 0]
        else:    
            slice_x = scan[int(scan.shape[0]/2),:,:, 0]
        if y:
            slice_y = scan[:,y,:, 0]
        else:
            slice_y = scan[:,int(scan.shape[1]/2),:, 0]
        if z:
            slice_z = scan[:,:,z, 0]
        else:
            slice_z = scan[:,:,int(scan.shape[2]/2), 0]
        slices = [slice_x, slice_y, slice_z]
        
    ###
    #selecting specific label
    if mask_valu:
        mask = np.isin(mask, mask_valu)
        mask = mask.astype(int)
    ###
    #indexing the mask    
    if len(mask.shape) == 3:
        if x:
            slice_x = mask[x,:,:]
        else:    
            slice_x = mask[int(mask.shape[0]/2),:,:]
        if y:
            slice_y = mask[:,y,:]
        else:
            slice_y = mask[:,int(mask.shape[1]/2),:]
        if z:
            slice_z = mask[:,:,z]
        else:
            slice_z = mask[:,:,int(mask.shape[2]/2)]
        slices_mask = [slice_x, slice_y, slice_z]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, origin="lower", cmap='gray')
        for i, slice in enumerate(slices_mask):
            axes[i].imshow(slice.T, origin="lower", cmap='gnuplot', alpha=alpha)
    elif len(mask.shape) == 4: 
        if x:
            slice_x = mask[x,:,:, 0]
        else:    
            slice_x = mask[int(mask.shape[0]/2),:,:, 0]
        if y:
            slice_y = mask[:,y,:, 0]
        else:
            slice_y = mask[:,int(mask.shape[1]/2),:, 0]
        if z:
            slice_z = mask[:,:,z, 0]
        else:
            slice_z = mask[:,:,int(mask.shape[2]/2), 0]
        slices_mask = [slice_x, slice_y, slice_z]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, origin="lower", cmap='gray')
        for i, slice in enumerate(slices_mask):
            axes[i].imshow(slice.T, origin="lower", cmap='gnuplot', alpha=alpha)
            