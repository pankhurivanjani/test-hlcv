#!/usr/bin/env python
# coding: utf-8

# # Train RAMS Deep Neural Network on Proba-V Dataset
# ![proba_v_dataset](media/rams_architecture.png "Logo Title Text 1")
# 
# The following notebook provides a script to train the residual attention network for multi-image super-resolution (RAMS). It makes use of the pre-processed dataset (train and validation) saved in the 'dataset' folder and using the main settings it selects a band to train with. 
# 
# **NB**: We strongly discouraged to run this notebook without an available GPU on the host machine. The original training (ckpt folder) has been performed on a 2080 Ti GPU card with 11GB of memory in approximately 24 hours.
# 
# **The notebook is divided in**:
# - 1.0 [Dataset Loading](#loading)
# - 2.0 [Dataset Pre-Processing](#preprocessing)
#     - 2.1 Make patches
#     - 2.2 Clarity patches check
#     - 2.3 Pre-augment dataset (temporal permutation)
# - 3.0 [Build the network](#network)
# - 4.0 [Train the network](#train)

#Akshay system: run this in coda hlcv


# import utils and basic libraries
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.preprocessing import gen_sub, bicubic
from utils.loss import l1_loss, psnr, ssim
from utils.network import RAMS
from utils.training import Trainer
from skimage import io
from zipfile import ZipFile
import pathlib

# gpu settings (we strongly discouraged to run this notebook without an available GPU)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

print(os.system("ls"))

exp_name = "RED_RAMS"


#-------------
# General Settings
#-------------
PATH_DATASET = "/home/pankhuri/Desktop/Project/RAMS-master_pankhuri/npy_files" #'../training_datasets/Holopix50k_burst/grayscale' # pre-processed dataset path
# PATH_DATASET = '/home/adv8/Study/Projects/hlcv2021/Project/training_datasets/Holopix50k_burst/grayscale' # pre-processed dataset path
name_net = 'RAMS' # name of the network
LR_SIZE = 32 # pathces dimension
SCALE = 3 # upscale of the proba-v dataset is 3
HR_SIZE = LR_SIZE * SCALE # upscale of the dataset is 3
OVERLAP = 32 # overlap between pathces
CLEAN_PATH_PX = 0.85 # percentage of clean pixels to accept a patch
# band = 'NIR' # choose the band for the training
checkpoint_dir = f'ckpt/{exp_name}' # weights path
log_dir = f'logs/{exp_name}' # tensorboard logs path
submission_dir = 'submission' # submission dir


#-------------
# Network Settings
#-------------
FILTERS = 32 # features map in the network
KERNEL_SIZE = 3 # convolutional kernel size dimension (either 3D and 2D)
CHANNELS = 9 # number of temporal steps
R = 8 # attention compression
N = 12 # number of residual feature attention blocks
lr = 1e-4 # learning rate (Nadam optimizer)
BATCH_SIZE = 32 # batch size
EPOCHS_N = 2 # number of epochs


# create logs folder
#if not os.path.exists(log_dir):
    #os.mkdir(log_dir)
pathlib.Path(checkpoint_dir).mkdir(parents=True,exist_ok=True)
pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
# load training dataset
X_train = np.load(os.path.join(PATH_DATASET, f'X_train.npy'))
y_train = np.load(os.path.join(PATH_DATASET, f'y_train.npy'))
y_train_mask = np.load(os.path.join(PATH_DATASET, f'y_train_masks.npy'))
# y_train_mask =np.ones(y_train.shape)


# load validation dataset
X_val = np.load(os.path.join(PATH_DATASET, f'X_val.npy'))
y_val = np.load(os.path.join(PATH_DATASET, f'y_val.npy'))
y_val_mask = np.load(os.path.join(PATH_DATASET, f'y_val_masks.npy'))
# y_val_mask =np.ones(y_val.shape)

# print loaded dataset info
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('y_train_mask: ', y_train_mask.shape)


print('X_val: ', X_val.shape)
print('y_val: ', y_val.shape)
print('y_val_mask: ', y_val_mask.shape)


# # 2.0 Dataset Pre-Processing

# ## 2.1 Make patches

# create patches for LR images
d = LR_SIZE  # 32x32 patches
s = OVERLAP  # overlapping patches
# Ex: n = (128-d)/s+1 = 7 -> 49 sub images from each image
print('X_train[...,0]', X_train[...,0].shape)
#X_train_patches = gen_sub(X_train,d,s)
X_train_patches = gen_sub(X_train,d,s)
print("X_train_patches")
print(X_train_patches.shape)
X_val_patches = gen_sub(X_val,d,s)
print("X_val_patches")
print(X_val_patches.shape)
#X_val_patches = gen_sub(X_val,d,s)


# create patches for HR images and masks
d = HR_SIZE  # 96x96 patches
s = OVERLAP * SCALE  # overlapping patches
# Ex: n = (384-d)/s+1 = 7 -> 49 sub images from each image

y_train_patches = gen_sub(y_train,d,s)
print("y_train_patches")
print(y_train_patches.shape)
y_train_mask_patches = gen_sub(y_train_mask,d,s)
print("y_train_mask_patches")
print(y_train_mask_patches.shape)
y_val_patches = gen_sub(y_val,d,s)
print("y_val_patches")
print(y_val_patches.shape)
y_val_mask_patches = gen_sub(y_val_mask,d,s)
print("y_val_mask_patches")
print(y_val_mask_patches.shape)

# print first patch and check if LR is in accordance with HR
fig, ax = plt.subplots(1,2, figsize=(10,10))
ax[0].imshow(X_train_patches[0,:,:,0], cmap = 'gray')
ax[1].imshow(y_train_patches[0,:,:,0], cmap = 'gray')


# free up memory
# del X_train, y_train, y_train_mask
del X_train, y_train

# del X_val, y_val, y_val_mask
del X_val, y_val


# build rams network
rams_network = RAMS(scale=SCALE, filters=FILTERS, 
                 kernel_size=KERNEL_SIZE, channels=CHANNELS, r=R, N=N)


# print architecture structure
rams_network.summary(line_length=120)

#case 1
#rams_network.trainable = False #for freezing all layers

#case 2
#rams_network.trainable = True #for allowing all weight to update

#case 3

#for fine-tuning the last few layers in the model
print('No of network layers:', len(rams_network.layers)) #number of layers in the model
fine_tune_at = len(rams_network.layers) - 0 #layer from which fine tuning to be started 
for layer in rams_network.layers[:fine_tune_at]:
  rams_network.trainable = False


trainer_rams = Trainer(rams_network, HR_SIZE, name_net,
                      loss=l1_loss,
                      metric=psnr,
                      optimizer=tf.keras.optimizers.Nadam(learning_rate=lr),
                      checkpoint_dir=os.path.join(checkpoint_dir),
                      log_dir=log_dir)



trainer_rams.fit(X_train_patches,
                [y_train_patches.astype('float32'), y_train_mask_patches], initial_epoch = 0,
                batch_size=BATCH_SIZE, evaluate_every=400, data_aug = True, epochs=EPOCHS_N,
                validation_data=(X_val_patches, [y_val_patches.astype('float32'), y_val_mask_patches]))

