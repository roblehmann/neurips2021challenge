#!/usr/bin/env python
# 
# Autoencoder with concatenated latent space for joint embedding 
# keras implementation
# 

import os
import csv
import sys
from pathlib import Path
import argparse
import time
import collections
import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy

from keras.optimizers import adam_v2
from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from numpy.random import seed

import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

## VIASH START

# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.

dataset_path = 'output/datasets_phase2/joint_embedding/openproblems_bmmc_multiome_phase2/openproblems_bmmc_multiome_phase2.censor_dataset.output_'

par = {
    'input_mod1': dataset_path + 'mod1.h5ad',
    'input_mod2': dataset_path + 'mod2.h5ad',
    'output': 'output.h5ad',
}

## VIASH END


#####################
method_id = "lsl"

##############################################
# parameters
##############################################

inMod1 = par['input_mod1']
inMod2 = par['input_mod2']
output = par['output']

mod1encoderDim = [64]
mod2encoderDim = [64]
mod1_outputLayer_act = 'relu'
mod2_outputLayer_act = 'relu'
latent_dim = 64
dropout = 0.1
learningRate = 0.0001
batchSize = 32
epochs = 600
validation_split = 0.2

ATAC_peak_minFracOfCells = 3
filterHVG = True
refBatchWeight = None

##############################################
# reproducibility settings

# seed(1) # numpy seed
# set_random_seed(2) # tensorflow seed

##############################################
# IO
##############################################
def load_modality(path, atac_cell_frac = 5, filterHVG = False):
    # load h5ad input file
    # param atac_cell_frac: fraction of cells the peak needs to have signal in to be retained
    
    logging.info("loading input file: %s", path)
    input_data = ad.read_h5ad(path)

    if(input_data.var["feature_types"][1] == "GEX"):
        # check if log transformed, otherwise do that
        if np.max(input_data.X) > 1000:
            logging.info("Data seems not log-transformed, applying log1p ...")
            input_data.X = np.log1p(input_data.X)

        # calc HVG 
        if filterHVG:
            logging.info('filtering for HVG')
            scanpy.pp.highly_variable_genes(input_data, 
                batch_key = 'batch', 
                layer = 'counts', 
                flavor='seurat', 
                subset = True)
            logging.info('retaining %i HVG ...', input_data.X.shape[1])
        input_data.Xproc = input_data.X
        return input_data

    if(input_data.var["feature_types"][1] == "ATAC"):
        # if ATAC data, filter peaks down
        # select peaks that have signal in at least atac_cell_frac% of cells
        nCellsCutoff =  input_data.X.shape[0]/100 * atac_cell_frac
        scanpy.pp.filter_genes(input_data, 
            min_cells = nCellsCutoff,
            inplace=True)
        # binarize signal
        input_data.Xproc = input_data.X
        input_data.Xproc[input_data.Xproc > 1] = 1
        logging.info('retaining %i peaks...', input_data.Xproc.shape[1])
        return input_data

    if(input_data.var["feature_types"][1] == "ADT"):
        # for now just use data directly
        input_data.Xproc = input_data.X
        return input_data


inMod1_ad = load_modality(inMod1, 
    atac_cell_frac = ATAC_peak_minFracOfCells, 
    filterHVG = filterHVG)
inMod2_ad = load_modality(inMod2, 
    atac_cell_frac = ATAC_peak_minFracOfCells, 
    filterHVG = filterHVG)

ncol_mod1 = inMod1_ad.Xproc.shape[1]
ncol_mod2 = inMod2_ad.Xproc.shape[1]

# make weight vector for reference batch vs. batches to integrate
# find biggest batch as reference
if not refBatchWeight is None:
    batchname, _ = collections.Counter(inMod1_ad.obs['batch']).most_common(1)[0]
    sample_weight = np.ones(shape=(inMod1_ad.obs.shape[0],))
    sample_weight[inMod1_ad.obs['batch'] == batchname] = refBatchWeight
else:
    sample_weight = None

###############################################################################
# Orthogonal weight constraint for encoder layers
class WeightsOrthogonalityConstraint(Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        self.idMat = K.eye(self.encoding_dim)

    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - self.idMat #K.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(m)))
        else:
            m = K.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)

###############################################################################
# Model setup
###############################################################################
logging.info('Generating model...')

###############################################################################
# Input Layers
# Mod1 input
input_layer_mod1 = Input(shape = (ncol_mod1, ), name = "mod1")
dropout_mod1 = Dropout(dropout, name = "Dropout_mod1")(input_layer_mod1)
encoded_mod1 = Dense(
    mod1encoderDim[0], 
    activation = 'relu', 
    name = "Encoder_mod1",
    use_bias=True, 
    kernel_regularizer=WeightsOrthogonalityConstraint(64, weightage=1., axis=0))(dropout_mod1)


# Mod2 input
input_layer_mod2 = Input(shape = (ncol_mod2, ), name = "mod2")
dropout_mod2 = Dropout(dropout, name = "Dropout_mod2")(input_layer_mod2)
encoded_mod2 = Dense(
    mod2encoderDim[0], 
    activation = 'relu', 
    name = "Encoder_mod2",
    use_bias=True, 
    kernel_regularizer=WeightsOrthogonalityConstraint(64, weightage=1., axis=0))(dropout_mod2)

# Merging Encoder layers from different OMICs
merge = concatenate([encoded_mod1,  encoded_mod2])

# Bottleneck compression
bottleneck = Dense(
    latent_dim, 
    kernel_initializer = 'uniform', 
    activation = 'linear', 
    name = "Bottleneck")(merge)

#Inverse merging
merge_inverse = Dense(
    mod1encoderDim[0] + mod2encoderDim[0], 
    activation = 'relu', 
    name = "Concatenate_Inverse")(bottleneck)

###############################################################################
# Decoder layer for each OMIC
decoded_mod1 = Dense(
    ncol_mod1, 
    activation = mod1_outputLayer_act, 
    name = "Decoder_mod1")(merge_inverse)
decoded_mod2 = Dense(
    ncol_mod2, 
    activation = mod2_outputLayer_act, 
    name = "Decoder_mod2")(merge_inverse)

###############################################################################
# Construct model
autoencoder = Model(
    inputs = [input_layer_mod1, input_layer_mod2], 
    outputs = [decoded_mod1, decoded_mod2])

opt = adam_v2.Adam(learning_rate=learningRate)

autoencoder.compile(optimizer = opt, 
    loss={
        'Decoder_mod1': 'mean_squared_error', 
        'Decoder_mod2': 'mean_squared_error', })

autoencoder.summary()

###############################################################################
# Training
es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    verbose=1, 
    patience=20)

logging.info('Training...')
start_time = time.time()
estimator = autoencoder.fit(
    [inMod1_ad.Xproc.toarray(), inMod2_ad.Xproc.toarray()], 
    [inMod1_ad.Xproc.toarray(), inMod2_ad.Xproc.toarray()], 
    sample_weight = sample_weight,
	epochs = epochs, 
    batch_size = batchSize, 
    validation_split = validation_split, 
    shuffle = True, 
    verbose = 1, 
	callbacks=[es])

elapsed = (time.time() - start_time)/60
logging.info(f'Total Training Time: %f min', elapsed)

logging.info("Training Loss: %f",	estimator.history['loss'][-1])
logging.info("Validation Loss: %f",	estimator.history['val_loss'][-1])

# Encoder model
encoder = Model(
    inputs = [input_layer_mod1, input_layer_mod2], 
    outputs = bottleneck)
bottleneck_representation = encoder.predict(
    [inMod1_ad.Xproc.toarray(), inMod2_ad.Xproc.toarray()])

###############################################################################
# Result Persistence
###############################################################################
logging.info('writing latent space to file')
latent_adata = ad.AnnData(X=bottleneck_representation,
                obs=inMod1_ad.obs,
                uns = {
                    'dataset_id': inMod1_ad.uns['dataset_id'],
                    'method_id': method_id,
    })
latent_adata.write_h5ad(output)
