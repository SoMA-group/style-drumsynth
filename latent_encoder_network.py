#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:45:42 2021

@author: jake
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Input, UpSampling1D ,Embedding, Concatenate
from tensorflow.keras.layers import Dense, Reshape, LeakyReLU, Flatten, Conv1D, add, Cropping1D
from tensorflow.keras import backend as K

import numpy as np
import math
# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

#leakyRELU <<<<<<

# to do, fix error with inconsequential noise when using scale 4 instead of 2


class Networks(object):
    def __init__(self, latent_size=64, 
                  start_size=16, kernel_len=16, 
                  n_filters=1024, n_classes=16, n_chan=1, num_res=10, scale_base=2, 
                  embedding_dim=50, mapping_size=3):
        

        #param
        self._latent_size = latent_size
        self._label_in_dim = embedding_dim
        self._mapping_size = mapping_size
        # self._mapping_filters = mapping_filters

        #wav/params
        self._start_size = start_size  
        self._n_filters = n_filters
        self._n_classes = n_classes
        self._n_chan = n_chan
        self._kernel_len = kernel_len
        self._num_res = num_res
        self._scale_base = scale_base
        self._end_size = self._start_size*np.power(self._scale_base, self._num_res)

        
        self._L = self.latent_encoder()

        

    
    def sample_norm(self, x):
        """
        Fix multiply or divide?
        """
        return x / tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-7


    
    def d_block(self, x, filters, kernel_len, stride, act='relu'):
        # res = Conv1D(filters, 1)(x)
        
        # # convolve and add residual skip
        # x = Conv1D(filters, kernel_len, 1, padding = 'same')(x)
        # x = Activation(act)(x)
        # x = add([res, x])        
        
        # downsample
        x = Conv1D(filters, kernel_len, stride, padding = 'same')(x)
        x = Activation(act)(x)

        x = self.apply_phaseshuffle(x)
        return x


    def apply_phaseshuffle(self, x, rad=2, pad_type='reflect'):
        b, x_len, nch = x.get_shape().as_list()
        phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
        pad_l = tf.maximum(phase, 0)
        pad_r = tf.maximum(-phase, 0)
        phase_start = pad_r
        x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)
        x = x[:, phase_start:phase_start+x_len]
        x.set_shape([b, x_len, nch])  
        return x

    def num_filters(self, block_id, fmap_decay=1.0, fmap_max=4096):
        fmap_base = self._n_filters
        return int(min(fmap_base / math.pow(2.0, block_id * fmap_decay), fmap_max))
    
    def latent_encoder(self):
  
        audio_input = Input([self._end_size, self._n_chan])
        
        # concat label as a channel
        x = audio_input

        for i in range(self._num_res-1, 0, -1):
            x = self.d_block(x, 
                              self.num_filters(i), 
                              self._kernel_len, 
                              self._scale_base,
                              act=LeakyReLU(alpha=0.2))

        x = Conv1D(self.num_filters(0), self._kernel_len, self._scale_base, padding = 'same')(x)
        x = Activation(LeakyReLU(alpha=0.2))(x)  
        x = Flatten()(x)
            
        class_output = Dense(self._latent_size)(x)
            
        latent_encoder = Model(inputs = [audio_input], outputs = class_output)
        
        return latent_encoder
        
    
    def model_summary(self):
        self._G.summary()
        self._L.summary()
