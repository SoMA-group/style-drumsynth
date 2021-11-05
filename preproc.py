#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:20:44 2020

@author: Jake Drysdale
"""

# =============================================================================
# preprocessing
# =============================================================================

import numpy as np
import os
import librosa
import random
from natsort import natsorted

import tensorflow as tf





class Preproc(object):
    def __init__(self, size=16384, sr=44100, shuffle=False, filetype='.wav'):
        
        #arguments
        self._size = size
        self._sr = sr
        self._shuffle = shuffle
        self._filetype = filetype
    
    def load_audio(self, path):
        folders = natsorted([x for x in os.listdir(path) if not x.startswith('.')])
        
        flns=[]
        for i in range(len(folders)):
            flns.append([x for x in natsorted(os.listdir(path+'/'+folders[i])) if x.endswith(self._filetype)])
            
        audio=[]
        for i in range(len(flns)):
            audio.append([librosa.load(path+'/'+folders[i]+'/'+x, sr=self._sr)[0][:self._size] for x in flns[i]])
        return audio, flns


    def pad_audio(self, audio):
        padded_audio=[]
        for i in range(len(audio)):
            if len(audio[i]) < self._size:
                pad_size = self._size-len(audio[i])
                pad = np.zeros(pad_size)
                padded = np.append(audio[i],pad)
                padded_audio.append(padded)
            else: 
                padded_audio.append(audio[i])
        return padded_audio
    
    
    def fade_audio(self, audio, fade_r=0.2, fade_l=0.001):
        faded_audio=[]
        for i in range(len(audio)):
            fade_r_len = int(len(audio[i])*fade_r)
            fade_l_len = int(len(audio[i])*fade_l)
            
            xfade_r = np.hstack((np.ones(len(audio[i])-fade_r_len), np.linspace(1,0,fade_r_len)))
            xfade_l = np.hstack((np.linspace(0,1,fade_l_len), np.ones(len(audio[i])-fade_l_len)))
            
            fade=audio[i]*xfade_r
            faded_audio.append(fade*xfade_l)
            
        return faded_audio
    
    
    def shuffle_audio(self, audio):
        audio = random.sample(audio, len(audio))
        return audio
    

    def save_audio(self, audio, in_path, flns):#
        # if not os.path.exists(out_path+'/'+name):
        #     os.mkdir(out_path+'/'+name)
        os.mkdir(in_path+'/preproc')
        for i in range(len(audio)):
            for j in range(len(audio[i])):
                reshape = np.expand_dims(audio[i][j], axis=1)
                pcm = tf.audio.encode_wav(reshape, self._sr)
                tf.io.write_file(in_path+'/preproc/'+str(i)+'_drum_'+str(j)+self._filetype, 
                                 pcm)


    def preproc(self, in_path):
        audio, flns = self.load_audio(in_path)    
        audio = [self.pad_audio(x) for x in audio]
        audio = [self.fade_audio(x) for x in audio] 
        # audio = self.normalise_all(audio)
        if self._shuffle:
            audio = self.shuffle_audio(audio)
        self.save_audio(audio, in_path, flns)
        return audio
        

# in_dir = '/home/jake/Documents/data/dafx2020_data/in'
# # out_dir = '/home/jake/Documents/data/delete_plz/out' 

# preproc = Preproc(size=4096)
# audio = preproc.preproc(in_dir)

       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


# in_dir = '/home/jake/Documents/data/drum_generator_dataset/sliced/slices' 
# out_dir = '/home/jake/Documents/data/drum_generator_dataset/preprocess_4_gan' 







# def process_all(in_dir, out_dir):
#     sub_dirs = os.listdir(in_dir)
#     for i in range(len(sub_dirs)):
#         preproc.preproc(in_dir+'/'+sub_dirs[i], out_dir)

# in_dir = '/home/jake/Documents/data/DAFX_GAN_data/raw'
# out_dir = '/home/jake/Documents/data/DAFX_GAN_data/preprocessed' 

# process_all(in_dir, out_dir)

    




