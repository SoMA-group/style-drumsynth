#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 21:42:09 2021

@author: jake drysdale
"""


# =============================================================================
# STYLEGAN DRUM SYNTH
# =============================================================================
import os
import time
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools
import threading
from natsort import natsorted 
import numpy as np
import tensorflow as tf


#local imports
import module
import tf2lib as tl
import pylib as py
import latent_encoder_network


#fixed params
z_dim = 128
num_res = 10
n_classes = 3
channels = 1
n_filters = 2048
scale_base = 2
n_mapping = 6
lr = 0.0002
beta_1 =  0.5
audio_length=16384
sr=44100


#directories
generator_ckpt_dir = py.join(os.getcwd(),'checkpoints/generator')
encoder_ckpt_dir = py.join(os.getcwd(), 'checkpoints/encoder' )
directions_dir = py.join(os.getcwd(), 'fixed_settings/components_0-5.npy' )
fixed_w_dir = py.join(os.getcwd(),'fixed_settings/fixed_w.npy') 
save_dir= py.join(os.getcwd(), 'generations' )
encoder_dir = py.join(os.getcwd(),'input_audio') 




def load_networks():
    networks = module.Networks(latent_size=z_dim, 
                               num_res=num_res, 
                               n_classes=n_classes,
                               n_chan=channels,
                               n_filters=n_filters,
                               scale_base=scale_base,
                               embedding_dim=50,
                               mapping_size=n_mapping)
    
    latent_network = latent_encoder_network.Networks(latent_size=z_dim, 
                               num_res=num_res, 
                               n_classes=n_classes,
                               n_chan=channels,
                               n_filters=n_filters,
                               scale_base=scale_base,
                               embedding_dim=50,
                               mapping_size=n_mapping)
    
    C = latent_network._L
    
    G = networks._G
    S = networks._style_net
    
    
    # checkpoints for generator
    checkpoint = tl.Checkpoint(dict(G=G,
                                    S=S),
                               generator_ckpt_dir,
                               max_to_keep=5)
    
    checkpoint_encoder = tl.Checkpoint(dict(C=C),
                               encoder_ckpt_dir,
                               max_to_keep=5)
    
    try:  # restore checkpoint including the epoch counter
        checkpoint.restore().expect_partial()
        checkpoint_encoder.restore().expect_partial()
    except Exception as e:
        print(e)
    
    return G, S, C



def tf_load_audio(path, size=audio_length):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1,
                              desired_samples=size, name=None)
    return audio



def load_encoder_audio(encoder_dir):
    audio_dirs = natsorted([py.join(encoder_dir,x) for x in os.listdir(encoder_dir)])
    audio = [tf_load_audio(x) for x in audio_dirs]
    return tf.stack(audio,0)


def encode(audio_in, C):
    return C([audio_in], training=False)
    

def layer_noise(num):
    threshold = np.int32(np.random.uniform(0.0, 5, size = [num])) #check this num was 64
    n1 = tf.random.normal(shape=(num, z_dim))
    n2 = tf.random.normal(shape=(num, z_dim))
    
    n = []
    for i in range(num_res-1):
        n.append([])
        for j in range(num):
            if i < threshold[j]:
                n[i].append(n1[j])
            else:
                n[i].append(n2[j])
        n[i] = tf.convert_to_tensor(n[i])
    return n


def feature_slider(w, directions, component, amount):

    w_hat = np.clip(w+(directions[component]*amount), 0,1000)

    return w_hat


@tf.function
def sample(styles, ones, inc_noise, G):
    return G(styles + [ones, inc_noise], training=False)


def synthesize(cond, 
               cons_noise_amount, 
               component, 
               direction_slider, 
               amount_to_generate, 
               prinicpal_directions,
               G,
               S,
               C,
               encoder,
               randomize=True):    

    
    if randomize:
        test_noise = layer_noise(amount_to_generate)
        test_labels = np.ones(amount_to_generate)*cond
        w_test = S([test_noise[0], test_labels])
        
    if not randomize:
        test_noise = layer_noise(3)
        test_labels = np.array([0,1,2])
        w_test = tf.convert_to_tensor(np.load(fixed_w_dir))
    
    if encoder == True:
        audio_input = load_encoder_audio(encoder_dir)
        w_test = encode(audio_input, C)
    
    
    w_hat = feature_slider(w_test, prinicpal_directions, component, direction_slider)
    w_mod=[]
    for i in range(num_res-1):
        w_mod.append(w_hat)
    
    
    
    ones_ = np.ones((1, 1), dtype=np.float32)
    inc_n_ = np.random.uniform(0.0, cons_noise_amount,
                                   size = [1, audio_length, channels])
    


    return sample(w_mod, ones_, inc_n_, G)


def save_generations(generations, cond, component, direction_slider, encoder):
    if cond == 0:
        drum_type = 'kick'
    if cond == 1:
        drum_type = 'snare'
    if cond == 2:
        drum_type = 'hat'
    if encoder == True:
        drum_type = 'regeneration'
    for i in range(len(generations)):
    # write_wav(py.join(sample_dir, 'beat_iter-%09d.wav' % G_optimizer.iterations.numpy()),np.array(flat_list), 44100)
        # reshape = np.expand_dims(np.array(x_fake[i]), axis=1)

        pcm = tf.audio.encode_wav(generations[i,:,:], sr)
        tf.io.write_file(py.join(save_dir, 
                                  str(i)+'_generated_'+drum_type+'_direction-'+str(component)+'_amount-'+str(direction_slider)+'.wav' ), pcm)
        
    


def main(args, G, S, C):
    #args
    cond=args.condition
    component = args.direction
    direction_slider = args.direction_slider
    cons_noise_amount=args.stocastic_variation
    amount_to_generate=args.num_generations
    randomize = args.randomize
    
    prinicpal_directions = np.load(directions_dir)
    generated_drums = synthesize(cond, 
                                 cons_noise_amount, 
                                 component, 
                                 direction_slider, 
                                 amount_to_generate, 
                                 prinicpal_directions,
                                 G,
                                 S,
                                 C,
                                 args.encode,
                                 randomize=randomize)
    save_generations(generated_drums, cond, component, direction_slider, args.encode)
    
    tf.keras.backend.clear_session() #needed?
    
    
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rGenerating drum sounds ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

def get_args():
    # command line
    #TODO: add optional path to encode a drum sound
    py.arg('-c', '--condition', type=int, default=1, help='0: kick, 1: snare, 2:hat')
    py.arg('-d', '--direction', type=int, default=0, help='synthesis controls [0:4]')
    py.arg('-ds', '--direction_slider', type=int, default=5, help='how much to move in a particular direction')
    py.arg('-n', '--num_generations', type=int, default=3 ,help='number of examples to generate')
    py.arg('-v', '--stocastic_variation', type=float, default=0.6, help='amount of inconsequential noise injected')
    py.arg('-r', '--randomize', type=bool, default=True, help='if set to False, a fixed latent vector is used to generated a drum sound from each condition')
    py.arg('-e', '--encode', type=bool, default=False, help='regenerates drum sounds from encoder folder')
    return(py.args())  

if __name__ == "__main__":
    # animate()
    
    args = get_args()
    done = False
    t = threading.Thread(target=animate)
    t.start()
    G, S, C = load_networks()
    main(args, G, S, C)
    time.sleep(0.2)
    done = True
    # done = 'true'
    


