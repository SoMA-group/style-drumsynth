#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 21:42:09 2021

@author: jake drysdale
"""
#imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import functools
import numpy as np
import threading
import itertools
import tqdm
from natsort import natsorted
import re
import sys
import time

#local imports
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import data
import module
from preproc import Preproc


def main(args):
    # =========================================================================
    # set up
    # =========================================================================
    
    # output_dir
    if args.experiment_name == 'none':
        args.experiment_name = '%s_%s' % (args.experiment_name, 
                                          args.adversarial_loss_mode)
        if args.gradient_penalty_mode != 'none':
            args.experiment_name += '_%s' % args.gradient_penalty_mode
    output_dir = py.join('output', args.experiment_name)
    py.mkdir(output_dir)
    
    # save settings
    py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)
    
    
    # =========================================================================
    # data loading and preprocessing
    # =========================================================================
    
    if args.preproc == True:
        preproc = Preproc(size=args.audio_length)
        preproc.preproc(args.dataset)
        path = args.dataset+'/preproc'
    else:
        path = args.dataset
    

    audio_paths = py.glob(path, '*.wav')    
    audio_paths = natsorted(audio_paths)
    labels=([int(re.findall(r'\d+', os.path.basename(audio_paths[i]))[1]) 
             for i, x in enumerate(audio_paths)])
    
    
    dataset, shape, len_dataset = data.make_custom_dataset(audio_paths, 
                                                           args.batch_size,
                                                           labels,
                                                           resize=args.audio_length,
                                                           shuffle=True)
    
    
    # =========================================================================
    # model building
    # =========================================================================
    networks = module.Networks(latent_size=args.z_dim, 
                               num_res=args.num_res, 
                               n_classes=args.n_classes,
                               n_chan=args.channels,
                               n_filters=args.n_filters,
                               scale_base=args.scale_base,
                               embedding_dim=50,
                               mapping_size=args.n_mapping)
    
    G = networks._G
    D = networks._D
    G_s = networks._G_synth
    S = networks._style_net
    # G.summary()
    # D.summary()
    
    
    # loss functions
    d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
    
    G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
    D_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
    
    # constant input for generator
    constant_input = np.ones((args.batch_size, 1), dtype=np.float32)
    
    
    # =========================================================================
    # training functions
    # =========================================================================
    
    def layer_noise(num):
        threshold = np.int32(np.random.uniform(0.0, 5, size = [num]))
        n1 = tf.random.normal(shape=(num, args.z_dim))
        n2 = tf.random.normal(shape=(num, args.z_dim))
        
        n = []
        for i in range(args.num_res-1):
            n.append([])
            for j in range(num):
                if i < threshold[j]:
                    n[i].append(n1[j])
                else:
                    n[i].append(n2[j])
            n[i] = tf.convert_to_tensor(n[i])
        return n
    
    
    @tf.function
    def train_G():
        with tf.GradientTape() as t:
    
            n = layer_noise(args.batch_size)
    
            labels = tf.random.uniform(shape=[args.batch_size],
                                       minval=0, 
                                       maxval=args.n_classes, 
                                       dtype=tf.int64)
            
            w = [S([x, labels]) for x in n]
            
            inc_noise = np.random.uniform(0.0, 1.0,
                                          size = [args.batch_size, 
                                                  args.audio_length, 
                                                  args.channels])
                
    
            
            x_fake = G(w + [constant_input, inc_noise], training=True)
            
            x_fake_d_logit = D([x_fake, labels], training=True)
            G_loss = g_loss_fn(x_fake_d_logit)
    
        G_grad = t.gradient(G_loss, G_s.trainable_variables)
        G_optimizer.apply_gradients(zip(G_grad, G_s.trainable_variables))
    
        return {'g_loss': G_loss}
    
    
    @tf.function
    def train_D(x_real, y_real):
        with tf.GradientTape() as t:
            
            n = layer_noise(args.batch_size)
            w = [S([x, y_real]) for x in n]
            
            inc_noise = np.random.uniform(0.0, 1.0,
                                          size = [args.batch_size, 
                                                  args.audio_length, 
                                                  args.channels])
                
            x_fake = G(w + [constant_input, inc_noise], training=True)
            
            x_real_d_logit = D([x_real, y_real], training=True)
            x_fake_d_logit = D([x_fake, y_real], training=True)
    
            x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, 
                                                     x_fake_d_logit)
            
            gp = gan.gradient_penalty(functools.partial(D, training=True), 
                                      x_real, 
                                      x_fake, 
                                      y_real, 
                                      mode=args.gradient_penalty_mode)
    
            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight
    
        D_grad = t.gradient(D_loss, D.trainable_variables)
        D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))
    
        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}
    
    
    @tf.function
    def sample(styles, ones, inc_noise):
        return G(styles + [ones, inc_noise], training=False)
    
    
    # =========================================================================
    # training procedure
    # =========================================================================
    
    # epoch counter
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    
    
    # checkpoint
    checkpoint = tl.Checkpoint(dict(G=G,
                                    D=D,
                                    S=S,
                                    G_s=G_s,
                                    G_optimizer=G_optimizer,
                                    D_optimizer=D_optimizer,
                                    ep_cnt=ep_cnt),
                               py.join(output_dir, 'checkpoints'),
                               max_to_keep=5)
    try:  # restore checkpoint including the epoch counter
        checkpoint.restore().assert_existing_objects_matched()
    except Exception as e:
        print(e)
    
    
    # summary
    train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 
                                                                 'summaries', 
                                                                 'train'))

    # sample
    sample_dir = py.join(output_dir, 'samples_training')
    py.mkdir(sample_dir)
    
    
    # # main loop
    # z = tf.random.normal((args.n_classes, args.z_dim))
    # labels = tf.random.uniform(shape=[args.n_classes],
    #                            minval=0, maxval=args.n_classes, dtype=tf.int64)
    
    
    const_save = np.ones((args.n_classes, 1), dtype=np.float32)
    inc_n_save = np.random.uniform(0.0, 0.5,
                                   size = [args.n_classes, 
                                           args.audio_length, args.channels])
    
    
    n_save = layer_noise(args.n_classes)
    
    
    labels_save=np.array(list(range(args.n_classes)))
    
    
    all_labels_save=[]
    for i in range(args.num_res-1):
        all_labels_save.append(labels_save)
    
    
    w_save=[]
    for i in range(len(n_save)):
        w_save.append(S([n_save[i], all_labels_save[i]]))
    
    
    # =========================================================================
    # main training loop
    # =========================================================================

    with train_summary_writer.as_default():
        for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
            if ep < ep_cnt:
                continue
    
            # update epoch counter
            ep_cnt.assign_add(1)
    
            # train for an epoch
            for x_real in tqdm.tqdm(dataset, 
                                    desc='Batch', 
                                    total=len_dataset):
                
                D_loss_dict = train_D(x_real[0], x_real[1])
                tl.summary(D_loss_dict, 
                           step=D_optimizer.iterations, 
                           name='D_losses')
    
                if D_optimizer.iterations.numpy() % args.n_d == 0:
                    G_loss_dict = train_G()
                    tl.summary(G_loss_dict, 
                               step=G_optimizer.iterations, 
                               name='G_losses')
    
                # sample audio
                if G_optimizer.iterations.numpy() % 20 == 0:
                
                    x_fake = sample(w_save, const_save, inc_n_save)
                    for i in range(len(x_fake)):
                        pcm = tf.audio.encode_wav(x_fake[i], args.sr)
                        tf.io.write_file(py.join(sample_dir, 
                                                  'iter-%09d_class_' % G_optimizer.iterations.numpy()+str(i)+'.wav'), pcm)
    
                    
    
            # save checkpoint
            checkpoint.save(ep)


def get_args():
    #dataset arguments
    py.arg('-d', '--dataset', type=str, default='/home/your-name/Documents/data/training_aug' )
    py.arg('-p', '--preproc', type=bool, default=False)
    py.arg('-cl', '--n_classes', type=int, default=3)
    py.arg('--sr', type=int, default=44100)
    py.arg('-len','--audio_length', type=int, default=16384)
    
    #training arguments
    py.arg('-z', '--z_dim', type=int, default=128)
    py.arg('-ch', '--channels', type=int, default=1)
    py.arg('-b', '--batch_size', type=int, default=64)
    py.arg('-e', '--epochs', type=int, default=10000)
    py.arg('--lr', type=float, default=0.0002)
    py.arg('--beta_1', type=float, default=0.5)
    py.arg('--n_d', type=int, default=5)
    
    #model arguments
    py.arg('--e_dim', type=int, default=50)
    py.arg('--num_res', type=int, default=10)
    py.arg('--scale_base', type=int, default=2)
    py.arg('--n_filters', type=int, default=2048)
    py.arg('--n_mapping', type=int, default=6)
    py.arg('--adversarial_loss_mode', default='wgan', choices=['wgan'])
    py.arg('--gradient_penalty_mode', default='wgan-gp', choices=['wgan-gp'])
    py.arg('--gradient_penalty_weight', type=float, default=10.0)
    py.arg('--experiment_name', default='ISMIR_LBD_DEMO')

    return(py.args()) 


if __name__ == "__main__":
    args = get_args()
    main(args)

