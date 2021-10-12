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

        
        #network building
        self._G = self.make_conditional_generator(self._latent_size)
        self._D = self.make_conditional_discriminator()  
        self._L = self.latent_encoder()
        # self.S = None
        self._style_net = self.style_network()
        self._G_synth = self.generator_synth()
        # self.G_synth.set_weights(self._G.get_weights(),by_name=True) 
        # self._S = self.style_network(self._latent_size)
        
    def load_g_weights(self):
        self._G.save_weights('generator_weights.model')
        self.G_synth.load_weights('generator_weights.model', by_name=True, skip_mismatch=True)

    def Adain(self, x):
        mean = K.mean(x[0], axis=[1], keepdims=True)
        stddev = K.std(x[0], axis=[1], keepdims=True) + 1e-7
        norm = (x[0] - mean) / stddev
        
        pool_shape = [-1, 1, norm.shape[-1]]
        gamma = K.reshape(x[1], pool_shape)
        beta = K.reshape(x[2], pool_shape)
        return norm * gamma + beta
    
    def sample_norm(self, x):
        """
        Fix multiply or divide?
        """
        return x / tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-7


    def g_block(self, x, filters, kernel_len, stride, latent_vector, inc_noise, name, act='relu'):
        gamma = Dense(filters, name='g_gamma1_block_'+name)(latent_vector)
        beta = Dense(filters, name='g_beta1_block_'+name)(latent_vector)
        noise = Conv1D(filters, kernel_size=1, padding='same', name='g_inc_noise_conv1_block_'+name)(inc_noise)
        
        x = UpSampling1D(stride, name = 'g_up_samp_block_'+name)(x)
        x = Conv1D(filters, kernel_len, padding = 'same', name='g_conv1_block_'+name)(x)
        x = add([x, noise])
        x = self.Adain([x, gamma, beta])
        x = Activation(act)(x)
        
        gamma = Dense(filters, name='g_gamma2_block_'+name)(latent_vector)
        beta = Dense(filters, name='g_beta2_block_'+name)(latent_vector)
        noise = Conv1D(filters, kernel_size=1, padding='same', name='g_inc_noise_conv2_block_'+name)(inc_noise)
        
        x = Conv1D(filters, kernel_len, padding = 'same', name='g_conv2_block_'+name)(x)
        x = add([x, noise])
        x = self.Adain([x, gamma, beta])
        x = Activation(act)(x)
        
        # x = self.sample_norm(x)
        return x


    # def g_block(self, x, filters, kernel_len, stride, latent_vector, inc_noise, act='relu'):
    #     gamma = Dense(filters)(latent_vector)
    #     beta = Dense(filters)(latent_vector)
        
    #     noise = Conv1D(filters, kernel_size=1, padding='same')(inc_noise)
        
    #     x = UpSampling1D(stride)(x)
    #     x = Conv1D(filters, kernel_len, padding = 'same')(x)
        
    #     x = self.Adain([x, gamma, beta])
    #     x = Activation(act)(x)
    #     # x = self.sample_norm(x)
    #     return x
    
    
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
    
    def style_network(self, name='style'):

        
        inp_style = Input(shape = [self._latent_size], name= 'style_in_mapping')

        in_label = Input(shape=(), name='s_label_input')
        label_in = Embedding(self._n_classes, self._label_in_dim)(in_label)
        label_in = Dense(self._latent_size, name='g_label_dense')(label_in)   
        
        conc_latent = Concatenate()([inp_style, label_in])

        norm_style = self.sample_norm(conc_latent)
        style = Dense(self._latent_size, 
                      kernel_initializer = 'he_normal', 
                      bias_initializer = 'zeros', 
                      name='g_style_dense1_'+name)(norm_style)
        
        style = LeakyReLU(0.01)(style)
    
        for i in range(self._mapping_size-2):
            sty = Dense(self._latent_size, 
                        kernel_initializer = 'he_normal', 
                        bias_initializer = 'zeros',
                        name = 'g_style_dense'+str(i+2)+'_'+name)(style)
            sty = LeakyReLU(0.01)(sty)
        
        sty = Dense(self._latent_size, 
                    kernel_initializer = 'he_normal', 
                    bias_initializer = 'zeros',
                    name = 'g_style_final_dense_'+name)(sty)
        sty = LeakyReLU(0.01)(sty)
        style_net = Model(inputs = [inp_style, in_label], outputs = sty)
        return style_net
        


    def make_conditional_generator(self, latent_size):
        
        # # style mapping
        # self.S = Sequential()
        # self.S.add(Dense(self._latent_size, input_shape=[self._latent_size]))
        # self.S.add(LeakyReLU(0.2))
        # for i in range(self._mapping_size-2):
        #     self.S.add(Dense(self._latent_size))
        #     self.S.add(LeakyReLU(0.2))
        # self.S.add(Dense(self._latent_size))
        # self.S.add(LeakyReLU(0.2))

        
        # # label input
        # in_label = Input(shape=(), name='g_label_input')
        # label_in = Embedding(self._n_classes, self._label_in_dim)(in_label)
        # n_nodes = self._start_size
        # label_in = Dense(n_nodes, name='g_label_dense')(label_in)     
        # label_in = Reshape((n_nodes, self._n_chan))(label_in)
        
        # inconsequential noise
        inc_noise = Input([self._end_size, self._n_chan], name='g_inc_n_input')
        noise = [Activation('linear')(inc_noise)]
        noise_size = self._end_size
        for i in range(1, self._num_res):
            noise_size=int(noise_size/self._scale_base)
            noise.append(Cropping1D(cropping=int(noise_size/self._scale_base))(noise[-1]))
            
                
        # constant input
        const_inp = Input((1,), name='g_const_input')
        x = Dense(self._start_size*self.num_filters(0), activation = 'relu', name = 'g_const_dense')(const_inp)
        x = Reshape([self._start_size,self.num_filters(0)])(x)
        x = Activation('relu')(x)
        # x = Concatenate()([x, label_in])
        
        
        # layer styles
        layer_style = []
        for i in range(1, self._num_res):
            layer_style.append(Input(shape = [latent_size], name = 'g_style_layer'+str(i)))


            
        # styles=[]
        # for i in range(len(layer_style)):
        #     styles.append(self.style_network(layer_style[i], name=str(i)))
            
        
        # generator blocks
        for i in range(1, self._num_res):
            x = self.g_block(x, 
                             self.num_filters(i), 
                             self._kernel_len, 
                             stride=self._scale_base,
                             latent_vector=layer_style[i-1],
                             inc_noise=noise[-i],
                             name=str(i))

        
        x = UpSampling1D(self._scale_base, name='g_final_upsamp')(x)
        x = Conv1D(self._n_chan, self._kernel_len, padding = 'same', name='g_final_conv')(x)
        image_output = Activation('tanh')(x)


        generator = Model(inputs = layer_style + [const_inp, inc_noise], outputs = image_output)
        
        return generator

    def generator_synth(self):
        
        in_label = Input(shape=())
        
        layer_style=[]
        styles=[]
        for i in range(1, self._num_res):
            layer_style.append(Input(shape = [self._latent_size]))
            styles.append(self._style_net([layer_style[i-1], in_label]))
        
        inc_noise= Input([self._end_size, self._n_chan])
        const_inp = Input((1,))
        
        generation = self._G(styles + [const_inp, inc_noise])
        
        
        gen_synth = Model(inputs = layer_style + [in_label, const_inp, inc_noise], outputs = generation)
        
        return gen_synth




    def make_conditional_discriminator(self):
        in_label = Input(shape=())
        
        # embedding for label
        label_in = Embedding(self._n_classes, self._label_in_dim)(in_label)
        
        # scale up to dimensions
        label_in = Dense(self._end_size)(label_in)
        
        # reshape to additional channel
        label_in = Reshape((self._end_size, self._n_chan))(label_in)
        
        audio_input = Input([self._end_size, self._n_chan])
        
        # concat label as a channel
        x = Concatenate()([audio_input, label_in])

        for i in range(self._num_res-1, 0, -1):
            x = self.d_block(x, 
                              self.num_filters(i), 
                              self._kernel_len, 
                              self._scale_base,
                              act=LeakyReLU(alpha=0.2))

        x = Conv1D(self.num_filters(0), self._kernel_len, self._scale_base, padding = 'same')(x)
        x = Activation(LeakyReLU(alpha=0.2))(x)  
        x = Flatten()(x)
            
        class_output = Dense(1)(x)
            
        discriminator = Model(inputs = [audio_input, in_label], outputs = class_output)
        
        return discriminator

    def latent_encoder(self):
        in_label = Input(shape=())
        
        # embedding for label
        label_in = Embedding(self._n_classes, self._label_in_dim)(in_label)
        
        # scale up to dimensions
        label_in = Dense(self._end_size)(label_in)
        
        # reshape to additional channel
        label_in = Reshape((self._end_size, self._n_chan))(label_in)
        
        audio_input = Input([self._end_size, self._n_chan])
        
        # concat label as a channel
        x = Concatenate()([audio_input, label_in])

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
            
        latent_encoder = Model(inputs = [audio_input, in_label], outputs = class_output)
        
        return latent_encoder
        
    
    def model_summary(self):
        self._G.summary()
        self._D.summary()



# nets=Networks()
# nets.model_summary()





# import math
# def num_filters(block_id, fmap_base=4096, fmap_decay=1.0, fmap_max=4096):
#   """Computes number of filters of block `block_id`."""
#   return int(min(fmap_base / math.pow(2.0, block_id * fmap_decay), fmap_max))

# # def num_filters(block_id, fmap_base=256, fmap_decay=1.0):
# #   """Computes number of filters of block `block_id`."""
# #   return int((fmap_base / math.pow(2.0, block_id * fmap_decay)))#




# # dim_mul = 1
# # label input
# in_label = Input(shape=())

# # embedding for categorical input
# label_in = Embedding(3, 50)(in_label)

# # scale up to image dimensions with linear activation
# label_in = Dense(65536)(label_in)

# # reshape to additional channel
# label_in = Reshape((self._end_size, 1))(label_in)