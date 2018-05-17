#! /usr/bin/python
# -*- coding: utf8 -*-


import os, sys
import numpy as np
import tensorflow as tf
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model


def residual_block(layer_input, name=None):
    """Residual block described in paper"""
    with tf.name_scope(name):
        d = Conv2D(64, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(64, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
    return d


def deconv2d(layer_input, name=None):
    """Layers used during upsampling"""
    with tf.name_scope(name):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
    return u


def d_block(layer_input, filters, strides=1, bn=True, name=None):
    """Discriminator layer"""
    with tf.name_scope(name):
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
    return d


def srgan_g(lr_shape, n_residual_blocks=16, n_upsampling=2, is_train=True):
    """Returning Model of Converter from LR to HR"""
    img_lr = Input(shape=lr_shape, name="srgan_g_input")
    # Pre-residual block
    with tf.name_scope("srgan_g_pre-residual_block"):
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)
    # Residual block sequence
    r = residual_block(c1, name="srgan_g_residual_block1")
    for idx in range(n_residual_blocks - 1):
        r = residual_block(r, name="srgan_g_residual_block"+str(idx+2))
    # Post-residual block
    with tf.name_scope("srgan_g_post-residual_block"):
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])
    # Upsampling
    u1 = deconv2d(c2, name="srgan_g_deconv1")
    u2 = deconv2d(u1, name="srgan_g_deconv2")
    #Output High Resolution Images
    gen_hr = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh', name="srgan_g_output")(u2)
    g_model = Model(img_lr, gen_hr, name="srgan_generator")
    g_model.trainable = is_train
    return g_model


def srgan_d(hr_shape, df=64, is_train=True): # hr_shape = lr_shape * 4
    """Returning Model of Discriminator"""
    # Input img
    d0 = Input(shape=hr_shape, name="srgan_d_input")
    # Block sequence
    d1 = d_block(d0, self.df, bn=False, name="srgan_d_block1")
    d2 = d_block(d1, self.df, strides=2, name="srgan_d_block2")
    d3 = d_block(d2, self.df*2, name="srgan_d_block3")
    d4 = d_block(d3, self.df*2, strides=2, name="srgan_d_block4")
    d5 = d_block(d4, self.df*4, name="srgan_d_block5")
    d6 = d_block(d5, self.df*4, strides=2, name="srgan_d_block6")
    d7 = d_block(d6, self.df*8, name="srgan_d_block7")
    d8 = d_block(d7, self.df*8, strides=2, name="srgan_d_block8")
    d9 = Dense(self.df*16, name="srgan_d_block9")(d8)
    d10 = LeakyReLU(alpha=0.2, name="srgan_d_block10")(d9)
    validity = Dense(1, activation='sigmoid', name="srgan_d_output")(d10)

    d_model = Model(d0, validity, name="srgan_discriminator")
    d_model.trainable = is_train
    return d_model