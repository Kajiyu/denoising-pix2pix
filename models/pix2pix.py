#! /usr/bin/python
# -*- coding: utf8 -*-


import numpy as np
import tensorflow as tf
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model


def unet_conv2d(layer_input, filters, f_size=4, bn=True, name=None):
    """Layers used during downsampling"""
    with tf.name_scope(name):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
    return d


def unet_deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0, name=None):
    """Layers used during upsampling"""
    with tf.name_scope(name):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
    return u


def d_layer(layer_input, filters, f_size=4, bn=True, name=None):
    """Discriminator layer"""
    with tf.name_scope(name):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
    return d


def pix2pix_g(img_shape=(256, 256, 3), is_train=True, gf=64):
    """Returning Model of UNet-Generator"""
    # w_init = tf.random_normal_initializer(stddev=0.02)
    # b_init = None  # tf.constant_initializer(value=0.0)
    # g_init = tf.random_normal_initializer(1., 0.02)
    d0 = Input(shape=img_shape, name="pix2pix_g_input")
    # Encoding
    d1 = unet_conv2d(d0, gf, bn=False, name="pix2pix_g_e1")
    d2 = unet_conv2d(d1, gf*2, name="pix2pix_g_e2")
    d3 = unet_conv2d(d2, gf*4, name="pix2pix_g_e3")
    d4 = unet_conv2d(d3, gf*8, name="pix2pix_g_e4")
    d5 = unet_conv2d(d4, gf*8, name="pix2pix_g_e5")
    d6 = unet_conv2d(d5, gf*8, name="pix2pix_g_e6")
    d7 = unet_conv2d(d6, gf*8, name="pix2pix_g_e7")

    # Decoding
    u1 = unet_deconv2d(d7, d6, gf*8, name="pix2pix_g_d1")
    u2 = unet_deconv2d(u1, d5, gf*8, name="pix2pix_g_d2")
    u3 = unet_deconv2d(u2, d4, gf*8, name="pix2pix_g_d3")
    u4 = unet_deconv2d(u3, d3, gf*4, name="pix2pix_g_d4")
    u5 = unet_deconv2d(u4, d2, gf*2, name="pix2pix_g_d5")
    u6 = unet_deconv2d(u5, d1, gf, name="pix2pix_g_d6")

    u7 = UpSampling2D(size=2, name="pix2pix_g_d7")(u6)
    output_img = Conv2D(img_shape[-1], kernel_size=4, strides=1, padding='same', activation='tanh', name="pix2pix_g_output")(u7)
    g_model = Model(d0, output_img, name="pix2pix_generator")
    g_model.trainable = is_train
    return g_model


def pix2pix_d(img_shape=(256, 256, 3), is_train=True, df=64):
    """Returning Model of Discriminator"""
    img_A = Input(img_shape, name="pix2pix_d_input_a") # Predicted Image
    img_B = Input(img_shape, name="pix2pix_d_input_b") # Target Image
    combined_imgs = Concatenate(axis=-1, name="pix2pix_d_input_merged")([img_A, img_B]) # Merge Input

    d1 = d_layer(combined_imgs, df, bn=False, name="pix2pix_d_l1")
    d2 = d_layer(d1, df*2, name="pix2pix_d_l2")
    d3 = d_layer(d2, df*4, name="pix2pix_d_l3")
    d4 = d_layer(d3, df*8, name="pix2pix_d_l4")
    
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same', name="pix2pix_d_output")(d4)
    
    d_model = Model([img_A, img_B], validity, name="pix2pix_discriminator")
    d_model.trainable = is_train
    return d_model