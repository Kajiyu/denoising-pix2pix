#! /usr/bin/python
# -*- coding: utf8 -*-

import scipy

from keras.datasets import mnist
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os, sys
import glob
import numpy as np
import tensorflow as tf
from models.pix2pix import pix2pix_g, pix2pix_d
import datetime
import matplotlib.pyplot as plt
from data_loader import Pix2PixDataLoader as DataLoader


def sample_images(epoch, batch_i, dataset_name, data_loader, generator):
    os.makedirs('images/%s' % dataset_name, exist_ok=True)
    r, c = 3, 3
    imgs_A, imgs_B = data_loader.load_data(batch_size=3, is_testing=True)
    fake_A = generator.predict(imgs_B)
    
    gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i))
    plt.close()


if __name__ == '__main__':
    img_rows = 256
    img_cols = 256
    channels = 3
    epochs = 1000
    batch_size=16
    sample_interval=200
    img_shape = (img_rows, img_cols, channels)
    dataset_name = 'edges2handbags'
    data_loader = DataLoader(dataset_name=dataset_name, img_res=(img_rows, img_cols))

    # Calculate output shape of D (PatchGAN)
    patch = int(img_rows / 2**4)
    disc_patch = (patch, patch, 1)

    # Number of filters in the first layer of G and D
    gf = 64
    df = 64

    optimizer = Adam(0.0002, 0.5)

    # Build and compile the discriminator
    discriminator = pix2pix_d(img_shape=img_shape)
    discriminator.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Build the generator
    generator = pix2pix_g(img_shape=img_shape)
    # Input images and their conditioning images
    img_A = Input(shape=img_shape)
    img_B = Input(shape=img_shape)
    # By conditioning on B generate a fake version of A
    fake_A = generator(img_B)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # Discriminators determines validity of translated images / condition pairs
    valid = discriminator([fake_A, img_B])

    combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
    combined.compile(
        loss=['mse', 'mae'],
        loss_weights=[1, 100],
        optimizer=optimizer
    )

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    print("Training Start ...")
    generator.save('./data/weights/pix2pix/'+dataset_name+'_generator.h5')
    discriminator.save('./data/weights/pix2pix/'+dataset_name+'_discriminator.h5')
    for epoch in range(epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
            # Condition on B and generate a translated version
            fake_A = generator.predict(imgs_B)
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generators
            g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" %
                (
                    epoch,
                    epochs,
                    batch_i,
                    data_loader.n_batches,
                    d_loss[0], 100*d_loss[1],
                    g_loss[0],
                    elapsed_time
                )
            )
            # If at save interval => save generated image samples
            # if batch_i % sample_interval == 0:
                # sample_images(epoch, batch_i, dataset_name, data_loader, generator)
        generator.save('./data/weights/pix2pix/'+dataset_name+'_generator.h5')
        discriminator.save('./data/weights/pix2pix/'+dataset_name+'_discriminator.h5')