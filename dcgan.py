#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
@author:Yoann Poupart

File to implement the general structure of the discriminator and the generator. Should the structure be changed, beware
of the heights and widths compatibilities which is not always handled for now.

"""

# Libraries imports
from keras.models import Sequential
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Conv1DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import LocallyConnected1D
from keras.utils.vis_utils import plot_model
import numpy as np


# Classes definition
class FLEXI_DCGAN:
    @staticmethod
    def build_generator(w_end, w_1=0, w_2=0, s_1=1, s_2=1, s_end=1, h=1, w_start=1_000, batch_norm=True,
                        final_act="linear"):
        model = Sequential(name="Generator")
        w_prev = w_start
        model.add(Reshape((w_prev, 1)))
        if w_1:
            model.add(Conv1D(h*5, w_prev - (w_1 - 1) * s_1, strides=s_1))
            model.add(Activation("relu"))
            if batch_norm:
                model.add(BatchNormalization())
            w_prev = w_1
        if w_2:
            if w_prev >= w_2:
                raise NotImplementedError('The generator has to be upscaling')
            model.add(Reshape((w_prev, h*5)))
            model.add(Conv1DTranspose(h, w_2 - (w_prev - 1) * s_2, strides=s_2))
            model.add(Activation("relu"))
            if batch_norm:
                model.add(BatchNormalization())
            w_prev = w_2

        if w_prev >= w_end:
            raise NotImplementedError('The generator has to be upscaling')
        model.add(Reshape((w_prev, h)))
        model.add(Conv1DTranspose(1, w_end - (w_prev - 1) * s_end, strides=s_end))
        model.add(Flatten())
        model.add(Activation(final_act))
        return model

    @staticmethod
    def build_discriminator(w_start, w_1=0, w_2=0, w_3=0, s_1=1, s_2=1, h=1, alpha=0.2, dropout=0.3, batch_norm=True):
        model = Sequential(name="Discriminator")
        w_prev = w_start
        model.add(Reshape((w_prev, 1)))
        if w_1:
            if w_prev <= w_1:
                raise NotImplementedError('The discriminator has to be downscaling')
            model.add(Conv1D(h, w_prev - (w_1 - 1) * s_1, strides=s_1))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(dropout))
            if batch_norm:
                model.add(BatchNormalization())
            w_prev = w_1
        if w_2:
            if w_prev <= w_2:
                raise NotImplementedError('The discriminator has to be downscaling')
            model.add(Reshape((w_prev, h)))
            model.add(Conv1D(h, w_prev - (w_2 - 1) * s_2, strides=s_2))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(dropout))
            if batch_norm:
                model.add(BatchNormalization())
            w_prev = w_2
        if w_3:
            if w_prev <= w_3:
                raise NotImplementedError('The discriminator has to be downscaling')
            model.add(Flatten())
            model.add(Dense(units=w_3))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(dropout, name="Dropout_3"))
            if batch_norm:
                model.add(BatchNormalization())
        model.add(Flatten(name="Flatten"))
        model.add(Dense(1, name="Final_dense"))
        model.add(Activation("sigmoid"))
        return model


class FLEXI_C_DCGAN:
    @staticmethod
    def build_generator(w_end, w_1=0, w_2=0, s_1=1, s_2=1, s_end=1, h=2, w_start=1_000, batch_norm=True,
                        final_act="linear", w_cond=2):
        in_coord = Input(shape=(w_cond,))
        li = Dense(w_start)(in_coord)
        li = Reshape((w_start, 1))(li)
        in_data = Input(shape=(w_start,))
        data = Reshape((w_start, 1))(in_data)
        in_prev = Concatenate()([data, li])

        w_prev = w_start
        if w_1:
            in_prev = Conv1D(h*5, w_prev - (w_1 - 1) * s_1, strides=s_1)(in_prev)
            in_prev = Activation("relu")(in_prev)
            if batch_norm:
                in_prev = BatchNormalization()(in_prev)
            w_prev = w_1
        if w_2:
            if w_prev >= w_2:
                raise NotImplementedError('The generator has to be upscaling')
            in_prev = Reshape((w_prev, h*5))(in_prev)
            in_prev = Conv1DTranspose(h, w_2 - (w_prev - 1) * s_2, strides=s_2)(in_prev)
            in_prev = Activation("relu")(in_prev)
            if batch_norm:
                in_prev = BatchNormalization()(in_prev)
            w_prev = w_2

        if w_prev >= w_end:
            raise NotImplementedError('The generator has to be upscaling')
        in_prev = Reshape((w_prev, h))(in_prev)
        in_prev = Conv1DTranspose(1, w_end - (w_prev - 1) * s_end, strides=s_end)(in_prev)
        in_prev = Flatten()(in_prev)
        out_layer = Activation(final_act)(in_prev)
        return Model([in_data, in_coord], out_layer, name="Generator")

    @staticmethod
    def build_discriminator(w_start, w_1=0, w_2=0, w_3=0, s_1=1, s_2=1, h=2, alpha=0.2, dropout=0.3, batch_norm=True,
                            w_cond=2):
        in_coord = Input(shape=(w_cond,))
        li = Dense(w_start)(in_coord)
        li = Reshape((w_start, 1))(li)
        in_data = Input(shape=(w_start,))
        data = Reshape((w_start, 1))(in_data)
        in_prev = Concatenate()([data, li])

        w_prev = w_start
        if w_1:
            if w_prev <= w_1:
                raise NotImplementedError('The discriminator has to be downscaling')
            in_prev = Conv1D(h, w_prev - (w_1 - 1) * s_1, strides=s_1)(in_prev)
            in_prev = LeakyReLU(alpha=alpha)(in_prev)
            in_prev = Dropout(dropout)(in_prev)
            if batch_norm:
                in_prev = BatchNormalization()(in_prev)
            w_prev = w_1
        if w_2:
            if w_prev <= w_2:
                raise NotImplementedError('The discriminator has to be downscaling')
            in_prev = Reshape((w_prev, h))(in_prev)
            in_prev = Conv1D(h, w_prev - (w_2 - 1) * s_2, strides=s_2)(in_prev)
            in_prev = LeakyReLU(alpha=alpha)(in_prev)
            in_prev = Dropout(dropout)(in_prev)
            if batch_norm:
                in_prev = BatchNormalization()(in_prev)
            w_prev = w_2
        if w_3:
            if w_prev <= w_3:
                raise NotImplementedError('The discriminator has to be downscaling')
            in_prev = Flatten()(in_prev)
            in_prev = Dense(units=w_3)(in_prev)
            in_prev = LeakyReLU(alpha=alpha)(in_prev)
            in_prev = Dropout(dropout, name="Dropout_3")(in_prev)
            if batch_norm:
                in_prev = BatchNormalization()(in_prev)
        in_prev = Flatten(name="Flatten")(in_prev)
        in_prev = Dense(1, name="Final_dense")(in_prev)
        out_layer = Activation("sigmoid")(in_prev)

        return Model([in_data, in_coord], out_layer, name="Discriminator")


# Functions definition

# Constants definition
SAMPLING_SIZE = 165

BATCH_SIZE = 100
H = 10
W_COND = 2
LAST_ACTI = 'linear'

TEST_FLEXI_DCGAN = True
TEST_FLEXI_C_DCGAN = False

if __name__ == '__main__':
    if TEST_FLEXI_DCGAN:
        print("[INFO] building generator...")
        gen = FLEXI_DCGAN.build_generator(w_end=SAMPLING_SIZE, w_start=SAMPLING_SIZE,
                                          w_1=110, s_1=1, w_2=150, final_act=LAST_ACTI, h=H)
        print("[INFO] building discriminator...")
        disc = FLEXI_DCGAN.build_discriminator(w_start=SAMPLING_SIZE,
                                               w_1=150, w_2=90, w_3=15, h=H)
        print('[INFO] test...')
        noise = np.random.uniform(size=(BATCH_SIZE, SAMPLING_SIZE))
        zeros = np.zeros((BATCH_SIZE, SAMPLING_SIZE))
        gen_data = gen.predict(noise, verbose=1)
        pred_bias = disc.predict(zeros, verbose=1)
        print(gen_data.shape, pred_bias.shape)
        print(f"[INFO] discriminator summary : ")
        disc.summary()
        print(f"[INFO] generator summary : ")
        gen.summary()

        plot_model(disc, to_file='./disc.svg', show_shapes=True,
                   show_dtype=False,
                   show_layer_names=True,
                   rankdir="LR",
                   expand_nested=False,
                   dpi=None,
                   )
        plot_model(gen, to_file='./gen.svg', show_shapes=True,
                   show_dtype=False,
                   show_layer_names=True,
                   rankdir="LR",
                   expand_nested=False,
                   dpi=None,
                   )

    if TEST_FLEXI_C_DCGAN:
        print("[INFO] building generator...")
        gen = FLEXI_C_DCGAN.build_generator(w_end=SAMPLING_SIZE, w_start=SAMPLING_SIZE,
                                            w_1=110, s_1=1, w_2=150, final_act=LAST_ACTI, h=H, w_cond=W_COND)
        print("[INFO] building discriminator...")
        disc = FLEXI_C_DCGAN.build_discriminator(w_start=SAMPLING_SIZE,
                                                 w_1=150, w_2=90, w_3=15, h=H, w_cond=W_COND)
        print('[INFO] test...')
        noise = np.random.uniform(size=(BATCH_SIZE, SAMPLING_SIZE))
        zeros = np.zeros((BATCH_SIZE, SAMPLING_SIZE))
        zeros_lab = np.zeros((BATCH_SIZE, 2))
        gen_data = gen.predict((noise, zeros_lab), verbose=1)
        pred_bias = disc.predict((zeros, zeros_lab), verbose=1)
        print(gen_data.shape, pred_bias.shape)
        print(f"[INFO] discriminator summary : ")
        disc.summary()
        print(f"[INFO] generator summary : ")
        gen.summary()

        plot_model(disc, to_file='./disc.svg', show_shapes=True,
                   show_dtype=False,
                   show_layer_names=True,
                   rankdir="LR",
                   expand_nested=False,
                   dpi=None,
                   )
        plot_model(gen, to_file='./gen.svg', show_shapes=True,
                   show_dtype=False,
                   show_layer_names=True,
                   rankdir="LR",
                   expand_nested=False,
                   dpi=None,
                   )
