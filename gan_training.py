#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
@author:Yoann Poupart

File containing an example of GAN training.

"""

# Libraries imports
from dcgan import FLEXI_DCGAN

import scipy.stats as sc
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.utils import shuffle

# Class definition

# Functions definition

# Constants definition
BLOCK_SIZE = 90
SAMPLING_SIZE = 156
DATA_SET_SIZE = 41472
FILE = './Samples/real_ds_dist_pluiv_glob.npy'

FILE_GEN_STEP = "./Networks/gan_dist_pluiv_step_gen"
FILE_DISC_STEP = "./Networks/gan_dist_pluiv_step_disc"

NUM_EPOCHS = 1  # 50_000
BATCH_SIZE = 400
INIT_LR = 4e-4
LAST_ACTI = "linear"
H = 10
FACT_LR = 1
GRAD_NORM = 1

TYPE = 'student'
SHAPE = 5
SIGMA_NOISE_DATA = 1e-5

SIMU_GAN = True
SIMU_DIST = True
SIMU_TREND = False
SAVE_STEPS = False
SAVE_FIG = False
SAVE_TEX = False
FS = 30

if __name__ == '__main__':
    if SIMU_GAN:
        X = np.load(FILE)

        if SAMPLING_SIZE == 165:
            print("[INFO] building generator...")
            gen = FLEXI_DCGAN.build_generator(w_end=SAMPLING_SIZE, w_start=SAMPLING_SIZE,
                                              w_1=110, s_1=1, w_2=150, final_act=LAST_ACTI, h=H)
            print("[INFO] building discriminator...")
            disc = FLEXI_DCGAN.build_discriminator(w_start=SAMPLING_SIZE,
                                                   w_1=150, w_2=90, w_3=15, h=H)
        elif SAMPLING_SIZE == 156:
            print("[INFO] building generator...")
            gen = FLEXI_DCGAN.build_generator(w_end=SAMPLING_SIZE, w_start=SAMPLING_SIZE,
                                              w_1=100, s_1=1, w_2=140, final_act=LAST_ACTI, h=H)
            print("[INFO] building discriminator...")
            disc = FLEXI_DCGAN.build_discriminator(w_start=SAMPLING_SIZE,
                                                   w_1=140, w_2=80, w_3=15, h=H)
        else:
            raise NotImplementedError
        try:
            gen = load_model(FILE_GEN_STEP, compile=False)
            disc = load_model(FILE_DISC_STEP, compile=False)
        except:
            pass
        print('[INFO] test...')
        discOpt = Adam(learning_rate=INIT_LR, beta_1=0.999, decay=INIT_LR / NUM_EPOCHS, clipnorm=GRAD_NORM)
        disc.compile(loss="binary_crossentropy", optimizer=discOpt)

        zeros = np.zeros((BATCH_SIZE, SAMPLING_SIZE))
        _ = gen.predict(zeros, verbose=1)
        _ = disc.predict(zeros, verbose=1)
        print(f"[INFO] discriminator summary : ")
        disc.summary()
        print(f"[INFO] generator summary : ")
        gen.summary()

        print("[INFO] building GAN...")
        disc.trainable = False
        ganInput = Input(shape=(SAMPLING_SIZE))
        ganOutput = disc(gen(ganInput))
        gan = Model(ganInput, ganOutput)
        ganOpt = Adam(learning_rate=INIT_LR / 4, beta_1=0.5, decay=INIT_LR / 4 / NUM_EPOCHS, clipnorm=GRAD_NORM)
        gan.compile(loss="binary_crossentropy", optimizer=discOpt)

        print("[INFO] starting training...")
        benchmark_noise = np.random.normal(size=(1, SAMPLING_SIZE))
        disc_losses = []
        gen_losses = []

        for epoch in range(0, NUM_EPOCHS):
            print("[INFO] starting epoch {} of {}...".format(epoch + 1,
                                                             NUM_EPOCHS))
            batchesPerEpoch = int(X.shape[0] / BATCH_SIZE)
            for i in range(0, batchesPerEpoch):
                batch_data = X[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :] + np.random.normal(loc=SIGMA_NOISE_DATA, size=(
                    BATCH_SIZE, SAMPLING_SIZE))
                noise = np.random.normal(size=(BATCH_SIZE, SAMPLING_SIZE))
                gen_data = gen.predict(noise, verbose=0)

                X_tr = np.concatenate((batch_data, gen_data))
                y_tr = np.array(([1] * BATCH_SIZE) + ([0] * BATCH_SIZE))
                (X_tr, y_tr) = shuffle(X_tr, y_tr)
                disc_loss = disc.train_on_batch(X_tr, y_tr)
                disc_losses.append(disc_loss)

                noise = np.random.normal(size=(BATCH_SIZE, SAMPLING_SIZE))
                fake_labels = np.array([1] * BATCH_SIZE)
                gen_loss = gan.train_on_batch(noise, fake_labels)
                gen_losses.append(gen_loss)

            if NUM_EPOCHS >= 10:
                if epoch % (NUM_EPOCHS // 10) == 0:
                    if SIMU_DIST:
                        gen_sample = gen.predict(benchmark_noise, verbose=0)
                        plt.subplot(2, 5, epoch // (NUM_EPOCHS // 10) + 1)
                        plt.hist(gen_sample.flatten(), bins=100)
                    if SIMU_TREND:
                        gen_sample = gen.predict(benchmark_noise, verbose=0)
                        plt.subplot(2, 5, epoch // (NUM_EPOCHS // 10) + 1)
                        plt.plot(gen_sample.mean(axis=0))
            if SAVE_STEPS:
                gen.save(FILE_GEN_STEP, include_optimizer=True)
                disc.save(FILE_DISC_STEP, include_optimizer=True)

        if SAVE_FIG:
            if SAVE_TEX:
                tikzplotlib.save("./Images/states.tex")
            else:
                plt.savefig("./Images/States.png")
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(15, 15))
        steps = 1 + np.arange(len(gen_losses))
        plt.plot(steps, gen_losses, label="Loss of the generator")
        plt.plot(steps, disc_losses, label="Loss of the discriminator")
        plt.xticks(fontsize=FS)
        plt.yticks(fontsize=FS)
        plt.xlabel("Steps", fontsize=FS)
        plt.ylabel(r"Loss", fontsize=FS)
        plt.legend(loc='best', fontsize=FS)
        if SAVE_FIG:
            if SAVE_TEX:
                tikzplotlib.save("Images/Loss.tex")
            else:
                plt.savefig("./Images/Loss.png")
            plt.close()
        else:
            plt.show()
