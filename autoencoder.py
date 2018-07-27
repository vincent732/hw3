# -*- coding: utf-8 -*-

from keras.layers import Dense, Input
from keras.models import Model
from main import read_input
import numpy as np
from matplotlib import pyplot
from scipy.misc import toimage
from keras.callbacks import TensorBoard
from utils import euclidean_distance
from tqdm import tqdm
class AutoEncoder():

    def __init__(self, train, test):
        self.encoding_dim = 256
        self.train = train
        self.test = test
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    def start_train(self):
        input_img = Input(shape=(3072,))

        # first try with 256 neron in hidden layer
        encoded = Dense(1024, activation="relu",
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros')(input_img)
        encoded = Dense(512, activation='relu')(encoded)
        encoded = Dense(256, activation='relu')(encoded)

        decoded = Dense(512, activation='relu')(encoded)
        decoded = Dense(1024, activation='relu')(decoded)
        decoded = Dense(3072, activation='sigmoid')(decoded)

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.fit(
            self.train,
            self.train,
            epochs=200,
            batch_size=64,
            shuffle=True,
            validation_data=(self.test, self.test),
            callbacks=[TensorBoard(log_dir='./autoencoder')]
        )
        self.autoencoder.save('./model/deep_autoencoder.h5')
        # this model maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = Dense(512, activation='relu')(encoded_input)
        decoder_layer = Dense(1024, activation='relu')(decoder_layer)
        decoder_layer = Dense(3072, activation='sigmoid')(decoder_layer)
        # create the decoder model
        self.decoder = Model(encoded_input, decoder_layer)

    def predict(self, x):
        encoding_imgs = self.encoder.predict(x)
        decoding_imgs = self.decoder.predict(encoding_imgs)

        # visualize img
        x = np.reshape(x, [-1, 3, 32, 32])
        decoding_imgs = np.reshape(decoding_imgs, [-1, 3, 32, 32])
        pyplot.figure(figsize=[20, 4])
        num_images = x.shape[0]
        for i in range(num_images):
            # original
            pyplot.subplot(2, 10, i + 1)
            pyplot.imshow(toimage(x[i]))

            # predicted
            pyplot.subplot(2, 10, i + num_images + 1)
            pyplot.imshow(toimage(decoding_imgs[i]))
        pyplot.show()

    def assign_label(self):
        all_labeled_y, all_labeled_x = read_input('./datas/all_label.p')
        all_labeled_imgs = self.encoder.predict(all_labeled_x)
        all_unlabeled_x = read_input('./datas/all_unlabel.p')
        all_unlabeled_imgs = self.encoder.predict(all_unlabeled_x)
        index = -1
        for unlabeled_img in tqdm(all_unlabeled_imgs):
            index += 1
            target_img = all_unlabeled_x[index]
            target_img = target_img.reshape((1,) + target_img.shape)

            all_distance = [euclidean_distance(unlabeled_img, labeled_img) for labeled_img in all_labeled_imgs]
            index_of_max = min(range(len(all_distance)), key = lambda i: all_distance[i])
            assigned_label = all_labeled_y[index_of_max]
            all_labeled_x = np.concatenate((all_labeled_x, target_img), axis=0)
            all_labeled_y = np.concatenate((all_labeled_y, np.array([assigned_label])), axis=0)

            # # for debug
            # origin_img = all_labeled_x[index_of_max]
            # target_img = np.reshape(target_img, (3, 32, 32))
            # origin_img = np.reshape(origin_img, (3, 32, 32))
            # pyplot.figure(figsize=[4, 4])
            # pyplot.subplot(2, 2, 1)
            # pyplot.imshow(toimage(origin_img))
            # pyplot.subplot(2, 2, 2)
            # pyplot.imshow(toimage(target_img))
            # pyplot.show()

        output_dict = dict({'data': all_labeled_x, 'labels': all_labeled_y})
        output_path = 'datas/relabeled_img.p'
        import pickle
        with open(output_path, 'wb') as handle:
            print("Save output at %s " % output_path)
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    x = read_input('./datas/all_unlabel.p')
    np.random.shuffle(x)
    x_train, x_test = x[:-500], x[-500:]
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    autoencoder = AutoEncoder(x_train, x_test)
    autoencoder.start_train()
    autoencoder.predict(x_test[:10])
    autoencoder.assign_label()

