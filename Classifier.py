from keras.initializers import he_normal, Zeros
from keras.layers import MaxPooling2D, Flatten, Dense, Conv2D, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn.base import BaseEstimator
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard

class Classifier(BaseEstimator):

    def preprocess(self, X):
        X = np.reshape(X, [-1, 3, 32, 32])
        # normalize
        return X.astype(np.float32) / 255.

    def preprocess_y(self, y):
        return np_utils.to_categorical(y)

    def fit(self, X, y):
        tensor_board = TensorBoard(log_dir='./tmp/',
                                   write_images=True)
        check_point = ModelCheckpoint('./tmp/best_weight.hdf5',
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='max')
        X = self.preprocess(X)
        y = self.preprocess_y(y)

        hyper_parameters = None

        print("FIT PARAMS : ")
        print(hyper_parameters)

        self.model = build_model(hyper_parameters)

        self.model.fit(X, y,
                       epochs=100,
                       batch_size=32,
                       verbose=1,
                       callbacks=[check_point, tensor_board],
                       validation_split=0.2,
                       validation_data=None,
                       shuffle=True)
        self.model.save('./models/v1.h5')
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.model.predict_classes(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.evaluate(self, X, y, verbose=1, sample_weight=None)

def build_model(hp):
    num_classes = 10
    model = Sequential()
    weight_initializer = he_normal(seed=None) # use relu weight initializer
    model.add(Conv2D(32,
                     (3, 3),
                     input_shape=(3, 32, 32),
                     padding='same',
                     activation='relu',
                     data_format='channels_first',
                     kernel_regularizer=l2()))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # compile
    epochs = 100
    lrate = 0.002
    optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=lrate/epochs)
    #SGD(lr=lrate, momentum=0.9, decay=(lrate / epochs), nesterov=False)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
