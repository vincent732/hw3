# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os, os.path
import numpy as np
import pickle
from tqdm import tqdm


def find_max_element_index(input_data):
    import collections
    length = len(input_data)
    if isinstance(input_data, collections.Iterable):
        return max(range(length), key=lambda i: input_data[i])
    else:
        raise TypeError

def cosine_similarity(a, b):
    from numpy import dot
    from numpy.linalg import norm

    return dot(a, b) / (norm(a) * norm(b))


def euclidean_distance(a, b):
    return np.linalg.norm((a - b))


def read_input(filepath):
    raw = pickle.load(open(filepath, 'rb'))
    if 'labels' in raw:
        return np.array(raw['labels']), np.array(raw['data'])
    else:
        return np.array(raw['data'])


# input should be 3 * 32 * 32 with height * width * channel
def process_image(x, num_to_produce):
    datagen = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        zca_whitening=True,
        fill_mode='nearest',
        data_format='channels_first')
    path = './tmp_images/'
    i = 1
    output = []
    x = x.reshape((1,) + x.shape)
    datagen.fit(x)
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=path, save_prefix='tmp', save_format='jpeg'):
        if i >= num_to_produce:
            break  # otherwise the generator would loop indefinitely
        i = i+1
    for f in os.listdir(path):
        img_path = os.path.join(path, f)
        img = load_img(img_path)
        x = img_to_array(img, data_format='channels_first') # should be (3, 32, 32)
        x = np.reshape(x, 3072)
        output.append(x)
        os.remove(img_path)
    return np.array(output)


def augment_img(path_of_pickle):
    raw_y, raw_x = read_input(path_of_pickle)
    reshape_raw_x = np.reshape(raw_x, (-1, 3, 32, 32))

    for index in tqdm(range(len(reshape_raw_x))):
        generated_img_arrays = process_image(reshape_raw_x[index], 9)
        raw_x = np.concatenate((raw_x, generated_img_arrays), axis=0)
        label = np.array([raw_y[index] for i in range(generated_img_arrays.shape[0])])
        raw_y = np.concatenate((raw_y, label), axis=0)

    output_dict = dict({'data':raw_x, 'labels':raw_y})
    output_path = 'datas/augmented_img.p'
    with open(output_path, 'wb') as handle:
        print("Save output at %s " % output_path)
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    augment_img('./datas/all_label.p')
