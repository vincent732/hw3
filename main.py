# -*- coding: utf-8 -*-

from utils import read_input
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from Classifier import Classifier, build_model
from pandas import DataFrame

def unit_test(x_train, y_train, nb_iter=1):
    test_size = 0.2
    random_state = 15
    cv = StratifiedShuffleSplit(y_train, nb_iter, test_size=test_size, random_state=random_state)
    scores = cross_val_score(Classifier(), X=x_train, y=y_train, scoring='accuracy', cv=cv)
    return scores

def hyperparameter_optim(X_train, y_train, params, nb_iter=10, cv=3):
    clf = RandomizedSearchCV(estimator=Classifier(),
                             param_distributions=params,
                             n_iter=nb_iter,
                             cv=cv,
                             scoring='accuracy')
    clf.fit(X_train, y_train)

    print("Best parameters set found:")
    print(clf.best_params_)
    print()
    print("Grid scores:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
               % (mean, std * 2, params))
    print()

    return clf

# np array of np array with shape 3072
def img_transform(input):
    # reshape to 32 * 32 * 3
    input = np.reshape(input, [-1, 3, 32, 32])
    # normalize
    return input.astype(np.float32) / 255.

# used to visualize the input img
def visualize_vector(x, y, n_sample_to_visual=7):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    from matplotlib import pyplot
    from scipy.misc import toimage
    # get distinct 10 classes of images
    pyplot.figure(figsize=[num_classes*2, n_sample_to_visual*2])
    for i, cls in enumerate(classes):
        idxs = [j for j, val in enumerate(y) if val == i][:n_sample_to_visual]
        images = x[idxs]
        images = np.reshape(images, (-1, 3, 32, 32))
        for k, img in enumerate(images):
            count = k * num_classes + i + 1
            pyplot.subplot(n_sample_to_visual, num_classes, count)
            pyplot.imshow(toimage(img))
            pyplot.axis('off')
    pyplot.show()


def generate_result(weight_path):
    model = build_model(None)
    model.load_weights(weight_path)
    raw_x = read_input('./datas/test.p')
    result = model.predict_classes(img_transform(raw_x))
    from pandas import DataFrame
    output_dict = {"ID":range(raw_x.shape[0]), "class":result}
    DataFrame(output_dict).to_csv('./datas/result.csv', index=None)


if __name__ == "__main__":
    raw_y, raw_x = read_input('./datas/all_label.p')
    cls = Classifier()
    cls.fit(raw_x, raw_y)
    test_x = read_input('./datas/test.p')
    output_dict = {"ID": range(test_x.shape[0]), "class": cls.predict(test_x)}
    DataFrame(output_dict).to_csv('./datas/result.csv', index=None)
