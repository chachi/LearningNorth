#!/usr/bin/env python
import scipy.misc
from sklearn import linear_model
from skimage import color, io
import argparse
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import matplotlib.pyplot as plt


def load_dir(dir_path):
    """Load headings and image paths from the given directory.

    Returns array with headings and list of image filenames for this
    directory.

    """
    orientations_file = os.path.join(dir_path, 'orientation.csv')
    data = pd.read_csv(orientations_file)

    images = [os.path.join(dir_path, 'IMG_' + f + '.jpg')
              for f in data['timestamp'] if f != 'orientation.csv']
    return images, data['yaw']


def load_images_labels(dirs):
    """ Given training and test directories, run the learning algorithms
and evaluate their performance.
    """
    labels = np.array([])
    images = []
    for d in dirs:
        (ii, ll) = load_dir(os.path.expanduser(d))
        images += ii
        labels = np.r_[labels, ll]
    return images, labels


def features(images):
    """Compute an ndarray of features of size num-images x num-features
from a list of absolute image paths.

    """
    # Feature list:
    # - Sum brightness of each row/column
    features = np.empty([len(images), 0])

    brightness_sums = None
    for idx, img_path in enumerate(images):
        sys.stdout.write("\rLoading image %d" % idx)
        sys.stdout.flush()

        # Unfortunately, skimage.io is *much* slower, so we just use scipy
        #img = io.imread(img_path, as_grey=True, plugin='imread')
        img = scipy.misc.imread(img_path, flatten=True)

        if brightness_sums is None:
            brightness_sums = np.zeros((len(images), np.sum(img.shape)))

        mean_lum = img.mean()
        marginal_lum = np.concatenate((img.sum(0), img.sum(1)))
        brightness_sums[idx, :] = marginal_lum
    print ""

    return np.c_[features, brightness_sums]


def learn_north(model, train_labels, train_images, test_labels, test_images):
    """Using a given model, learn the correct labels and test against
other images.

    """

    print "Learning north from {} images.".format(len(train_labels))
    train_features = features(train_images)

    model.fit(train_features, train_labels)

    print "Testing against {} images.".format(len(test_labels))
    test_features = features(test_images)

    y = model.predict(test_features)
    return (abs(y - test_labels) % 360)


def parse_args():
    parser = argparse.ArgumentParser(description='Learn headings from images.')
    parser.add_argument('-train', nargs='+',
                        help='Directories containing training data')
    parser.add_argument('-test', nargs='+',
                        help='Directories containing testing data')
    parser.add_argument('--lasso', help='Run a LASSO regression',
                        action="store_true")
    parser.add_argument('--ridge', help='Run a Ridge regression',
                        action="store_true")
    return parser, parser.parse_args()

if __name__ == '__main__':
    parser, args = parse_args()
    if args.lasso:
        model = linear_model.Lasso(max_iter=10000)
    elif args.ridge:
        model = linear_model.Ridge()
    else:
        print "Must specify a learning type."
        parser.print_help()
        sys.exit(-1)

    [train_images, train_labels] = load_images_labels(args.train)
    [test_images, test_labels] = load_images_labels(args.test)
    error = learn_north(model,
                        train_labels, train_images,
                        test_labels, test_images)
    plt.scatter(test_labels, error)
    print "Mean error {}".format(error.mean())
    print "Median error {}".format(np.median(error))
    plt.show()
