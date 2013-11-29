import scipy.misc
from sklearn import linear_model
from skimage import color
import argparse
import numpy as np
import os
import pandas as pd
import re
import sys
import time


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
    features = np.ndarray([len(images), 1])
    for idx, img_path in enumerate(images):
        print "Loading image %d" % idx
        img = scipy.misc.imread(img_path, flatten=True)
        features[idx, 0] = img.mean()

    return features


def learn_north(model, train_labels, train_images, test_labels, test_images):
    """Using a given model, learn the correct labels and test against
other images.

    """

    print "Learning north from {} images.".format(len(train_labels))
    train_features = features(train_images)

    model.fit(train_features, train_labels)

    print "Testing against {} images.".format(len(test_labels))
    test_features = features(test_images)

    error = 0.
    for idx, heading in enumerate(train_labels):
        y = model.predict(test_features[idx, 0])
        error += (abs(y - heading) % 360)

    print error/len(train_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn headings from images.')
    parser.add_argument('-train', nargs='+',
                        help='Directories containing training data')
    parser.add_argument('-test', nargs='+',
                        help='Directories containing testing data')
    parser.add_argument('--lasso', help='Run a LASSO regression',
                        action="store_true")
    parser.add_argument('--ridge', help='Run a Ridge regression',
                        action="store_true")

    args = parser.parse_args()

    if args.lasso:
        model = linear_model.Lasso()
    elif args.ridge:
        model = linear_model.Ridge()
    else:
        print "Must specify a learning type."
        parser.print_help()
        sys.exit(-1)

    [train_images, train_labels] = load_images_labels(args.train)
    [test_images, test_labels] = load_images_labels(args.test)
    learn_north(model, train_labels, train_images, test_labels, test_images)
