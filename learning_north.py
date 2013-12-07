#!/usr/bin/env python
import scipy.misc
from sklearn import linear_model, svm, tree
from skimage import color, io, transform
import argparse
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import matplotlib.pyplot as plt
from sklearn import cross_validation


N_FOLDS=10

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
        img /= 255

        thumb = transform.resize(img, np.array(img.shape)/70)
        if brightness_sums is None:
            brightness_sums = np.zeros((len(images), np.sum(thumb.shape)))

        marginal_lum = np.concatenate((thumb.sum(0), thumb.sum(1)))
        brightness_sums[idx, :] = marginal_lum

    print ""

    return np.c_[features, brightness_sums]


def learn_north(model, labels, images):
    """Using a given model, learn the correct labels and test against
other images.

    """
    print "Learning north from {} images.".format(len(labels))
    x = features(images)
    # Keep zero at 0, move 180-360 onto 0-180
    y = np.abs(np.abs(labels - 180)-180)

    #plt.hist(labels)
    #plt.show()

    #######
    # Choose your cross-validation splitting strategy
    #######
    #fold_gen = cross_validation.StratifiedKFold(labels, n_folds=N_FOLDS)
    #fold_gen = cross_validation.KFold(n=N_FOLDS, n_folds=N_FOLDS)
    fold_gen = cross_validation.LeaveOneOut(len(y))
    #fold_gen = cross_validation.ShuffleSplit(len(y), random_state=0, n_iter=50)
    if isinstance(fold_gen, cross_validation.KFold):
        print "Performing {}-fold cross validation".format(N_FOLDS)
    
    scores = cross_validation.cross_val_score(model, x, y,
                                              scoring='mean_squared_error',
                                              n_jobs=-1,
                                              cv=fold_gen)
    return -scores


def parse_args():
    parser = argparse.ArgumentParser(description='Learn headings from images.')
    parser.add_argument('-data', nargs='+',
                        help='Directories containing data')
    parser.add_argument('--lasso', help='Run a LASSO regression',
                        action="store_true")
    parser.add_argument('--ridge', help='Run a Ridge regression',
                        action="store_true")
    parser.add_argument('--svm_rbf', help='Run an SVM classifier with rbf kernel',
                        action="store_true")
    parser.add_argument('--svm_poly', help='Run an SVM classifier with ploynomial kernel (3)', 
                        action="store_true")
    parser.add_argument('--svm_lin', help='Run a Linear SVM classifier', 
                        action="store_true")
    parser.add_argument('--tree', help='Run a Decision Tree classifier', 
                        action="store_true")
    return parser, parser.parse_args()

if __name__ == '__main__':
    parser, args = parse_args()

    #Machine Learning Models

    if args.lasso:
        model = linear_model.Lasso(max_iter=10000)
    elif args.ridge:
        model = linear_model.Ridge()
    elif args.svm_rbf:
        model = svm.SVC(kernel='rbf', gamma=0.7, C=1)
    elif args.svm_poly:
        model = svm.SVC(kernel='poly', degree=3, C=1)
    elif args.svm_lin:
        model = svm.LinearSVC(C=1)
    elif args.tree:
        model = tree.DecisionTreeClasssifier()
    else:
        print "Must specify a learning type."
        parser.print_help()
        sys.exit(-1)

    [images, labels] = load_images_labels(args.data)
    scores = learn_north(model, labels, images)
    scores = np.sqrt(scores)

    plt.plot(scores, 'ro')
    print "Mean scores {}".format(scores.mean())
    print "Std scores {}".format(np.std(scores))
    print "Median scores {}".format(np.median(scores))
    plt.show()
