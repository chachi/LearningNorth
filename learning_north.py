#!/usr/bin/env python
import scipy.misc
from sklearn import linear_model, svm, tree
from skimage import transform
import argparse
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn import cross_validation


class LearningOptions:
    def __init__(self, bin_min, bin_max, bin_width):
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_width = bin_width

REGRESSION = "Regression"
CLASSIFICATION = "Classification"

ns_regression_opts = LearningOptions(0, 180, 5)
ew_regression_opts = LearningOptions(-90, 90, 5)

ns_classification_opts = LearningOptions(0, 30, 1)
ew_classification_opts = LearningOptions(-15, 15, 1)

MODELS = {
    'lasso': linear_model.Lasso(max_iter=10000),
    'ridge': linear_model.Ridge(),
    'svm_rbf': svm.SVC(kernel='rbf', gamma=0.7, C=1),
    'svm_poly': svm.SVC(kernel='poly', degree=3, C=1),
    'svm_linear': svm.LinearSVC(C=1),
    'tree': tree.DecisionTreeClassifier()
}

TITLES = {
    'lasso': 'LASSO',
    'ridge': 'Ridge',
    'svm_rbf': 'SVM with RBF Kernel',
    'svm_poly': 'SVM with Polynomial Kernel',
    'svm_linear': 'Linear SVM',
    'tree': 'Decision tree'
}

TYPES = {
    'lasso': REGRESSION,
    'ridge': REGRESSION,
    'svm_rbf': CLASSIFICATION,
    'svm_poly': CLASSIFICATION,
    'svm_linear': CLASSIFICATION,
    'tree': CLASSIFICATION
}


N_FOLDS = 10
N_CLASSES = 12


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
    return images, np.array(labels)


def compute_features(images):
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
    print ''
    return np.c_[features, brightness_sums]


def learn_and_plot(model, x, y, model_type, orientation_type, options):
    print "Learning {} from {} images with a {}.".format(orientation_type,
                                                         len(y),
                                                         model_type)

    scores = evaluate_learning(model, x, y)
    scores = np.sqrt(scores)

    plot_model(model_type, orientation_type, y, scores, options)

    print "Mean scores {}".format(scores.mean())
    print "Std scores {}".format(np.std(scores))
    print "Median scores {}".format(np.median(scores))


def evaluate_learning(model, x, y):
    """Using a given model, learn the correct labels and test against
other images.

    """
    #######
    # Choose your cross-validation splitting strategy
    #######
    #fold_gen = cross_validation.StratifiedKFold(labels, n_folds=N_FOLDS)
    #fold_gen = cross_validation.KFold(n=N_FOLDS, n_folds=N_FOLDS)
    fold_gen = cross_validation.LeaveOneOut(len(y))
    #fold_gen = cross_validation.ShuffleSplit(len(y), random_state=0,
    #                                         n_iter=50)
    if isinstance(fold_gen, cross_validation.KFold):
        print "Performing {}-fold cross validation".format(N_FOLDS)

    scores = cross_validation.cross_val_score(model, x, y,
                                              scoring='mean_squared_error',
                                              n_jobs=-1,
                                              cv=fold_gen)
    return -scores


def plot_model(model_type, orientation_type, labels, scores, options):
    # Plot:
    # Histogram of true angle vs. average error
    # Raw scores
    # Overlay mean error
    bins = np.arange(options.bin_min, options.bin_max, options.bin_width)
    avg_error = np.zeros(bins.shape)
    for i, bin in enumerate(bins):
        current = (labels >= bin) & (labels < bin + options.bin_width)
        avg_error[i] = scores[current].mean()

    plt.figure()
    plt.bar(left=bins, height=avg_error, width=options.bin_width)
    plt.title(model_type)
    plt.savefig('_'.join((model_type, orientation_type, 'bar.eps')))

    plt.figure()
    plt.plot(scores, 'ro')
    plt.savefig('_'.join((model_type, orientation_type, 'plot.eps')))
    plt.close('all')


def east_west_labels(labels):
    """Takes 0-360 labels and returns -90-90"""
    return np.degrees(np.arcsin(np.sin(np.radians(labels))))


def north_south_labels(labels):
    """Takes 0-360 labels and returns 0-180"""
    return np.degrees(np.arccos(np.cos(np.radians(labels))))


def run(x, model, model_type, learn_type):
    # Plot North-South orientation
    if learn_type is REGRESSION:
        y = north_south_labels(labels)
        opts = ns_regression_opts
    else:
        y = np.floor(north_south_labels(labels/N_CLASSES))
        opts = ns_classification_opts

    orientation_type = "North-South"
    learn_and_plot(model, x, y, model_type, orientation_type, opts)

    #  Plot East-West orientation
    if learn_type is REGRESSION:
        y = east_west_labels(labels)
        opts = ew_regression_opts
    else:
        y = np.floor(east_west_labels(labels/N_CLASSES))
        opts = ew_classification_opts

    orientation_type = "East-West"
    learn_and_plot(model, x, y, model_type, orientation_type, opts)


def parse_args():
    parser = argparse.ArgumentParser(description='Learn headings from images.')
    parser.add_argument('-data', nargs='+',
                        help='Directories containing data')
    parser.add_argument('--lasso', help='Run a LASSO regression',
                        action='store_true')
    parser.add_argument('--ridge', help='Run a Ridge regression',
                        action='store_true')
    parser.add_argument('--svm_rbf',
                        help='Run SVM classifier with rbf kernel',
                        action='store_true')
    parser.add_argument('--svm_poly',
                        help='Run SVM classifier with ploynomial kernel (3)',
                        action='store_true')
    parser.add_argument('--svm_lin', help='Run a Linear SVM classifier',
                        action='store_true')
    parser.add_argument('--tree', help='Run a Decision Tree classifier',
                        action='store_true')
    parser.add_argument('--all', help='Run all classifiers and regressions.',
                        action='store_true')
    parser.add_argument('-outdir', help='Output directory for saved plots.',
                        action='store', default='.')
    return parser, parser.parse_args()


if __name__ == '__main__':
    parser, args = parse_args()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    os.chdir(args.outdir)

    #Machine Learning Models
    if args.all:
        models_to_use = MODELS.keys()
    elif args.lasso:
        models_to_use = ('lasso',)
    elif args.ridge:
        models_to_use = ('ridge',)
    elif args.svm_rbf:
        models_to_use = ('svm_rbf',)
    elif args.svm_poly:
        models_to_use = ('svm_poly',)
    elif args.svm_lin:
        models_to_use = ('svm_linear',)
    elif args.tree:
        models_to_use = ('tree',)
    else:
        print "Must specify a learning type."
        parser.print_help()
        sys.exit(-1)

    [images, labels] = load_images_labels(args.data)
    x = compute_features(images)
    for model_type in models_to_use:
        run(x, MODELS[model_type], TITLES[model_type], TYPES[model_type])
