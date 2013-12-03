#!/usr/bin/env python
import scipy.misc
from sklearn import linear_model
from skimage import data, color, io, measure, exposure
import argparse
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.measure import LineModel, ransac

#This is a small program demoing how one would grab different data sets from a photo. Used purely for demo purposes. 

def contour_finding(image):
    #Find all contours at a constant value of 0.8
    contours = measure.find_contours(image, 0.8)
    plt.imshow(img, interpolation='nearest')
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def image_histograms(image):
    print "mer"

# Don't use this yet. There's nothing to compare; at least, not right now.
# Directly copied from the scikit-image documentation 
def RANSAC(image): 
    np.random.seed(seed=1)
    # generate coordinates of line
    x = np.arange(-200, 200)
    y = 0.2 * x + 20
    data = np.column_stack([x, y])

    # add faulty data
    faulty = np.array(30 * [(180., -100)])
    faulty += 5 * np.random.normal(size=faulty.shape)
    data[:faulty.shape[0]] = faulty

    # add gaussian noise to coordinates
    noise = np.random.normal(size=data.shape)
    data += 0.5 * noise
    data[::2] += 5 * noise[::2]
    data[::4] += 20 * noise[::4]

    # fit line using all data
    model = LineModel()
    model.estimate(data)

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModel, min_samples=2,
                                   residual_threshold=1, max_trials=1000)
    outliers = inliers == False

    # generate coordinates of estimated models
    line_x = np.arange(-250, 250)
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_x)

    plt.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
             label='Inlier data')
    plt.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
             label='Outlier data')
    plt.plot(line_x, line_y, '-k', label='Line model from all data')
    plt.plot(line_x, line_y_robust, '-b', label='Robust line model')
    plt.legend(loc='lower left')
    plt.show()


def brightness_gradients(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(64, 64),
                       cells_per_block=(1, 1), visualise=True, normalise=True)
    plt.figure(figsize=(8, 4))
    plt.subplot(121).set_axis_off()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Input image')
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    plt.subplot(122).set_axis_off()
    plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    plt.title('Histogram of Oriented Gradients')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Test different image analyzers.')
    parser.add_argument('-imagedir', nargs='+',
                        help='The path to our image')
    return parser, parser.parse_args()


if __name__ == '__main__':
    parser, args = parse_args()
    imagedir = args.imagedir
    img = scipy.misc.imread(imagedir[0], flatten=True)
    #contour_finding(img)
    #image_histograms(img)
    # RANSAC(img)
    brightness_gradients(img)
    
