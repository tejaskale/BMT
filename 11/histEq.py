# -*- coding: utf-8 -*-
#####################################################################
#
#   histEq.py
#   written by: Anca Stefanoiu
#               Chair for Computer Aided Medical Procedures
#               & Augmented Reality
#               Technische Universität München
#               12-01-2018
#
#####################################################################
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# this function computes the normalized histogram
# provided an input image and the desired number of bins
def compute_histogram(img, nbins):
    image_size = img.shape
    hist = np.zeros(nbins)

    for i in range(image_size[0]):
        for j in range(image_size[1]):
            hist[int(img[i][j])] = hist[int(img[i][j])] + 1

    hist = hist / (imageSize[0] * imageSize[1])  # normalize histogram
    # print hist
    return hist


# this function applies histogram equalization
# the returned image is contrast enhanced
def apply_transformation(image, imageSize, maxInt, p):
    image_adjusted = np.zeros([imageSize[0], imageSize[1]])
    for i in range(imageSize[0]):
        for j in range(imageSize[1]):
            image_adjusted[i][j] = (maxInt-1) * sum(p[0:int(image[i][j]+1)])
    return image_adjusted


# load image
image = np.array(Image.open("pout.tif"))
imageSize = image.shape

# intensity range
minInt = 0
maxInt = 255
nBins = maxInt + 1  # number of histogram bins

# build histogram
p = compute_histogram(image, nBins)

# histogram equalizatiopAfterEq = pAfterEq*nBinsn
imageEq = apply_transformation(image, imageSize, maxInt, p)

# histogram after equalization
pAfterEq = compute_histogram(imageEq, nBins)

# Apply the transformation again. There is no change in the histogram or the image
# Because the image intensities are already distributed according to the CDF.
# Recalculating the histogram will result in the same CDF and hence the same intensity for
# Each pixel.

imageEq2 = apply_transformation(imageEq, imageSize, maxInt , pAfterEq)

pAfterEq2 = compute_histogram(imageEq2, nBins)

# show results
plt.figure()

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('original image')

plt.subplot(2, 3, 2)
plt.imshow(imageEq, cmap='gray')
plt.title('image after equalization')

plt.subplot(2, 3, 3)
plt.imshow(imageEq2, cmap='gray')
plt.title('image after equalization2')


plt.subplot(2, 3, 4)
bins = np.linspace(0, nBins, nBins)
plt.bar(bins, p, align='center', width=1)
plt.title('original histogram [normalized]')

plt.subplot(2, 3, 5)
plt.bar(bins, pAfterEq, align='center', width=1)
plt.title('histogram after eq [normalized]')

plt.subplot(2, 3, 6)
plt.bar(bins, pAfterEq2, align='center', width=1)
plt.title('histogram after eq [normalized]2')


plt.show()

