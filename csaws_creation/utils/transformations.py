""" This scripts contains functions to perform elastic transformations on
the training images of the mammo dataset """

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2


def elastic_transform(image, annotation=None, alpha=0, sigma=0):

    """
    Elastic deformation of images as described in [Simard2003]_.
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """

    assert len(image.shape) == 2
    if annotation is not None:
        assert len(annotation.shape) == 2
    shape = image.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                         sigma,
                         mode="constant",
                         cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                         sigma,
                         mode="constant",
                         cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    if annotation is not None:
        return map_coordinates(image, indices, order=1).reshape(shape), \
               map_coordinates(annotation, indices, order=1).reshape(shape)
    else:
        return map_coordinates(image, indices, order=1).reshape(shape)


def draw_grid(im, grid_size):

    """
    Draw grid lines
    """

    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))
