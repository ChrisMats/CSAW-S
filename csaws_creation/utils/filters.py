import numpy as np


def my_filter(buffer):

    """
    Implements a filter that return the most common value in buffer

    Parameters:
    -----------
    buffer: array_like
        Patch around pixel under consideration
    """

    counts = np.bincount(buffer.astype('int64'))
    return np.argmax(counts)