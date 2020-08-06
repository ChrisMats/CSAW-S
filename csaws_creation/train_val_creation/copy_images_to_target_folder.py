"""Copy all orginal screenings to a new directory."""

import re
import glob
import shutil
import numpy as np
from PIL import Image
from utils import colorize

import os
from joblib import Parallel, delayed
from settings import (
    ANONYMIZED_DATA_DIRECTORY, IMAGES_DIRECTORY)


def copy_image(origin_folder, target_folder):
    """
    Find original screening in origin_folder and copy it to target_folder.
    """

    # Define path to all png files in the folder
    path = origin_folder + "/*.png"

    # Create list with addresses of all files in the folder
    addrs = sorted(glob.glob(path))

    # Get screenings path
    all_images = re.compile(".*_[0-9].png")
    image_list = list(filter(all_images.match, addrs))
    number_of_images = len(image_list)

    # If no images
    if number_of_images == 0:
        all_images = re.compile(".*_[0-9][0-9].png")
        image_list = list(filter(all_images.match, addrs))

    # Copy to new folder
    for image in image_list:

        # Check if image is RGB or 8bit
        img = np.array(Image.open(image))

        if len(img.shape) != 2:
            print(img.shape)
            print(colorize("{}".format(origin_folder), "red"))

        shutil.copy(image, target_folder)


if __name__ == "__main__":

    if not os.path.isdir(IMAGES_DIRECTORY):
        os.mkdir(IMAGES_DIRECTORY)
    list_of_patient_dirs = glob.glob(os.path.join(ANONYMIZED_DATA_DIRECTORY, "*"))
    Parallel(n_jobs=-1, verbose=1)(
        delayed(copy_image)(
            origin_folder=patient_dir,
            target_folder=IMAGES_DIRECTORY,
            )
        for patient_dir in sorted(list_of_patient_dirs))
