"""
This scripts generates global segmentation maps from the binary segmentation
maps of the specified classes.
"""

import cv2
import re
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from utils import colorize
import pdb
import os
from joblib import Parallel, delayed
from settings import (
    ANONYMIZED_DATA_DIRECTORY, SEGMENTATIONS_DIRECTORY,
    CLASSES, CLASSES_SMALL, APPLY_SMOOTHING, NUM_TO_LABEL)
LABEL_TO_NUM = {v: k for k, v in NUM_TO_LABEL.items()}

def generate_single_segmentation_map(
        path_to_binary_maps_folder,
        dataset_classes,
        small_object_classes,
        target_directory,
        verbose=False,
        apply_smoothing=False
):
    """

    Merges binary segmentation maps of one single patient.

    """

    # Errors variable
    errors = False

    # Define path to all png files in the folder
    path = os.path.join(path_to_binary_maps_folder ,"*.png")

    # Create list with addresses of all files in the folder
    addrs = glob.glob(path)

    # Find out how many screenings we have from this patient
    all_images = re.compile(".*_[0-9].png")
    image_list = list(filter(all_images.match, addrs))
    number_of_images = len(image_list)

    # If no images, exit the program
    if number_of_images == 0:
        all_images = re.compile(".*_[0-9][0-9].png")
        image_list = list(filter(all_images.match, addrs))
        number_of_images = len(image_list)
        if number_of_images == 0:
            return

    # For each screening..
    for i in range(number_of_images):

        # Create a list with all binary map files in dataset_classes
        binary_maps = []
        for pattern in dataset_classes:
            try:
                binary_maps.append(list(
                    filter(re.compile(
                        image_list[i][:-4] + "_" + pattern).match, addrs))[0])
            except IndexError:
                if verbose:
                    print(
                        colorize("\tmissing binary map for class {}".format(
                            pattern), "red"))
                else:
                    errors = True
                continue

        # Create a new "empty" array
        image = np.array(Image.open(image_list[i]))
        if image.shape[-1] == 3:
            try:
                image = np.array(Image.open(image_list[i][:-4] + "_8bit.png"))
            except Exception:
                print(colorize("\timage has RGB format", "red"))
        # Label as background all non-labelled pixels
        label_map = np.ones(image.shape) * LABEL_TO_NUM['background']

        # For each segmentation label, following the defined priority order..
        for segmentation_name in reversed(dataset_classes):
            label = LABEL_TO_NUM[segmentation_name]

            # Select the binary file path
            pattern = '.*' + segmentation_name + '*'
            m = re.compile(pattern)
            map_path = list(filter(m.match, binary_maps))

            # Only one path should be selected
            if len(map_path) != 1:
                if verbose:
                    print(colorize(
                        "\terror when selecting binary mask {}".format(
                            segmentation_name), "red"))
                else:
                    errors = True
                continue

            # Open binary mask file and apply gaussian filter
            try:
                mask = np.array(Image.open(map_path[0]))
            except IOError:
                if verbose:
                    print(
                        colorize("\tproblems to open binary map {}".format(
                            segmentation_name), "red"))
                else:
                    errors = True
                continue

            if segmentation_name not in small_object_classes and apply_smoothing:
                mask = gaussian_filter(mask, sigma=1)
            mask[mask>0] = 1

            # Label pixels
            try:
                label_map[mask == 1] = label
            except ValueError:
                if verbose:
                    print(colorize("\t binary mask {} ".format(
                        segmentation_name) +
                                   "with different size than image", "red"))
                else:
                    errors = True
                continue
            

        # Save segmentation map
        save_name = os.path.join(target_directory,  
                                 image_list[i][:-4].split("/")[-1] + "_annotations.png")           

        # Report if errors found
        if errors is True and verbose is False:
            print(
                colorize(
                    "\t Errors detected!", "red"))
        elif errors is False:
            print("{}".format(image_list[i].split("/")[-1]))
            cv2.imwrite(save_name, label_map)

        if errors is True:
            print("\n======================================================\n")


if __name__ == "__main__":

    if not os.path.isdir(SEGMENTATIONS_DIRECTORY):
        os.mkdir(SEGMENTATIONS_DIRECTORY)

    list_of_patient_dirs = glob.glob(os.path.join(ANONYMIZED_DATA_DIRECTORY, "*"))
    Parallel(n_jobs=-1, verbose=1)(
        delayed(generate_single_segmentation_map)(
            path_to_binary_maps_folder=patient_dir,
            dataset_classes=CLASSES,
            small_object_classes=CLASSES_SMALL,
            target_directory=SEGMENTATIONS_DIRECTORY,
            verbose=True,
        apply_smoothing=APPLY_SMOOTHING)
        for patient_dir in sorted(list_of_patient_dirs))
