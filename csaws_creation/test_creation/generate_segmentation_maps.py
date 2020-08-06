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

import os
from joblib import Parallel, delayed
from settings import (TEST_TARGET_DIRECTORY,
    TEST_ANONYMIZED_DATA_DIRECTORY, TEST_SEGMENTATIONS_DIRECTORY, ANNOTATOR_PATHS,
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
    path = path_to_binary_maps_folder + "/*.png"

    # Create list with addresses of all files in the folder
    
    image_list = glob.glob(path)
    # Find out how many screenings we have from this patient
    patient_list = np.unique(['_'.join(imp.split('/')[-1].split('_')[:2]) for imp in image_list]).tolist()
    number_of_images = len(patient_list)
    
    # If no patient annotations, exit the program
    if number_of_images == 0:
            return

    # For each screening..
    for i in range(number_of_images):

        # Create a list with all binary map files in dataset_classes
        binary_maps = []
        for pattern in dataset_classes:
            try:
                binary_maps.append(os.path.join(path_to_binary_maps_folder, 
                                                '{}_{}.png'.format(patient_list[i], pattern)))
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
        for segmentation_name, map_path in reversed(list(zip(dataset_classes, binary_maps))):
            label = LABEL_TO_NUM[segmentation_name]
            # Open binary mask file and apply gaussian filter
            try:
                mask = np.array(Image.open(map_path))
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
        save_name = os.path.join(target_directory, '{}_annotations.png'.format(patient_list[i]))

        # Report if errors found
        if errors is True and verbose is False:
            print(
                colorize(
                    "\t Errors detected!", "red"))
        elif errors is False:
            print("Masks blended for {}".format(patient_list[i]))
            cv2.imwrite(save_name, label_map)

        if errors is True:
            print("\n======================================================\n")


if __name__ == "__main__":

    for annotator_path in ANNOTATOR_PATHS:
        print("\nGenerating masks for: {}\n".format(annotator_path))
        data_dir = os.path.join(TEST_TARGET_DIRECTORY,annotator_path)
        target_dir = os.path.join(TEST_SEGMENTATIONS_DIRECTORY,annotator_path)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)        
        patient_ids = [ idd.split('/')[-1] for idd in glob.glob(os.path.join(data_dir, '*'))]
        for patient_id in sorted(patient_ids):
            print(annotator_path, patient_id)
            generate_single_segmentation_map(
                path_to_binary_maps_folder=os.path.join(data_dir,patient_id),
                dataset_classes=CLASSES,
                small_object_classes=CLASSES_SMALL,
                target_directory=target_dir,
                verbose=True,
                apply_smoothing=APPLY_SMOOTHING)