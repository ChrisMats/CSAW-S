"""This script creates mammo datasets as tfrecord files"""

import os
import sys
import glob
import json
import tqdm
import numpy as np
from PIL import Image
import multiprocessing
from datetime import datetime
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.ndimage import generic_filter
from multiprocessing import Process, Manager
from settings import NUM_TO_LABEL, CLASSES
from utils import calculate_num_crops

from settings import (
    TEST_SEGMENTATIONS_DIRECTORY, ANNOTATOR_PATHS,
    TEST_ANONYMIZED_DATA_DIRECTORY,
    TEST_RECORD_DIRECTORY,
    DATASET_SPECS,
    RANDOM_VALIDATION_SPLIT,
)

NUM_CLASSES = len(NUM_TO_LABEL)
to_range_256 = interp1d([0, NUM_CLASSES - 1], [0, 255])
to_range_num_classes = interp1d([0, 255], [0, NUM_CLASSES - 1])
LABEL_TO_NUM = {v: k for k, v in NUM_TO_LABEL.items()}



def process_image(target_folder, image_addrs, stuff_addrs, mode, crop_size,
                  crops_per_class, annotators):
    """ given an image, generates patches and saves them

     Parameters:
     -----------
     writer: writer object
        Path to file
     image_addrs: str
        Path to image
     stuff_addrs: str
        Path to annotations
     i: int
        image number in the dataset
     mode: str
        train, val or test

     Returns:
     --------
     crops_of_each_label: array_like
        if mode is 'train', number of crops with central pixel of each
        label type. If mode is 'test', 1.
     pixels_of_each_label: array_like
        number of pixels of each label among the crops generated
     """


    # Open image and array
    img = np.array(Image.open(image_addrs))
    label_path, mask_ID = stuff_addrs.split("/{}/".format(annotators[0]))
    stuff_paths = [os.path.join(label_path, annotator, mask_ID) 
                   for annotator in annotators]
    
    annotations = [np.array(Image.open(label)) for label in stuff_paths]
    img_ID = image_addrs.split("/")[-1][:-4]

    # Make sure is int16
    img = img.astype(np.uint16)
    for i in range(len(annotations)):
        annotations[i] = annotations[i].astype(np.uint8)
    
    # Define width and height
    width = img.shape[0]
    height = img.shape[1]

    # Define variables to save labels information
    crops_of_each_label = np.zeros(NUM_CLASSES)
    pixels_of_each_label = np.zeros(NUM_CLASSES)

    overlapping = 0
#         img = Image.fromarray(img)
    img = Image.fromarray(img.astype(np.uint16)) 
    for i in range(len(annotations)):
        annotations[i] = Image.fromarray(annotations[i].astype(np.uint8))

    # save full images
    full_img_save_path = os.path.join(TEST_RECORD_DIRECTORY, 'images_full', '{}.png'.format(img_ID))
    img.save(full_img_save_path)
    for i in range(len(annotations)):
        full_mask_save_path = os.path.join(TEST_RECORD_DIRECTORY, 'masks_full', annotators[i], '{}.png'.format(img_ID))
        annotations[i].save(full_mask_save_path)

    # get image and segments and start the patching
    x_max, y_max = img.size
    path_list = []
    x0 = 0
    while (x0 + crop_size) < (x_max + crop_size):
        y0 = 0
        while (y0 + crop_size) < (y_max + crop_size):  

            ## if patch exceeds img size then pad
            if ((y0 + crop_size) - y_max > 0) or ((x0 + crop_size) - x_max > 0):
#                     cropped_img = Image.fromarray(np.zeros((crop_size, crop_size)))
                cropped_img = Image.fromarray(np.zeros((crop_size, crop_size), dtype=np.uint16))

                x1 = x0 + crop_size
                y1 = y0 + crop_size                    
                area = (x0, y0, x1, y1)
                str_area = 'x'.join(map(str, area))

                if (y0 + crop_size) - y_max > 0:
                    y1 = y_max
                if (x0 + crop_size) - x_max > 0:
                    x1 = x_max
                area = (x0, y0, x1, y1) 

                t_cropped_img = img.crop(area)
                cropped_img.paste(t_cropped_img)
                for i in range(len(annotations)):
                    cropped_mask = Image.fromarray(np.ones((crop_size, crop_size), 
                                                              dtype=np.uint8)*LABEL_TO_NUM['background'])                 
                    t_cropped_mask = annotations[i].crop(area)
                    cropped_mask.paste(t_cropped_mask)
                    unique_labels = list(np.unique(cropped_mask))
                    # remove blank images
#                     if [LABEL_TO_NUM['background']] != unique_labels:
                    if len(np.unique(np.array(cropped_img))) != 1:
                        if i ==0:
                            img_crop_path = os.path.join(target_folder, 'images','{}-{}.png'.format(img_ID, str_area))
                            cropped_img.save(img_crop_path)
                        mask_crop_path = os.path.join(target_folder, 'masks', annotators[i],
                                                      '{}-{}.png'.format(img_ID, str_area))
                        cropped_mask.save(mask_crop_path)
            else:
                area = (x0, y0, x0 + crop_size, y0 + crop_size)
                str_area = 'x'.join(map(str, area))
                cropped_img = img.crop(area)
                for i in range(len(annotations)):
                    cropped_mask = annotations[i].crop(area)
                    unique_labels = list(np.unique(cropped_mask))
                    # remove blank images
#                     if [LABEL_TO_NUM['background']] != unique_labels:
                    if len(np.unique(np.array(cropped_img))) != 1:
                        if i ==0:
                            img_crop_path = os.path.join(target_folder, 'images','{}-{}.png'.format(img_ID, str_area))
                            cropped_img.save(img_crop_path)
                        mask_crop_path = os.path.join(target_folder, 'masks', annotators[i],
                                                      '{}-{}.png'.format(img_ID, str_area))
                        cropped_mask.save(mask_crop_path)                    
            y0 += crop_size - overlapping
        x0 += crop_size - overlapping


def check_path(fname):
    if not os.path.isdir(fname):
        os.mkdir(fname)
        
def generate_dataset(original_imgs_address, segmentation_addrs, target_folder,
                     mode, crop_size, crops_per_class, annotators):
    """ generates dataset according to defined mode

     Parameters:
     -----------

     segmentation_addrs: list
        List containing all annotations paths.
     target_folder: str
        Folder to save the datasets
     name: str
        Dataset name
     mode: str
        train, val or test
     """

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    if not os.path.isdir(os.path.join(target_folder, mode)):
        os.mkdir(os.path.join(target_folder, mode))
    if not os.path.isdir(os.path.join(target_folder, mode, 'images')):
        os.mkdir(os.path.join(target_folder, mode, 'images'))
    if not os.path.isdir(os.path.join(target_folder, mode, 'masks')):
        os.mkdir(os.path.join(target_folder, mode, 'masks'))
        for annotator in annotators:
            check_path(os.path.join(target_folder, mode, 'masks', annotator))
            
    if not os.path.isdir(os.path.join(TEST_RECORD_DIRECTORY, 'images_full')):
        os.mkdir(os.path.join(TEST_RECORD_DIRECTORY, 'images_full')) 
    if not os.path.isdir(os.path.join(TEST_RECORD_DIRECTORY, 'masks_full')):
        os.mkdir(os.path.join(TEST_RECORD_DIRECTORY, 'masks_full'))  
        for annotator in annotators:
            check_path(os.path.join(TEST_RECORD_DIRECTORY, 'masks_full', annotator))        

    # Read addresses and labels from the 'test' folder
    image_addrs = [os.path.join(original_imgs_address, 
                    segmentation.split("/")[-1][0:3], 
                    segmentation.split("/")[-1][:-16] + ".png")
                   for segmentation in segmentation_addrs]

    # Sort the list of addresses
    train_image_addrs = sorted(image_addrs)
    train_stuff_addrs = sorted(segmentation_addrs)

    # Check that train_image_addrs and train_stuff_addrs have the same length
    if len(train_image_addrs) != len(train_stuff_addrs):
        print("Error: image address list length and label address list"
              " length are different")
        sys.exit(1)

    # Define number of images
    n_images = len(train_image_addrs)
    if n_images < 1:
        print("no registered data found for {}".format(mode))
        return
    
    num_cores = multiprocessing.cpu_count()
    n_jobs = n_images if n_images < num_cores else -3
    print('Patching starts . . .')
    Parallel(n_jobs=n_jobs, verbose=1)(delayed(process_image)(
        target_folder=os.path.join(target_folder , mode), 
        image_addrs=train_image_addrs[i], 
        stuff_addrs=train_stuff_addrs[i], 
        mode=mode, 
        crop_size=crop_size,
        crops_per_class=crops_per_class,
        annotators=annotators)
        for i in range(n_images))   
    
            

if __name__ == "__main__":

    """Creates network datasets"""

    np.random.seed(2019)
    
    SEGMENTATIONS_LIST = sorted(glob.glob(
        os.path.join(TEST_SEGMENTATIONS_DIRECTORY, ANNOTATOR_PATHS[0], '*.png')))

    datasets_to_generate_parameters = []
    for crop_size in DATASET_SPECS["crop_sizes"]:
        for mode in ["test"]:

            dataset_folder = os.path.join(TEST_RECORD_DIRECTORY, 
                                  "crop_size_{}".format(crop_size))
            if not os.path.isdir(dataset_folder):
                os.mkdir(dataset_folder)
            datasets_to_generate_parameters.append(
                {
                    "original_imgs_address": TEST_ANONYMIZED_DATA_DIRECTORY,
                    "segmentation_addrs": SEGMENTATIONS_LIST,
                    "target_folder": dataset_folder,
                    "mode": mode,
                    "crop_size": crop_size,
                    "crops_per_class": calculate_num_crops(crop_size),
                    "annotators" : ANNOTATOR_PATHS,
                }
            )

    start = datetime.now()

    for dataset_parameters in datasets_to_generate_parameters:
        generate_dataset(**dataset_parameters)

    end = datetime.now()
    delta = end - start
    print(
        '\n\tDatasets generated in %d hours, %d minutes and %d seconds' % (
            delta.seconds // 3600, ((delta.seconds // 60) % 60),
            delta.seconds % 60))
