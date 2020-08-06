"""This script creates the patched dataset"""

import sys
import glob
import json
from tqdm import tqdm
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

import os
from utils import get_train_validation_split
from settings import (
    SEGMENTATIONS_DIRECTORY,
    ANONYMIZED_DATA_DIRECTORY,
    RECORD_DIRECTORY,
    DATASET_SPECS,
    RANDOM_VALIDATION_SPLIT,
)
        
np.random.seed(2019)    
NUM_CLASSES = len(NUM_TO_LABEL)
to_range_256 = interp1d([0, NUM_CLASSES - 1], [0, 255])
to_range_num_classes = interp1d([0, 255], [0, NUM_CLASSES - 1])
LABEL_TO_NUM = {v: k for k, v in NUM_TO_LABEL.items()}
SEGMENTATIONS_LIST = sorted(glob.glob(os.path.join(SEGMENTATIONS_DIRECTORY, "*.png")))

def process_image(target_folder, image_addrs, stuff_addrs, mode, crop_size,
                  crops_per_class):
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
    label = np.array(Image.open(stuff_addrs))
    img_ID = image_addrs.split("/")[-1][:-4]

    # Make sure is int16
    img = img.astype(np.uint16)
    annotations = label.astype(np.uint8)
    
    # Define width and height
    width = img.shape[0]
    height = img.shape[1]

    # Define variables to save labels information
    crops_of_each_label = np.zeros(NUM_CLASSES)
    pixels_of_each_label = np.zeros(NUM_CLASSES)

    if mode in ('train'):

        # create one list per each label with the positions
        positions = [[] for _ in range(NUM_CLASSES)]

        for pixel_col in range(width):
            for pixel_row in range(height):
                label = annotations[pixel_col, pixel_row]
                positions[label].append([pixel_col, pixel_row])

        # define dict
        positions_dict = {}
        for pos, _ in enumerate(positions):
            if positions[pos]:
                positions_dict[str(pos)] = positions[pos]

        # list of labels contained in this image
        unique_labels = list(np.unique(annotations))

        # remove background and mammary gland
        if ["mammary_gland"] in CLASSES:
            if LABEL_TO_NUM['background'] in unique_labels:
                unique_labels.remove(LABEL_TO_NUM['background'])
            if LABEL_TO_NUM["mammary_gland"] in unique_labels:
                unique_labels.remove(LABEL_TO_NUM['mammary_gland'])

        for unique_label in unique_labels:
            for crop_number in range(crops_per_class):

                # Sample random pixel of class unique_label
                sampled_pixel = np.random.randint(low=0, high=len(
                    positions_dict.get(str(unique_label))))

                # Get pixel coordinates
                coordinates = positions_dict.get(
                    str(unique_label))[sampled_pixel]

                # Find upper left corner of the crop
                x_coordinate = np.clip(
                    coordinates[0] - (crop_size // 2), 0, width)
                y_coordinate = np.clip(
                    coordinates[1] - (crop_size // 2), 0, height)

                # Check coordinates not too close from right or bottom side
                if x_coordinate + crop_size >= width:
                    x_coordinate = width - crop_size
                if y_coordinate + crop_size >= height:
                    y_coordinate = height - crop_size

                # Get crop
                img_crop = img[x_coordinate:x_coordinate + crop_size,
                           y_coordinate:y_coordinate + crop_size]
                annotation_crop = annotations[
                                  x_coordinate:x_coordinate + crop_size,
                                  y_coordinate: y_coordinate + crop_size]


                
                # Save img and mask patches in foler
                img_crop = Image.fromarray(img_crop.astype(np.uint16))
                annotation_crop = Image.fromarray(annotation_crop.astype(np.uint8))
                img_crop.save(os.path.join(target_folder, 'images',
                                           '{}-{}-{}.png'.format(img_ID,unique_label, crop_number)))
                annotation_crop.save(os.path.join(target_folder, 'masks',
                                                  '{}-{}-{}.png'.format(img_ID,unique_label, crop_number)))


                # Increase the number of crops of type unique_label
                crops_of_each_label[unique_label] += 1


    else:
        
        overlapping = 0
        img = Image.fromarray(img.astype(np.uint16))        
        annotations = Image.fromarray(annotations.astype(np.uint8))
        
        # save full images
        full_img_save_path = os.path.join(RECORD_DIRECTORY, 'images_full', '{}.png'.format(img_ID))
        full_mask_save_path = os.path.join(RECORD_DIRECTORY, 'masks_full', '{}.png'.format(img_ID))
        img.save(full_img_save_path)
        annotations.save(full_mask_save_path)
        
        # get image and segments and start the patching
        x_max, y_max = img.size
        path_list = []
        x0 = 0
        while (x0 + crop_size) < (x_max + crop_size):
            y0 = 0
            while (y0 + crop_size) < (y_max + crop_size):  
                
                ## if patch exceeds img size then pad
                if ((y0 + crop_size) - y_max > 0) or ((x0 + crop_size) - x_max > 0):
                    cropped_img = Image.fromarray(np.zeros((crop_size, crop_size), dtype=np.uint16))
                    cropped_mask = Image.fromarray(np.ones((crop_size, crop_size), dtype=np.uint8)*LABEL_TO_NUM['background'])
                    
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
                    t_cropped_mask = annotations.crop(area)
                    cropped_img.paste(t_cropped_img)
                    cropped_mask.paste(t_cropped_mask)
                    unique_labels = list(np.unique(cropped_mask))
                    # remove blank images
                    if [LABEL_TO_NUM['background']] != unique_labels:
                        img_crop_path = os.path.join(target_folder, 'images','{}-{}.png'.format(img_ID, str_area))
                        mask_crop_path = os.path.join(target_folder, 'masks','{}-{}.png'.format(img_ID, str_area))
                        cropped_img.save(img_crop_path)
                        cropped_mask.save(mask_crop_path)
                else:
                    area = (x0, y0, x0 + crop_size, y0 + crop_size)
                    str_area = 'x'.join(map(str, area))
                    cropped_img = img.crop(area)
                    cropped_mask = annotations.crop(area)
                    unique_labels = list(np.unique(cropped_mask))
                    # remove blank images
                    if [LABEL_TO_NUM['background']] != unique_labels:
                        img_crop_path = os.path.join(target_folder, 'images','{}-{}.png'.format(img_ID, str_area))
                        mask_crop_path = os.path.join(target_folder, 'masks','{}-{}.png'.format(img_ID, str_area))

                        cropped_img.save(img_crop_path)
                        cropped_mask.save(mask_crop_path)                    
                y0 += crop_size - overlapping
            x0 += crop_size - overlapping
            
    print("{} -- done ".format(img_ID))
    sys.stdout.flush()
    
            

def generate_dataset(original_imgs_address, segmentation_addrs, target_folder,
                     mode, crop_size, crops_per_class):
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
        
    if not os.path.isdir(os.path.join(RECORD_DIRECTORY, 'images_full')):
        os.mkdir(os.path.join(RECORD_DIRECTORY, 'images_full')) 
    if not os.path.isdir(os.path.join(RECORD_DIRECTORY, 'masks_full')):
        os.mkdir(os.path.join(RECORD_DIRECTORY, 'masks_full'))         

    # Read addresses and labels from the 'train' folder
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
    n_images = len(train_stuff_addrs)
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
        crops_per_class=crops_per_class)
        for i in range(n_images))        

if __name__ == "__main__":

    """Creates network datasets"""

    np.random.seed(2019)

    split_screenings = get_train_validation_split(
        SEGMENTATIONS_LIST,
        percent_test=0,
        percent_validation=10,
        random_split=RANDOM_VALIDATION_SPLIT
    )

    datasets_to_generate_parameters = []
    for crop_size in DATASET_SPECS["crop_sizes"]:
        for mode in ["train", "val"]:
            dataset_folder = os.path.join(RECORD_DIRECTORY, 
                                  "crop_size_{}".format(crop_size))
            if not os.path.isdir(dataset_folder):
                os.mkdir(dataset_folder)

            datasets_to_generate_parameters.append(
                {
                    "original_imgs_address": ANONYMIZED_DATA_DIRECTORY,
                    "segmentation_addrs": split_screenings[mode],
                    "target_folder": dataset_folder,
                    "mode": mode,
                    "crop_size": crop_size,
                    "crops_per_class": calculate_num_crops(crop_size),
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
