#!/usr/bin/env python
# coding: utf-8

import os
import glob
import shutil
from tqdm import tqdm
import multiprocessing
from datetime import datetime
from joblib import Parallel, delayed

from utils import *
from settings import *

import test_creation
import train_val_creation



import pdb

def main():
    start = datetime.now()  
    
    ## 1. generate segmentation maps from the original files  
    
    print("\n Generate trainval segmentation maps ")
    
    list_of_patient_dirs = glob.glob(os.path.join(ANONYMIZED_DATA_DIRECTORY, "*"))  
    Parallel(n_jobs=-1, verbose=1)(
        delayed(train_val_creation.generate_single_segmentation_map)(
            path_to_binary_maps_folder=patient_dir,
            dataset_classes=CLASSES,
            small_object_classes=CLASSES_SMALL,
            target_directory=SEGMENTATIONS_DIRECTORY,
            verbose=True,
        apply_smoothing=APPLY_SMOOTHING)
        for patient_dir in sorted(list_of_patient_dirs)) 
    
    print("\n Generate test segmentation maps ")
        
    for annotator_path in ANNOTATOR_PATHS:
        print("\nGenerating masks for: {}\n".format(annotator_path))
        data_dir = os.path.join(TEST_TARGET_DIRECTORY, annotator_path)
        target_dir = os.path.join(TEST_SEGMENTATIONS_DIRECTORY, annotator_path)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)        
        list_of_patient_dirs = glob.glob(os.path.join(data_dir, "*"))  
        
        Parallel(n_jobs=-1, verbose=1)(
            delayed(test_creation.generate_single_segmentation_map)(
                    path_to_binary_maps_folder=patient_dir,
                    dataset_classes=CLASSES,
                    small_object_classes=CLASSES_SMALL,
                    target_directory=target_dir,
                    verbose=True,
                    apply_smoothing=APPLY_SMOOTHING)
            for patient_dir in sorted(list_of_patient_dirs))                 

        
    ## 2. Copy all orginal screenings to a new directory

    print("\n Copy trainval orginal screenings to a new directory ")

    list_of_patient_dirs = glob.glob(os.path.join(ANONYMIZED_DATA_DIRECTORY, "*"))  
    Parallel(n_jobs=-1, verbose=1)(
        delayed(train_val_creation.copy_image)(
            origin_folder=patient_dir,
            target_folder=IMAGES_DIRECTORY,
            )
        for patient_dir in sorted(list_of_patient_dirs))  
    
    print("\n Copy test orginal screenings to a new directory ")
    
    list_of_patient_dirs = glob.glob(os.path.join(TEST_ANONYMIZED_DATA_DIRECTORY, "*"))  
    Parallel(n_jobs=-1, verbose=1)(
        delayed(train_val_creation.copy_image)(
            origin_folder=patient_dir,
            target_folder=TEST_IMAGES_DIRECTORY,
            )
        for patient_dir in sorted(list_of_patient_dirs))     
    
    
    ## 3. patched dataset creation

    print("\n Generate trainval patched dataset ")

    SEGMENTATIONS_LIST = sorted(glob.glob(os.path.join(SEGMENTATIONS_DIRECTORY, "*.png")))
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

    for dataset_parameters in datasets_to_generate_parameters:
        train_val_creation.generate_dataset(**dataset_parameters)


    print("\n Generate test patched dataset ")
    
    TEST_SEGMENTATIONS_LIST = sorted(glob.glob(os.path.join(TEST_SEGMENTATIONS_DIRECTORY, 
                                                            ANNOTATOR_PATHS[0], '*.png')))

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
                    "segmentation_addrs": TEST_SEGMENTATIONS_LIST,
                    "target_folder": dataset_folder,
                    "mode": mode,
                    "crop_size": crop_size,
                    "crops_per_class": calculate_num_crops(crop_size),
                    "annotators" : ANNOTATOR_PATHS,
                }
            )

    for dataset_parameters in datasets_to_generate_parameters:
        test_creation.generate_dataset(**dataset_parameters)
    
    ## 4. binary mask creation
    
    print("\n Generate tbinary masks for trainval ")

    print("generating croped binary masks from croped masks")
    crop_paths = [crop_path for crop_path in glob.glob(os.path.join(RECORD_DIRECTORY, '*')) 
               if 'crop_size_' in crop_path]
   
    for crop_path_size in tqdm(crop_paths):
        for mode in ["train", "val"]:
            folder_path = os.path.join(crop_path_size, mode,)
            bin_folder_path = os.path.join(folder_path, 'binary_masks')
            if not os.path.isdir(bin_folder_path):
                os.mkdir(bin_folder_path)
                
            mask_paths = glob.glob(os.path.join(folder_path, 'masks', '*.png'))
            num_cores = multiprocessing.cpu_count()
            n_jobs = len(mask_paths) if len(mask_paths) < num_cores else -3

            print('Binarization starts . . .')
            Parallel(n_jobs=n_jobs, verbose=1)(delayed(train_val_creation.save_mask_to_binary)(
                org_path=mask_paths[i], 
                target_path=os.path.join(bin_folder_path, mask_paths[i].split('/')[-1]), 
                dominant_class='cancer')
                for i in range(len(mask_paths)))

    print("generating full binary masks from full masks")
    full_mask_path = os.path.join(RECORD_DIRECTORY, 'masks_full')
    bin_folder_path = os.path.join(RECORD_DIRECTORY, 'binary_masks_full')
    if not os.path.isdir(bin_folder_path):
        os.mkdir(bin_folder_path)
        
    mask_paths = glob.glob(os.path.join(full_mask_path, '*.png'))
    num_cores = multiprocessing.cpu_count()
    n_jobs = len(mask_paths) if len(mask_paths) < num_cores else -3   

    print('Binarization starts . . .')
    Parallel(n_jobs=n_jobs, verbose=1)(delayed(train_val_creation.save_mask_to_binary)(
        org_path=mask_paths[i], 
        target_path=os.path.join(bin_folder_path, mask_paths[i].split('/')[-1]), 
        dominant_class='cancer')
        for i in range(len(mask_paths))) 


    print("\n Generate tbinary masks for test ")
    
    print("generating croped binary masks from croped masks")
    crop_paths = [crop_path for crop_path in glob.glob(os.path.join(TEST_RECORD_DIRECTORY, '*')) 
               if 'crop_size_' in crop_path]

    for crop_path_size in tqdm(crop_paths):
        for mode in ["test"]:
            folder_path = os.path.join(crop_path_size, mode)
            bin_folder_path = os.path.join(folder_path, 'binary_masks')
            mask_folder_path = os.path.join(folder_path, 'masks')
            test_creation.check_path(bin_folder_path)
            for annotator in ANNOTATOR_PATHS:
                bin_anot_folder_path = os.path.join(bin_folder_path, annotator)
                test_creation.check_path(bin_anot_folder_path)
                
                mask_paths = glob.glob(os.path.join(mask_folder_path, annotator, '*.png'))
                num_cores = multiprocessing.cpu_count()
                n_jobs = len(mask_paths) if len(mask_paths) < num_cores else -3   

                print('Binarization starts . . .')
                Parallel(n_jobs=n_jobs, verbose=1)(delayed(test_creation.save_mask_to_binary)(
                    org_path=mask_paths[i], 
                    target_path=os.path.join(bin_anot_folder_path, mask_paths[i].split('/')[-1]), 
                    dominant_class='cancer')
                    for i in range(len(mask_paths)))
            
    print("generating full binary masks from full masks")
    full_mask_path = os.path.join(TEST_RECORD_DIRECTORY, 'masks_full')
    bin_folder_path = os.path.join(TEST_RECORD_DIRECTORY, 'binary_masks_full')
    test_creation.check_path(bin_folder_path)
    for annotator in ANNOTATOR_PATHS:
        bin_anot_folder_path = os.path.join(bin_folder_path, annotator)
        test_creation.check_path(bin_anot_folder_path)    
        
        mask_paths = glob.glob(os.path.join(full_mask_path, annotator, '*.png'))
        num_cores = multiprocessing.cpu_count()
        n_jobs = len(mask_paths) if len(mask_paths) < num_cores else -3         
        Parallel(n_jobs=n_jobs, verbose=1)(delayed(test_creation.save_mask_to_binary)(
            org_path=mask_paths[i], 
            target_path=os.path.join(bin_anot_folder_path, mask_paths[i].split('/')[-1]), 
            dominant_class='cancer')
            for i in range(len(mask_paths)))
    
    
    ## 5. Rearange files and folers
    
    print("\n Rearange files and folers")
    test_crop_paths = [crop_path for crop_path in glob.glob(os.path.join(TEST_RECORD_DIRECTORY, '*')) 
               if 'crop_size_' in crop_path]
    crop_paths = [crop_path for crop_path in glob.glob(os.path.join(RECORD_DIRECTORY, '*')) 
               if 'crop_size_' in crop_path]
    crop_paths.sort()   
    test_crop_paths.sort()
    
    # move patches images, masks etc    
    for test_path, def_path in zip(test_crop_paths, crop_paths):
        test_path_id = os.path.basename(test_path)
        def_path_id = os.path.basename(def_path)
        assert test_path_id == def_path_id, "crop size path mismatch for {}/{}".format(test_path_id, def_path_id)
        
        test_path = os.path.join(test_path, "test")
        def_path = os.path.join(def_path, "test")
        shutil.move(test_path, def_path)
        
    # move full size images, masks etc
    full_size_dir = os.path.join(RECORD_DIRECTORY, "test_set_full")
    os.makedirs(full_size_dir)
    shutil.move(os.path.join(TEST_RECORD_DIRECTORY, "images_full"), 
                os.path.join(full_size_dir, "images_full"))
    shutil.move(os.path.join(TEST_RECORD_DIRECTORY, "masks_full"), 
                os.path.join(full_size_dir, "masks_full"))
    shutil.move(os.path.join(TEST_RECORD_DIRECTORY, "binary_masks_full"), 
                os.path.join(full_size_dir, "binary_masks_full"))    
    
    # remove test temp dir
    shutil.rmtree(TEST_RECORD_DIRECTORY)   
    
    
    ## -- ##
    end = datetime.now()
    delta = end - start
    print(
        '\n\tDatasets generated in %d hours, %d minutes and %d seconds' % (
            delta.seconds // 3600, ((delta.seconds // 60) % 60),
            delta.seconds % 60))
    
if __name__ == "__main__":
    main()

    