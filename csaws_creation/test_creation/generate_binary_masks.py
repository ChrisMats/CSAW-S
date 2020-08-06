import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from settings import  TEST_RECORD_DIRECTORY, NUM_TO_LABEL, ANNOTATOR_PATHS

LABEL_TO_NUM = {v: k for k, v in NUM_TO_LABEL.items()}

def check_path(fname):
    if not os.path.isdir(fname):
        os.mkdir(fname)
        
def save_mask_to_binary(org_path, target_path, dominant_class='cancer'):
    annotation_crop = np.array(Image.open(org_path))
    annotation_crop = annotation_crop.astype(np.int)
    annotation_crop[annotation_crop == LABEL_TO_NUM[dominant_class]] = -1
    annotation_crop[annotation_crop != -1] = 0
    annotation_crop[annotation_crop == -1] = 1
    annotation_crop = Image.fromarray(annotation_crop.astype(np.uint8))
    annotation_crop.save(target_path)
    
if __name__ == "__main__":      

    print("generating croped binary masks from croped masks")
    crop_paths = [crop_path for crop_path in glob.glob(os.path.join(TEST_RECORD_DIRECTORY, '*')) 
               if 'crop_size_' in crop_path]

    for crop_path_size in tqdm(crop_paths):
        for mode in ["test"]:
            folder_path = os.path.join(crop_path_size, mode)
            bin_folder_path = os.path.join(folder_path, 'binary_masks')
            mask_folder_path = os.path.join(folder_path, 'masks')
            check_path(bin_folder_path)
            for annotator in ANNOTATOR_PATHS:
                bin_anot_folder_path = os.path.join(bin_folder_path, annotator)
                check_path(bin_anot_folder_path)
                for mask_path in glob.glob(os.path.join(mask_folder_path, annotator, '*.png')):
                    mask_id = mask_path.split('/')[-1]
                    binary_mask_path = os.path.join(bin_anot_folder_path, mask_id)
                    save_mask_to_binary(mask_path, binary_mask_path)

    print("generating full binary masks from full masks")
    full_mask_path = os.path.join(TEST_RECORD_DIRECTORY, 'masks_full')
    bin_folder_path = os.path.join(TEST_RECORD_DIRECTORY, 'binary_masks_full')
    check_path(bin_folder_path)
    for annotator in ANNOTATOR_PATHS:
        bin_anot_folder_path = os.path.join(bin_folder_path, annotator)
        check_path(bin_anot_folder_path)    
        for mask_path in glob.glob(os.path.join(full_mask_path, annotator, '*.png')):
            mask_id = mask_path.split('/')[-1]
            binary_mask_path = os.path.join(bin_anot_folder_path, mask_id)
            save_mask_to_binary(mask_path, binary_mask_path)
