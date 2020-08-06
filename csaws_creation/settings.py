import os

# Target directory -- the directory of the original CSAW-S directory
TARGET_DIRECTORY = "../CsawS"


TARGET_DIRECTORY = os.path.abspath(TARGET_DIRECTORY)
if not os.path.isdir(TARGET_DIRECTORY):
    os.mkdir(TARGET_DIRECTORY)

# Test set prefix
TEST_PREFIX = "test_data"
TEST_TARGET_DIRECTORY = os.path.join(TARGET_DIRECTORY, TEST_PREFIX)

# Directory containing the anonymized dataset
ANONYMIZED_DATA_DIRECTORY = os.path.join(TARGET_DIRECTORY, "anonymized_dataset")
TEST_ANONYMIZED_DATA_DIRECTORY = os.path.join(TEST_TARGET_DIRECTORY, "anonymized_dataset")

# Annotator s paths
ANNOTATOR_PATHS = ["annotator_1", "annotator_2", "annotator_3"]

# Target folder for original images
IMAGES_DIRECTORY = os.path.join(TARGET_DIRECTORY, 'original_images')
TEST_IMAGES_DIRECTORY = os.path.join(TEST_TARGET_DIRECTORY, 'original_images')
if not os.path.isdir(IMAGES_DIRECTORY):
    os.mkdir(IMAGES_DIRECTORY)
if not os.path.isdir(TEST_IMAGES_DIRECTORY):
    os.mkdir(TEST_IMAGES_DIRECTORY)    

# Target folder for generated segmentation maps
SEGMENTATIONS_DIRECTORY = os.path.join(TARGET_DIRECTORY, "segmentation_maps")
TEST_SEGMENTATIONS_DIRECTORY = os.path.join(TEST_TARGET_DIRECTORY, "segmentation_maps")
if not os.path.isdir(SEGMENTATIONS_DIRECTORY):
    os.mkdir(SEGMENTATIONS_DIRECTORY)
if not os.path.isdir(TEST_SEGMENTATIONS_DIRECTORY):
    os.mkdir(TEST_SEGMENTATIONS_DIRECTORY)    

# Target folder for generated patches and final images
RECORD_DIRECTORY = os.path.join(TARGET_DIRECTORY, "patches")
TEST_RECORD_DIRECTORY = os.path.join(TEST_TARGET_DIRECTORY, "patches")
if not os.path.isdir(RECORD_DIRECTORY):
    os.mkdir(RECORD_DIRECTORY)
if not os.path.isdir(TEST_RECORD_DIRECTORY):
    os.mkdir(TEST_RECORD_DIRECTORY)    

# Validation split mode
RANDOM_VALIDATION_SPLIT = False

# Datasets parameters
DATASET_SPECS = {
    "crop_sizes": [512],
    "num_crops": 10
}

# List of dataset classes
CLASSES = [
    "cancer",
    "calcifications",
    "axillary_lymph_nodes",
    "thick_vessels",
    "foreign_object",
    "skin",
    "nipple",
    "text",
    "non-mammary_tissue",    
    "pectoral_muscle",
    "mammary_gland",
]

# Apply label smoothing to all but small objects
APPLY_SMOOTHING = False

# List os small classes in the dataset
CLASSES_SMALL = [
    "calcifications",
    "nipple",
    "axillary_lymph_nodes"
]

# Dictionary from label_num to label
NUM_TO_LABEL = {number: label for number, label in enumerate(CLASSES)}
NUM_TO_LABEL[len(CLASSES)] = "background"
LABEL_TO_NUM = {v: k for k, v in NUM_TO_LABEL.items()}
