import json
import numpy as np
from settings import NUM_TO_LABEL


def print_dataset_information(path_to_summary):
    """
    plots information about the dataset.

    Parameters:
    -----------
    path_to_summary: json file
        Path to json file generated along with the dataset
    """

    data_summary = json.load(open(path_to_summary, "r"))

    # Extract data
    mode = data_summary["mode"]
    info_crops = data_summary["info_crops"]
    info_pixels = data_summary["info_pixels"]

    print("\n****************************************************************")

    # Print data
    print("\n")
    print("Dataset summary: {}".format(path_to_summary.split("/")[-2]))
    print("================\n")
    print("mode: {}\n".format(mode))

    print('Total number of images: %d \n' % (np.sum(info_crops)))

    if mode != "test":
        # Normalise crops and pixels
        normalized_crops = info_crops / np.sum(info_crops)
        print("info about crops by central pixel class:")
        for i in range(len(NUM_TO_LABEL)):
            print('\t' + NUM_TO_LABEL[i] +
                  ': {:.4f}%'.format(normalized_crops[i] * 100))

    # Normalise pixels
    normalised_info_pixels = info_pixels / np.sum(info_pixels)
    print("\ninfo about pixels:")
    for i in range(len(NUM_TO_LABEL)):
        print('\t' + NUM_TO_LABEL[i] +
              ': {:.4f}%'.format(normalised_info_pixels[i] * 100))

    print("\n")
