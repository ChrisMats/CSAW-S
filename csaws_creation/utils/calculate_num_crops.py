

def calculate_num_crops(crop_size_target, num_crops_origin=10,
                        crop_size_origin=900):

    """
    From the number of crops when crop size is crop_size_origin, calculate an
    equivalent number of crops so that:
        (crop_size_origin ** 2) * number_of_crops_origin = \
            (crop_size_target ** 2) * number_of_crops_target
    """

    return num_crops_origin

    # return int(round(((crop_size_origin ** 2) * num_crops_origin) / (
    #     crop_size_target ** 2)))
