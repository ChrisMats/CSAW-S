import json
import random
import numpy as np
import pdb



def get_train_validation_split(segmentations_list,
                              percent_test,
                              percent_validation,
                              seed_test=2019,
                              seed_validation=2019,
                             random_split=False):
    """ splits the dataset into train, validation and test """
    if random_split:
        random.seed(seed_test)
        index_test = int(len(segmentations_list) * percent_test / 100)
        random.shuffle(segmentations_list)
        test = segmentations_list[:index_test]
        records_train = segmentations_list[index_test:]

        # split on patient level
        records_tr_patient_lvl = [rec.split('/')[-1].split('_')[0] for rec in records_train]
        records_tr_patient_lvl = np.unique(records_tr_patient_lvl).tolist()
        random.seed(seed_validation)
        index_validation = int(len(records_tr_patient_lvl) * percent_validation / 100)
        random.shuffle(records_tr_patient_lvl)
        validation_patient_lvl = records_tr_patient_lvl[:index_validation]
        train_patient_lvl = records_tr_patient_lvl[index_validation:]

        train = [rec for rec in records_train if rec.split('/')[-1].split('_')[0] in train_patient_lvl]
        validation = [rec for rec in records_train if rec.split('/')[-1].split('_')[0] in validation_patient_lvl]
    else:
        try:
            with open("suggested_validation_split.json") as f:
                validation_patient_lvl = json.load(f)
        except:
            raise FileNotFoundError("suggested_validation_split.json not found")
        validation = [rec for rec in segmentations_list if rec.split('/')[-1].split('_')[0] in validation_patient_lvl]
        train = [rec for rec in segmentations_list if rec.split('/')[-1].split('_')[0] not in validation_patient_lvl]
    
    return {"train": train, "val": validation}
