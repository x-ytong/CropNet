
CROP_TYPES = ['corn', 'soybeans', 'rice', 'wheat', 'sugarcane', 'cotton', 'other']
CROP_INDEX = {idx: crop for idx, crop in enumerate(CROP_TYPES)}


def label_mapping(train_data, crop_index):
    train_group = sorted(set(train_data['lbl']))
    train_crops = [crop_index[i] for i in train_group]
    train_index = {crop: idx for idx, crop in enumerate(train_crops)}

    old2new = {old: new for new, old in enumerate(train_group)}
    return train_index, old2new


def remap_prediction(test_pred, target, crop_label):
    if target == 'AUS':
        test_pred = [x if x in [crop_label['sugarcane'], crop_label['cotton']] else crop_label['other'] for x in test_pred]
    elif target == 'CHN':
        test_pred = [x if x in [crop_label['corn'], crop_label['soybeans'], crop_label['rice']] else crop_label['other'] for x in test_pred]
    elif target == 'ARG':
        test_pred = [x if x != crop_label['wheat'] else crop_label['other'] for x in test_pred]

    return test_pred
