# https://stackoverflow.com/questions/56650201/how-to-convert-35-classes-of-cityscapes-dataset-to-19-classes

import numpy as np

# Standard Cityscapes mapping: 35 classes → 19 classes + ignore_index=255
# This is the most widely used and benchmark-compatible mapping
id_to_trainid = {
    0: 255,   # unlabeled / void
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,
    10: 255,
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,    # pole
    18: 255,
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,
    30: 255,
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    -1: 255
}

def encode_labels(mask):
    """
    mask: numpy array (H, W) with original Cityscapes labelIds (0-33 or -1)
    Returns: numpy array (H, W) with values in 0..18 or 255 (ignore)
    """
    # Vectorized version - much faster
    label_mask = np.full_like(mask, 255, dtype=np.uint8)   # default = ignore
    
    for orig_id, train_id in id_to_trainid.items():
        label_mask[mask == orig_id] = train_id
    
    return label_mask