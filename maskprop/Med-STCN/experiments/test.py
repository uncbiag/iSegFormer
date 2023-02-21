import cv2
from PIL import Image
import numpy as np


#mask_path = '/work/data/DAVIS/2017/trainval/Annotations/480p/classic-car/00000.png'
mask_path = '/work/data/Internal/Abdomen1KDataset_frames/trainval/Annotations/Organ12_0001/0000020.png'
mask = Image.open(mask_path)
mask = np.array(mask)
print(np.count_nonzero(mask), mask.shape)
print(np.unique(mask), mask.dtype)