gt_root = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset/Mask'
seg_root = '/playpen-raid2/qinliu/projects/iSegFormer/maskprop/Med-STCN/results/ABD1K'

result_bl = f'{seg_root}/stcn'
result_ft = f'{seg_root}/stcn_med_abdomen1k_Aug01_15.34.08_ft_s012_10k_no_cc'
result_cc = f'{seg_root}/stcn_med_abdomen1k_Aug01_22.03.33_ft_s012_10k_cc'

import os
import SimpleITK as sitk
import cv2 
import numpy as np
import matplotlib.pyplot as plt


volume_idx = 'Organ12_0001'
gt_path = f'{gt_root}/{volume_idx}.nii.gz'
gt = sitk.ReadImage(gt_path)

gt_npy = sitk.GetArrayFromImage(gt)
seg = np.zeros_like(gt_npy)

for label_idx in range(1, 13):
    
    folder_cc = f'{result_cc}/label_{label_idx}/{volume_idx}'

    slices = []
    slice_names_all = os.listdir(folder_cc)
    for slice_name in slice_names_all:
        slice = cv2.imread(f'{folder_cc}/{slice_name}')[:,:,0]
        slice = cv2.resize(slice, (512, 512), interpolation=cv2.INTER_NEAREST)
        slice[slice>0] = label_idx

        slices.append(slice)
    slices = np.stack(slices, axis=0)
    seg[slices > 0] = label_idx

sitk.WriteImage(seg, f'{result_cc}/{volume_idx}.nii.gz')
    