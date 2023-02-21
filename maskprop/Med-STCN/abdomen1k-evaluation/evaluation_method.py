import numpy as np
import os
from os import path
from PIL import Image
from metrics import db_eval_boundary, db_eval_iou, db_statistics
import pandas as pd


# Only two labels (label 5 and 6) will be evaluated
gt_folder = '/work/data/Internal/Abdomen1KDataset_frames/trainval/Annotations/480p'
# seg_folder = '/work/results/stcn_med_abdomen1k/'
# seg_folder = '/work/results/stcn_med_abdomen1k_Jul20_19.26.32_ft_s012_400k_no_cc'
seg_folder = '/work/results/stcn_med_abdomen1k_Aug01_15.34.08_ft_s012_10k_no_cc'
seg_folder = '/work/results/stcn_med_abdomen1k_Aug01_22.03.33_ft_s012_10k_cc'
dataset_file = '/work/data/Internal/Abdomen1KDataset_frames/trainval/ImageSets/val.txt'

videos = []
with open(path.join(dataset_file), "r") as lines:
    for line in lines:
        video = line.rstrip('\n')
        videos.append(video)

metrics_res_label_5 = {}
metrics_res_label_5['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
metrics_res_label_5['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

metrics_res_label_6 = {}
metrics_res_label_6['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
metrics_res_label_6['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

iou_all_videos_label_5 = []
iou_all_videos_label_6 = []
bou_all_videos_label_5 = []
bou_all_videos_label_6 = []
for video in videos:
    print('Evaluating ', video)
    frames = os.listdir(os.path.join(gt_folder, video[:-5]))
    iou_single_video_label_5 = []
    iou_single_video_label_6 = []
    bou_single_video_label_5 = []
    bou_single_video_label_6 = []
    for idx, frame in enumerate(frames):
        gt_frame_file = os.path.join(gt_folder, video[:-5], '{:07d}.png'.format(idx))
        gt_frame = np.array(Image.open(gt_frame_file).convert('P'), dtype=np.uint8)

        seg_frame_file = os.path.join(seg_folder, video, '{:05d}.png'.format(idx))
        seg_frame = np.array(Image.open(seg_frame_file).convert('P'), dtype=np.uint8)

        # calculate J, M, and DSC across frames
        gt_frame_label_5 = np.zeros_like(gt_frame)
        gt_frame_label_5[gt_frame == 5] = 1
        if np.count_nonzero(gt_frame_label_5) > 10:
            seg_frame_label_5 = np.zeros_like(seg_frame)
            seg_frame_label_5[seg_frame == 5] = 1
            iou_label_5 = db_eval_iou(gt_frame_label_5, seg_frame_label_5)
            iou_single_video_label_5.append(iou_label_5)

            boundary_label_5 = db_eval_boundary(gt_frame_label_5, seg_frame_label_5)
            bou_single_video_label_5.append(boundary_label_5)            

        gt_frame_label_6 = np.zeros_like(gt_frame)
        gt_frame_label_6[gt_frame == 6] = 1
        if np.count_nonzero(gt_frame_label_6) > 10:
            seg_frame_label_6 = np.zeros_like(seg_frame)
            seg_frame_label_6[seg_frame == 6] = 1
            iou_label_6 = db_eval_iou(gt_frame_label_6, seg_frame_label_6)
            iou_single_video_label_6.append(iou_label_6)            

            boundary_label_6 = db_eval_boundary(gt_frame_label_6, seg_frame_label_6)
            bou_single_video_label_6.append(boundary_label_6)

    if len(iou_single_video_label_5) > 0:
        [JM, JR, JD] = db_statistics(np.array(iou_single_video_label_5))
        metrics_res_label_5['J']['M'].append(JM)
        metrics_res_label_5['J']['R'].append(JR)
        metrics_res_label_5['J']['D'].append(JD)
        metrics_res_label_5['J']['M_per_object'][video] = JM

    if len(iou_single_video_label_6) > 0:
        [JM, JR, JD] = db_statistics(np.array(iou_single_video_label_6))
        metrics_res_label_6['J']['M'].append(JM)
        metrics_res_label_6['J']['R'].append(JR)
        metrics_res_label_6['J']['D'].append(JD)
        metrics_res_label_6['J']['M_per_object'][video] = JM        

    if len(bou_single_video_label_5) > 0:
        [FM, FR, FD] = db_statistics(np.array(bou_single_video_label_5))
        metrics_res_label_5['F']['M'].append(FM)
        metrics_res_label_5['F']['R'].append(FR)
        metrics_res_label_5['F']['D'].append(FD)
        metrics_res_label_5['F']['M_per_object'][video] = FM

    if len(bou_single_video_label_6) > 0:
        [FM, FR, FD] = db_statistics(np.array(bou_single_video_label_6))
        metrics_res_label_6['F']['M'].append(FM)
        metrics_res_label_6['F']['R'].append(FR)
        metrics_res_label_6['F']['D'].append(FD)
        metrics_res_label_6['F']['M_per_object'][video] = FM

# Generate dataframe for the general results
g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']

# results for label 5
label_5_J = metrics_res_label_5['J']
label_5_F = metrics_res_label_5['F']
final_mean_label_5 = (np.mean(label_5_J["M"]) + np.mean(label_5_F["M"])) / 2.
g_res_label_5 = np.array([final_mean_label_5, np.mean(label_5_J["M"]), np.mean(label_5_J["R"]), np.mean(label_5_J["D"]), \
    np.mean(label_5_F["M"]), np.mean(label_5_F["R"]), np.mean(label_5_F["D"])])
g_res_label_5 = np.reshape(g_res_label_5, [1, len(g_res_label_5)])
table_g_label_5 = pd.DataFrame(data=g_res_label_5, columns=g_measures)

csv_path_label_5 = os.path.join(seg_folder, 'label_5.csv')
with open(csv_path_label_5, 'w') as f:
    table_g_label_5.to_csv(f, index=False, float_format="%.3f")
print(f'Global results saved in {csv_path_label_5}')

# results for label 6
label_6_J = metrics_res_label_6['J']
label_6_F = metrics_res_label_6['F']
final_mean_label_6 = (np.mean(label_6_J["M"]) + np.mean(label_6_F["M"])) / 2.
g_res_label_6 = np.array([final_mean_label_6, np.mean(label_6_J["M"]), np.mean(label_6_J["R"]), np.mean(label_6_J["D"]), \
    np.mean(label_6_F["M"]), np.mean(label_6_F["R"]), np.mean(label_6_F["D"])])
g_res_label_6 = np.reshape(g_res_label_6, [1, len(g_res_label_6)])
table_g_label_6 = pd.DataFrame(data=g_res_label_6, columns=g_measures)

csv_path_label_6 = os.path.join(seg_folder, 'label_6.csv')
with open(csv_path_label_6, 'w') as f:
    table_g_label_6.to_csv(f, index=False, float_format="%.3f")
print(f'Global results saved in {csv_path_label_6}')

# Generate a dataframe for the per sequence results
seq_measures = ['Sequence', 'J-Mean', 'F-Mean']

csv_path_per_sequence_label_5 = os.path.join(seg_folder, 'label_5_per.csv')
seq_names_label_5 = list(label_5_J['M_per_object'].keys())
J_per_object_label_5 = [label_5_J['M_per_object'][x] for x in seq_names_label_5]
F_per_object_label_5 = [label_5_F['M_per_object'][x] for x in seq_names_label_5]
table_seq_label_5 = pd.DataFrame(data=list(zip(seq_names_label_5, J_per_object_label_5, F_per_object_label_5)), columns=seq_measures)
with open(csv_path_per_sequence_label_5, 'w') as f:
    table_seq_label_5.to_csv(f, index=False, float_format="%.3f")
print(f'Per-sequence results saved in {csv_path_per_sequence_label_5}')

csv_path_per_sequence_label_6 = os.path.join(seg_folder, 'label_6_per.csv')
seq_names_label_6 = list(label_6_J['M_per_object'].keys())
J_per_object_label_6 = [label_6_J['M_per_object'][x] for x in seq_names_label_6]
F_per_object_label_6 = [label_6_F['M_per_object'][x] for x in seq_names_label_6]
table_seq_label_6 = pd.DataFrame(data=list(zip(seq_names_label_6, J_per_object_label_6, F_per_object_label_6)), columns=seq_measures)
with open(csv_path_per_sequence_label_6, 'w') as f:
    table_seq_label_6.to_csv(f, index=False, float_format="%.3f")
print(f'Per-sequence results saved in {csv_path_per_sequence_label_6}')
