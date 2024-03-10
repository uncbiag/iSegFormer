from argparse import ArgumentParser
import numpy as np
import os
from os import path
from PIL import Image
from metrics import db_eval_boundary, db_eval_iou, db_statistics
import pandas as pd


def evaluate(gt_folder, seg_folder, dataset_file, label):

    videos = []
    with open(path.join(dataset_file), "r") as lines:
        for line in lines:
            video = line.rstrip('\n')
            videos.append(video)

    metrics_res_label = {}
    metrics_res_label['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    metrics_res_label['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for video in videos:
        print('Evaluating ', video)
        frames = os.listdir(os.path.join(gt_folder, video))
        iou_single_video_label = []
        bou_single_video_label = []
        for idx, frame in enumerate(frames):
            gt_frame_file = os.path.join(gt_folder, video, '{:07d}.png'.format(idx))
            gt_frame = np.array(Image.open(gt_frame_file).convert('P'), dtype=np.uint8)

            seg_frame_file = os.path.join(seg_folder, video, '{:05d}.png'.format(idx))
            seg_frame = np.array(Image.open(seg_frame_file).convert('P'), dtype=np.uint8)

            # calculate J, M, and DSC across frames
            gt_frame_label = np.zeros_like(gt_frame)
            gt_frame_label[gt_frame == int(label)] = 1
            if np.count_nonzero(gt_frame_label) > 10:
                seg_frame_label = np.zeros_like(seg_frame)
                seg_frame_label[seg_frame > 0] = 1
                iou_label = db_eval_iou(gt_frame_label, seg_frame_label)
                iou_single_video_label.append(iou_label)

                boundary_label = db_eval_boundary(gt_frame_label, seg_frame_label)
                bou_single_video_label.append(boundary_label)

        if len(iou_single_video_label) > 0:
            [JM, JR, JD] = db_statistics(np.array(iou_single_video_label))
            metrics_res_label['J']['M'].append(JM)
            metrics_res_label['J']['R'].append(JR)
            metrics_res_label['J']['D'].append(JD)
            metrics_res_label['J']['M_per_object'][video] = JM        

        if len(bou_single_video_label) > 0:
            [FM, FR, FD] = db_statistics(np.array(bou_single_video_label))
            metrics_res_label['F']['M'].append(FM)
            metrics_res_label['F']['R'].append(FR)
            metrics_res_label['F']['D'].append(FD)
            metrics_res_label['F']['M_per_object'][video] = FM

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']

    # results for label 6
    label_J = metrics_res_label['J']
    label_F = metrics_res_label['F']
    final_mean_label = (np.mean(label_J["M"]) + np.mean(label_F["M"])) / 2.
    g_res_label = np.array([final_mean_label, np.mean(label_J["M"]), np.mean(label_J["R"]), np.mean(label_J["D"]), \
        np.mean(label_F["M"]), np.mean(label_F["R"]), np.mean(label_F["D"])])
    g_res_label = np.reshape(g_res_label, [1, len(g_res_label)])
    table_g_label = pd.DataFrame(data=g_res_label, columns=g_measures)

    csv_path_label = os.path.join(seg_folder, f'label_{label}.csv')
    with open(csv_path_label, 'w') as f:
        table_g_label.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_path_label}')

    # Generate a dataframe for the per sequence results
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']

    csv_path_per_sequence_label = os.path.join(seg_folder, f'label_{label}_per.csv')
    seq_names_label = list(label_J['M_per_object'].keys())
    J_per_object_label = [label_J['M_per_object'][x] for x in seq_names_label]
    F_per_object_label = [label_F['M_per_object'][x] for x in seq_names_label]
    table_seq_label = pd.DataFrame(data=list(zip(seq_names_label, J_per_object_label, F_per_object_label)), columns=seq_measures)
    with open(csv_path_per_sequence_label, 'w') as f:
        table_seq_label.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_path_per_sequence_label}')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_folder')
    parser.add_argument('--seg_folder')
    parser.add_argument('--label')
    args = parser.parse_args()

    gt_folder = f'{args.data_folder}/trainval/Annotations/480p'
    dataset_file = f'{args.data_folder}/trainval/ImageSets/val.txt'
    seg_folder = f'{args.seg_folder}/label_{args.label}'
    evaluate(gt_folder, seg_folder, dataset_file, args.label)
