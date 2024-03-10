import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import SimpleITK as sitk

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed


class MedDataset(Dataset):
    def __init__(self, im_root, gt_root, max_jump):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump

        self.volumes = []
        self.masks = []
        self.names = []
        vid_list = sorted(os.listdir(self.im_root))
        for vid in vid_list:
            if not vid.endswith('.nii.gz'):
                continue
            self.names.append(vid)
            vol_im_path = path.join(self.im_root, vid)
            vol_gt_path = path.join(self.gt_root, vid)

            self.volumes.append(self._load_volume(vol_im_path))
            self.masks.append(self._load_mask(vol_gt_path))

        print('%d out of %d volumes accepted in %s.' % (len(self.volumes), len(vid_list), im_root))

         # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.8,1.00), interpolation=InterpolationMode.BICUBIC)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.8,1.00), interpolation=InterpolationMode.NEAREST)
        ])

        # # Final transform without randomness
        # self.final_im_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     im_normalization,
        # ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def _load_volume(self, path, clip=(0,255)):
        assert path.endswith('.nii.gz')
        frames = sitk.GetArrayFromImage(sitk.ReadImage(path))
        min_val = np.percentile(frames, 1)
        max_val = np.percentile(frames, 99)
        assert max_val > min_val

        frames[frames < min_val] = min_val
        frames[frames > max_val] = max_val
        frames = clip[0] + (clip[1] - clip[0]) * (frames - min_val) / (max_val - min_val)
        frames = np.stack([frames, frames, frames], axis=3).astype(np.uint8)
        return frames

    def _load_mask(self, path):
        assert path.endswith('.nii.gz')
        frames = sitk.GetArrayFromImage(sitk.ReadImage(path))
        return np.expand_dims(frames, -1)

    def __getitem__(self, idx):
        volume = self.volumes[idx]
        gt = self.masks[idx]
        name = self.names[idx]
        num_frames = volume.shape[0]

        info = {}
        info['name'] = name

        trials = 0
        dist = None
        while trials < 5:
            info['frames'] = [] # store the frame index

            this_max_jump = min(num_frames - 1, self.max_jump)
            start_idx = np.random.randint(num_frames - 1 - this_max_jump)
            f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
            f1_idx = min(f1_idx, num_frames - 1)

            f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            f2_idx = min(f2_idx, num_frames - this_max_jump//2, num_frames - 1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            if np.random.rand() < 0.5:
                frames_idx = frames_idx[::-1]
            
            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_id in frames_idx:
                info['frames'].append(f_id)

                reseed(sequence_seed)
                this_im = Image.fromarray(volume[f_id]).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)

                this_gt = Image.fromarray(gt[f_id][:,:,0]).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)
            labels = np.unique(masks[0])
            # remove background
            labels = labels[labels!=0]

            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels != target_object]
                    second_object = np.random.choice(labels)

                # Compute distance from reference frame to target frame
                first_ref, tar_frame_int, secon_ref = frames_idx
                dist_1 = abs(int(first_ref)-tar_frame_int) / abs(int(first_ref)-int(secon_ref))
                dist_2 = abs(int(secon_ref)-tar_frame_int) / abs(int(first_ref)-int(secon_ref))
                dist = torch.FloatTensor([dist_1, dist_2])
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks == target_object).astype(np.float32)[:, np.newaxis, :, :]
        if has_second_object:
            sec_masks = (masks == second_object).astype(np.float32)[:, np.newaxis, :, :]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, 384, 384), dtype=int)
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'rgb': images,
            'gt': tar_masks,
            'cls_gt': cls_gt,
            'sec_gt': sec_masks,
            'selector': selector,
            'info': info,
            'dist': dist
        }

        return data

    def __len__(self):
        return len(self.volumes)