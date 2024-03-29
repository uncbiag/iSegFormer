import os
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image

davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'

# mask_folder = '/work/data/Internal/Abdomen1KDataset_volume/Mask_rib'
# mask_frames_folder = '/work/data/Internal/Abdomen1KDataset_frames/trainval/Annotations_rib/480'
# os.makedirs(mask_frames_folder, exist_ok=True)

size_480 = (480, 480)

def extract_images(volume_folder, volume_frames_folder):
    os.makedirs(volume_frames_folder, exist_ok=True)

    volumes = os.listdir(volume_folder)
    for volume in volumes:
        if volume.startswith('.'):
            continue

        print('volume: ', volume)
        volume_basename = volume[:-7]
        os.makedirs(os.path.join(volume_frames_folder, volume_basename), exist_ok=True)

        frames = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(volume_folder, volume)))
        min_val = np.percentile(frames, 1)
        max_val = np.percentile(frames, 99)
        assert max_val > min_val

        frames[frames < min_val] = min_val
        frames[frames > max_val] = max_val
        clip=(0, 255)
        frames = clip[0] + (clip[1] - clip[0]) * (frames - min_val) / (max_val - min_val)
        frames = np.stack([frames, frames, frames], axis=3).astype(np.uint8)

        frame_index = 0
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = cv2.resize(frame,dsize=size_480,interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(volume_frames_folder, volume_basename, f'{frame_index:07d}.jpg'), frame)

            frame_index += 1

def extract_masks(mask_folder, mask_frames_folder):
    os.makedirs(mask_frames_folder, exist_ok=True)

    masks = os.listdir(mask_folder)
    for mask in masks:
        if mask.startswith('.'):
            continue

        print('mask: ', mask)
        # mask_basename = mask[:12]
        mask_basename = mask[:-7]

        os.makedirs(os.path.join(mask_frames_folder, mask_basename), exist_ok=True)

        frames = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_folder, mask)))
        frames = frames.astype(np.uint8)
        # print(np.unique(frames), frames.dtype)
        frame_index = 0
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = Image.fromarray(frame)
            frame = frame.resize(size=size_480,resample=0)
            frame.putpalette(davis_palette)
            frame.save(os.path.join(mask_frames_folder, mask_basename, f'{frame_index:07d}.png'))
            # cv2.imwrite(os.path.join(mask_frames_folder, mask_basename, f'{frame_index:07d}.png'), frame)
            frame_index += 1


if __name__ == '__main__':

    DATASET = ''

    if DATASET == 'ABD1K':
        # AbdomenCT-1K
        volume_folder = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset/Image'
        volume_frames_folder = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset_frames/trainval/JPEGImages/480p'

        mask_folder = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset/Mask'
        mask_frames_folder = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset_frames/trainval/Annotations/480p'

    elif DATASET == 'MSD':
        # MSD
        volume_folder = '/playpen-raid2/qinliu/data/MSD/Task02_Heart/imagesTr'
        volume_frames_folder = '/playpen-raid2/qinliu/data/MSD/Task02_Heart_frames/trainval/JPEGImages/480p'

        mask_folder = '/playpen-raid2/qinliu/data/MSD/Task02_Heart/labelsTr'
        mask_frames_folder = '/playpen-raid2/qinliu/data/MSD/Task02_Heart_frames/trainval/Annotations/480p'

    else:
        # KiTS19
        volume_folder = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset/Image'
        volume_frames_folder = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset_frames/trainval/JPEGImages/480p'

        mask_folder = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset/Mask'
        mask_frames_folder = '/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset_frames/trainval/Annotations/480p'

    extract_images(volume_folder, volume_frames_folder)
    extract_masks(mask_folder, mask_frames_folder)
