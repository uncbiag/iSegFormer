import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.med_test_dataset import MedTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore

from progressbar import progressbar


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--med_path', default='/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset_frames')
parser.add_argument('--output')
parser.add_argument('--split', help='val', default='val')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

med_path = args.med_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(f'/playpen-raid2/qinliu/data/AbdomenCT-1K/Organ-12-Subset_frames/trainval/Annotations/480p/Organ12_0001/0000000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

def evaluate_single_label(label):
    valid_obj_labels = (label,)

    # Setup Dataset
    if args.split == 'val':
        test_dataset = MedTestDataset(med_path+'/trainval', imset='val.txt', valid_obj_labels=valid_obj_labels)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        raise NotImplementedError

    # Load our checkpoint
    top_k = args.top
    prop_model = STCN().cuda().eval()

    # Performs input mapping such that stage 0 model can be loaded
    prop_saved = torch.load(args.model)
    for k in list(prop_saved.keys()):
        if k == 'value_encoder.conv1.weight':
            if prop_saved[k].shape[1] == 4:
                pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
                prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
    prop_model.load_state_dict(prop_saved)

    total_process_time = 0
    total_frames = 0

    # start evaluation. Only 1 round propagation.
    for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

        with torch.cuda.amp.autocast(enabled=args.amp):
            # example shape
            # rgb: [1, T, 3, 480, 480]; msk: [N, T, 1, 480, 480]
            # T is the number of frames; N is the number of organs        
            rgb = data['rgb'].cuda()
            msk = data['gt'][0].cuda()
            info = data['info']
            name = info['name'][0]
            k = len(info['labels'][0])
            size = info['size_480p']
            num_frames = info['num_frames'].numpy()[0]

            # find the best starting frame
            max_area, max_area_idx = -1, num_frames // 2
            for i in range(msk.shape[1]):
                area = torch.count_nonzero(msk[:,i])
                if area > max_area:
                    max_area = area
                    max_area_idx = i

            torch.cuda.synchronize()
            process_begin = time.time()

            processor = InferenceCore(prop_model, rgb, k, top_k=top_k, 
                            mem_every=args.mem_every, include_last=args.include_last)
            processor.interact(msk[:,max_area_idx], max_area_idx)

            # Do unpad -> upsample to original size 
            out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
            for ti in range(processor.t):
                prob = unpad(processor.prob[:,ti], processor.pad)
                prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                out_masks[ti] = torch.argmax(prob, dim=0)

            out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

            # cautious: no overlap between the two label sets
            # needs a label mapper and re-allocate the memory
            for idx, label in enumerate(valid_obj_labels):
                out_masks[out_masks == idx + 1] = label

            torch.cuda.synchronize()
            total_process_time += time.time() - process_begin
            total_frames += out_masks.shape[0]

            # Save the results
            this_out_path = path.join(out_path, 'label_'+'_'.join([str(i) for i in valid_obj_labels]), name)
            os.makedirs(this_out_path, exist_ok=True)
            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                img_E.putpalette(palette)
                img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

            del rgb
            del msk
            del processor

    print('Total processing time: ', total_process_time)
    print('Total processed frames: ', total_frames)
    print('FPS: ', total_frames / total_process_time)



if __name__ == '__main__':

    for label in range(1, 13):
        print(f'Processing label {label} ...')
        evaluate_single_label(label)