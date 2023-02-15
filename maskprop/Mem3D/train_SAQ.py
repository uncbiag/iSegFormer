from dataloaders.data import ROOT, DATA_CONTAINER, multibatch_collate_fn
from dataloaders.transform import TrainTransform, TestTransform
from utils.logger import Logger, AverageMeter
from utils.loss import *
from utils.utility import write_mask, save_checkpoint, adjust_learning_rate
from models_STM.models_SAQ import STM
from jaccard import eval_jaccard

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import argparse
import random
from progress.bar import Bar
from collections import OrderedDict

from options import OPTION as opt

MAX_FLT = 1e6
np.set_printoptions(threshold=1e6)
def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--gpu', default='', type=str, help='set gpu id to train the network, split with comma')
    return parser.parse_args()

def main():

    start_epoch = 0
    random.seed(0)

    args = parse_args()
    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '' else str(opt.gpu_id)

    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    # Data
    print('==> Preparing dataset')

    input_dim = opt.input_size

    train_transformer = TrainTransform(size=input_dim)
    test_transformer = TestTransform(size=input_dim)


    try:
        if isinstance(opt.trainset, list):
            datalist = []
            for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):
                ds = DATA_CONTAINER[dataset](
                    train=True, 
                    sampled_frames=opt.sampled_frames, 
                    transform=train_transformer, 
                    max_skip=max_skip, 
                    samples_per_video=opt.samples_per_video
                )
                datalist += [ds] * freq

            trainset = data.ConcatDataset(datalist)

        else:
            max_skip = opt.max_skip[0] if isinstance(opt.max_skip, list) else opt.max_skip
            trainset = DATA_CONTAINER[opt.trainset](
                train=True, 
                sampled_frames=opt.sampled_frames, 
                transform=train_transformer, 
                max_skip=max_skip, 
                samples_per_video=opt.samples_per_video
                )
    except KeyError as ke:
        print('[ERROR] invalide dataset name is encountered. The current acceptable datasets are:')
        print(list(DATA_CONTAINER.keys()))
        exit()

    testset = DATA_CONTAINER[opt.valset](
        train=False,
        transform=test_transformer,
        samples_per_video=1
        )

    trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                  collate_fn=multibatch_collate_fn, drop_last=True)

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)
    # Model
    print("==> creating model")

    net = STM(opt.keydim, opt.valdim, 'train', 
            mode=opt.mode, iou_threshold=opt.iou_threshold)
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    net.eval()

    net = nn.DataParallel(net).cuda()

    # set training parameters
    for p in net.parameters():
        p.requires_grad = True

    criterion = None
    celoss = binary_entropy_loss

    if opt.loss == 'ce':
        criterion = celoss
    elif opt.loss == 'iou':
        criterion = mask_iou_loss
    elif opt.loss == 'both':
        criterion = lambda pred, target, obj: celoss(pred, target, obj) + lovasz_softmax(pred, target)
    else:
        raise TypeError('unknown training loss %s' % opt.loss)

    optimizer = None
    
    if opt.solver == 'sgd':

        optimizer = optim.SGD(net.parameters(), lr=opt.learning_rate,
                        momentum=opt.momentum[0], weight_decay=opt.weight_decay)
    elif opt.solver == 'adam':

        optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate,
                        betas=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise TypeError('unkown solver type %s' % opt.solver)

    # Resume
    title = 'STM'
    minloss = float('inf')
    max_jaccard = 0.0

    opt.checkpoint = osp.join(osp.join(opt.checkpoint, opt.valset))
    if not osp.exists(opt.checkpoint):
        os.mkdir(opt.checkpoint)

    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        # opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume)
        minloss = checkpoint['minloss']
        start_epoch = checkpoint['epoch']
        max_jaccard = checkpoint['max_jaccard']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        skips = checkpoint['max_skip']
        
        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    trainloader.dataset.datasets[idx].set_max_skip(skip)
            else:
                trainloader.dataset.set_max_skip(skips)
        except:
            print('[Warning] Initializing max skip fail')

        logger = Logger(os.path.join(opt.checkpoint, opt.mode+'_log.txt'), resume=True)
    else:
        if opt.initial:
            print('==> Initialize model with weight file {}'.format(opt.initial))
            weight = torch.load(opt.initial)
            if isinstance(weight, OrderedDict):
                net.module.load_param(weight)
            else:
                net.module.load_param(weight['state_dict'])

        logger = Logger(os.path.join(opt.checkpoint, opt.mode+'_log.txt'), resume=False)
        start_epoch = 0

    logger.set_items(['Epoch', 'LR', 'Train Loss'])

    # Train and val
    for epoch in range(start_epoch):
        adjust_learning_rate(optimizer, epoch, opt)

    for epoch in range(start_epoch, opt.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.learning_rate))
        adjust_learning_rate(optimizer, epoch, opt)

        net.module.phase = 'train'
        train_loss = train(trainloader,
                           model=net,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           use_cuda=True,
                           iter_size=opt.iter_size,
                           mode=opt.mode,
                           threshold=opt.iou_threshold)

        if (epoch + 1) % opt.epoch_per_test == 0:
            net.module.phase = 'test'
            test_loss = test(testloader,
                            model=net.module,
                            criterion=criterion,
                            epoch=epoch,
                            use_cuda=True)

        # append logger file
        logger.log(epoch+1, opt.learning_rate, train_loss)

        # adjust max skip
        if (epoch + 1) % opt.epochs_per_increment == 0:
            if isinstance(trainloader.dataset, data.ConcatDataset):
                for dataset in trainloader.dataset.datasets:
                    dataset.increase_max_skip()
            else:
                trainloader.dataset.increase_max_skip()

                # save model
        # max_jaccard = 0.0
        J = eval_jaccard()
        is_best = J >= max_jaccard
        max_jaccard = max(J, max_jaccard)

        # is_best = train_loss <= minloss
        minloss = min(minloss, train_loss)
        skips = [ds.max_skip for ds in trainloader.dataset.datasets] \
                if isinstance(trainloader.dataset, data.ConcatDataset) \
                 else trainloader.dataset.max_skip

        if (epoch + 1) % opt.epoch_per_save == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'loss': train_loss,
                'minloss': minloss,
                'max_jaccard':max_jaccard,
                'optimizer': optimizer.state_dict(),
                'max_skip': skips,
            }, epoch + 1, is_best, checkpoint=opt.checkpoint, filename=opt.mode)

    logger.close()

    print('minimum loss:')
    print(minloss)
    print('max jaccard:')
    print(max_jaccard)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, iter_size, mode, threshold):
    # switch to train mode
    loss_func = nn.MSELoss()
    data_time = AverageMeter()
    loss = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, data in enumerate(trainloader):
        frames, masks, objs, infos = data
        data_time.update(time.time() - end)
        
        if use_cuda:
            frames = frames.cuda()
            masks = masks.cuda()
            objs = objs.cuda()

        objs[objs==0] = 1

        N, T, C, H, W = frames.size()
        max_obj = masks.shape[2]-1

        total_loss = 0.0
        out, quality, ious = model(frame=frames, mask=masks, num_objects=objs, criterion=mask_iou_loss)
        for idx in range(N):
            for t in range(1, T):
                gt = masks[idx, t:t+1]
                pred = out[idx, t-1: t]
                No = objs[idx].item()
                    
                total_loss = total_loss + criterion(pred, gt, No)

        total_loss = total_loss / (N * (T-1))
        # print('1 {}'.format(total_loss))
        total_loss += loss_func(quality*100,torch.Tensor(ious).cuda()*100)/1000
        # print('2 {}'.format(total_loss))
        # record loss
        if total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)
        # compute gradient and do SGD step (divided by accumulated steps)
        total_loss /= iter_size
        total_loss.backward()

        if (batch_idx+1) % iter_size == 0:
            optimizer.step()
            model.zero_grad()

        # measure elapsed time
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=loss.avg
        )
        bar.next()
    bar.finish()

    return loss.avg

def test(testloader, model, criterion, epoch, use_cuda):

    data_time = AverageMeter()

    bar = Bar('Processing', max=len(testloader))

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames, masks, objs, infos = data

            if use_cuda:
                frames = frames.cuda()
                masks = masks.cuda()
                
            frames = frames[0]
            masks = masks[0]
            num_objects = objs[0]
            info = infos[0]
            max_obj = masks.shape[1]-1
            # compute output
            t1 = time.time()

            T, _, H, W = frames.shape
            pred = [masks[0:1]]
            keys = []
            vals = []
            for t in range(1, T):
                if t-1 == 0:
                    tmp_mask = masks[0:1]
                elif 'frame' in info and t-1 in info['frame']:
                    # start frame
                    mask_id = info['frame'].index(t-1)
                    tmp_mask = masks[mask_id:mask_id+1]
                    num_objects = max(num_objects, tmp_mask.max())
                else:
                    tmp_mask = out
                # memorize
                key, val, _ = model(frame=frames[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_objects)

                # segment
                tmp_key = torch.cat(keys+[key], dim=1)
                tmp_val = torch.cat(vals+[val], dim=1)
                logits, ps, quality = model(frame=frames[t:t+1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_objects, max_obj=max_obj)

                out = torch.softmax(logits, dim=1)
                pred.append(out)

                if (t-1) % opt.save_freq == 0 and t<=100:
                    keys.append(key)
                    vals.append(val)
            
            pred = torch.cat(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            write_mask(pred, info, opt, directory=opt.output_dir)

            toc = time.time() - t1

            data_time.update(toc, 1)
           
            # plot progress
            bar.suffix  = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.sum
            )
            bar.next()
        bar.finish()

    return

if __name__ == '__main__':
    main()
