import datasets
import models
from optimizer import adjust_learning_rate
from optimizer import get_criterion
from utils.helper import AverageMeter
from utils.helper import save_checkpoint
from utils.helper import post_checkpoint_to_eval

import argparse
import math
import numpy as np
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from tensorboardX import SummaryWriter
import datetime
import torchvision
best_criterion_val = 0
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-root', type=str, default=None,
                        help='path to train dataset')
    parser.add_argument('--val-root', type=str, default=None,
                        help='path to val dataset')
    parser.add_argument('--h', type=int, default=32,
                        help='the height of the input image to network')
    parser.add_argument('--w', type=int, default=100,
                        help='the width of the input image to network')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='mini-batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--warm-up', type=int, default=None,
                        help='warm up')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', type=int, default=-1,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=-1,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:23456',
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu id to use')
    parser.add_argument('--eval-url', type=str, default=None,
                        help='url used to eval model')
    parser.add_argument('--tfboard', type=str, default=None,
                        help='tensorboard path for logging')
    parser.add_argument('--model', type=str, default='single',
                        help='tensorboard path for logging')
    return parser.parse_args()

def main(args):
    print('Setting Arguments:', args)

    if args.tfboard and not args.evaluate:
        if os.path.exists(args.tfboard):
            shutil.rmtree(args.tfboard)
        args.writer = SummaryWriter(args.tfboard)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        return main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.world_size = args.world_size * ngpus_per_node
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = models.create_model(args,args.model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer=optim.Adam(model.parameters(),args.lr)
    if args.checkpoint:
        print('=> loading pytorch checkpoint ... '.format(args.checkpoint))
        map_location = None if args.gpu is None else 'cuda:{}'.format(args.gpu)
        state = torch.load(args.checkpoint, map_location=map_location)
        args.start_epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        print('=> loaded pytorch checkpoint {} (epoch {})'.format(args.checkpoint, args.start_epoch))
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    # from thop import profile
    # input=torch.rand(1,1,32,100).cuda()
    # macs, params = profile(model, inputs=(input, ))
    # print('model MACs',macs/1e9,' G')
    # print('modle Params',params/1e6,'M')
    criterion = get_criterion('ctc').cuda(args.gpu)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)

    # if args.checkpoint:
    #     print('=> loading pytorch checkpoint ... '.format(args.checkpoint))
    #     map_location = None if args.gpu is None else 'cuda:{}'.format(args.gpu)
    #     state = torch.load(args.checkpoint, map_location=map_location)
    #     args.start_epoch = state['epoch']
    #     model.load_state_dict(state['state_dict'])
    #     optimizer.load_state_dict(state['optimizer'])
    #     print('=> loaded pytorch checkpoint {} (epoch {})'.format(args.checkpoint, args.start_epoch))
    from datasets.dataset import Plate
    if not args.evaluate:
        train_dataset =Plate(root=args.train_root)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
    
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   collate_fn=datasets.CRNNCollate(h=args.h, w=args.w))

    val_dataset = Plate(root=args.val_root)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             collate_fn=datasets.CRNNCollate(h=args.h, w=args.w))

    if args.evaluate:
        return validate(val_loader, model, criterion, args)
    validate(val_loader, model, criterion, args)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.world_size == 0):
            state = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }

            if args.eval_url is None:
                global best_criterion_val
                criterion_val = validate(val_loader, model, criterion, args)
                is_best = criterion_val > best_criterion_val
                best_criterion_val = max(criterion_val, best_criterion_val)
                state['criterion'] = criterion_val
                save_checkpoint(state, args.checkpoint_dir, is_best)
            else:
                checkpoint_file = save_checkpoint(state, args.checkpoint_dir)
                post_checkpoint_to_eval(args.eval_url, args.val_root, checkpoint_file)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    forward_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    model.train()

    tic = time.time()
    for i, (images, labels, targets, target_lengths) in enumerate(train_loader):
        data_time_meter.update(time.time() - tic)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        lr = adjust_learning_rate(optimizer, epoch, len(train_loader), i, args)

        batch_size = images.size(0)

        forward_tic = time.time()
        preds = model(images)
        preds = preds.permute(1, 0, 2)
        input_lengths = torch.IntTensor([preds.size(0)] * batch_size)
        loss = criterion(preds, targets, input_lengths, target_lengths) / batch_size
        forward_time_meter.update(time.time() - forward_tic)

        loss_meter.update(loss.item())

        _, whole_accuracy = compute_accuracy(labels, preds.cpu().detach().numpy())
        accuracy_meter.update(whole_accuracy, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.world_size == 0):
            if i % args.print_freq == 0:
                print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
                print('Train Epoch: [{}][{}/{}]\t'
                      'Time {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t'
                      'Data {data_time_meter.val:.3f} ({data_time_meter.avg:.3f})\t'
                      'Forward {forward_time_meter.val:.3f} ({forward_time_meter.avg:.3f})\t'
                      'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                      'Accuracy {accuracy_meter.val:.3f} ({accuracy_meter.avg:.3f})\t'
                      'LR {lr:.6f}'
                      .format(epoch, i, len(train_loader),
                              batch_time_meter=batch_time_meter,
                              data_time_meter=data_time_meter,
                              forward_time_meter=forward_time_meter,
                              loss_meter=loss_meter,
                              accuracy_meter=accuracy_meter,
                              lr=optimizer.param_groups[-1]['lr']),
                      flush=True)

            if args.tfboard:
                current_iter = epoch * len(train_loader) + i
                args.writer.add_scalars('data/loss',
                                        {'loss_val': loss_meter.val, 'loss_avg': loss_meter.avg},
                                        current_iter)
                args.writer.add_scalars('data/accuracy',
                                        {'accuracy_val': accuracy_meter.val, 'accuracy_avg': accuracy_meter.avg},
                                        current_iter)
                args.writer.add_scalar('data/lr', optimizer.param_groups[-1]['lr'], current_iter)

        batch_time_meter.update(time.time() - tic)
        tic = time.time()


def validate(val_loader, model, criterion, args):
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    forward_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    model.eval()

    with torch.no_grad():
        tic = time.time()
        nowTime = 'log/'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.txt'  # 现在
        log_files=open(nowTime,'w')
        for i, (images, labels, targets, target_lengths) in enumerate(val_loader):
            data_time_meter.update(time.time() - tic)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            batch_size = images.size(0)
            forward_tic = time.time()
            preds = model(images)
            preds = preds.permute(1, 0, 2)
            input_lengths = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, targets, input_lengths, target_lengths) / batch_size
            forward_time_meter.update(time.time() - forward_tic)

            loss_meter.update(loss.item())

            _, whole_accuracy = compute_accuracy(labels, preds.cpu().numpy(),i,valid=True,log_files=log_files,batch_size=args.batch_size)
            accuracy_meter.update(whole_accuracy, batch_size)

            if i % args.print_freq == 0:
                print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
                print('Test: [{}/{}]\t'
                      'Time {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t'
                      'Data {data_time_meter.val:.3f} ({data_time_meter.avg:.3f})\t'
                      'Forward {forward_time_meter.val:.3f} ({forward_time_meter.avg:.3f})\t'
                      'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})'
                      .format(i, len(val_loader),
                              batch_time_meter=batch_time_meter,
                              data_time_meter=data_time_meter,
                              forward_time_meter=forward_time_meter,
                              loss_meter=loss_meter),
                      flush=True)

            batch_time_meter.update(time.time() - tic)
            tic = time.time()
    log_files.close()
    print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
    print('Accuracy {accuracy_meter.avg:.3f}'.format(accuracy_meter=accuracy_meter), flush=True)

    return accuracy_meter.avg


def compute_accuracy(ground_truth, predictions,batch_index=0,batch_size=100,valid=False,log_files=None):
    from utils.helper import StrLabelConverter
    from utils.keys import plate_keys

    converter = StrLabelConverter(plate_keys)
    predictions = np.argmax(predictions, axis=2).transpose(1, 0)

    preds_labels, _ = converter.decode(predictions)

    char_accuracy, whole_accuracy = [], []
    for index, label in enumerate(ground_truth):
        prediction = preds_labels[index]
        index_=batch_index*batch_size+index
        if label.lower() == prediction.lower():
            whole_accuracy.append(1)
        else:
            whole_accuracy.append(0)
            if valid:
                log_files.write(str(index_))
                log_files.write('\n')
                log_files.write(prediction.upper())
                log_files.write('\n')
                log_files.write(label.upper())
                log_files.write('\n')
        total_count = len(label)
        correct_count = 0
        try:
            for i, tmp in enumerate(label):
                if tmp.lower() == prediction[i].lower():
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                char_accuracy.append(float(correct_count) / total_count)
            except ZeroDivisionError:
                if len(prediction) == 0:
                    char_accuracy.append(1)
                else:
                    char_accuracy.append(0)
    char_accuracy = np.mean(np.array(char_accuracy).astype(np.float32), axis=0)
    whole_accuracy = np.mean(np.array(whole_accuracy).astype(np.float32), axis=0)

    return char_accuracy, whole_accuracy


if __name__ == '__main__':
    args = parse_args()
    main(args)
