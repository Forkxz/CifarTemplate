import os
import time
import yaml
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

import models
from utils.KD_loss import KD_loss
from utils.test import test
from utils.util import select_device, increment_dir, setup_seed, AverageMeter

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and name.islower() and callable(models.__dict__[name]))
optimizer_names = sorted(name for name in optim.__dict__ if not name.startswith("__") and callable(optim.__dict__[name]))

def main(args):
    setup_seed()
    log_dir = Path(args.logdir)
    os.makedirs(log_dir, exist_ok=True)
    wdir = log_dir / 'weights'  # weights directory
    os.makedirs(wdir, exist_ok=True)
    log_data = {'epoch': [], 'train loss': [],
                'test loss': [], 'test accuracy': []}
    best_accuracy = 0
    save_best = True
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'

    with open(log_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    device = select_device(args.device)
    # Create model
    if args.dataset == 'Cifar10':
        model = models.__dict__[args.student](num_classes=10).to(device)
    elif args.dataset == 'Cifar100':
        model = models.__dict__[args.student](num_classes=100).to(device)
    else:
        raise KeyError('dataset error')

    if args.Pretrain != '':
        state_dit = torch.load(args.Pretrain, map_location=device)
        model.load_state_dict(state_dit['Bit_dict'])
        del state_dit

    teacher_model = None
    if args.useTeacher:
        # teacher model is float
        if args.dataset == 'Cifar10':
            teacher_model = models.__dict__[args.teacher](num_classes=10).to(device)
        elif args.dataset == 'Cifar100':
            teacher_model = models.__dict__[args.teacher](num_classes=100).to(device)
        else:
            raise KeyError('dataset error')
        
        state_dit = torch.load(args.teacherDir, map_location=device)
        teacher_model.load_state_dict(state_dit['Bit_dict'])
        del state_dit
        teacher_model.eval()

    cudnn.benchmark = True
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    if args.dataset == 'Cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True,
                                            transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True,
                                           transform=transform_test)
    elif args.dataset == 'Cifar100':
        trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                            transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                           transform=transform_test)
    else:
        raise KeyError('dataset error')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
                                              shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        params += [{'params': [value], 'lr': base_lr, 'weight_decay':args.wd}]

    train_criterion = KD_loss() # for teacher model and student model
    optimizer = optim.__dict__[args.optimizer](params, lr=base_lr, weight_decay=args.wd)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - 0.2) + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if args.useTeacher:
        teacher_loss, teacher_acc = test(teacher_model, testloader, device)
        print(f"Teacher model loss = {teacher_loss}, Acc = {teacher_acc}")
        
    for epoch in range(args.start_epochs, args.epochs+args.start_epochs):
        train_loss,epoch_time = train(trainloader, model, teacher_model, train_criterion, optimizer, device)
        test_loss, test_acc = test(model, testloader, device)
        scheduler.step()
        print('\r Train Epoch: {}\t Time: {} \t Train Loss: {} \t Test Loss: {} \t ACC=: {}% \t LR: {}'.format(epoch, epoch_time, train_loss, test_loss, test_acc,optimizer.param_groups[0]['lr']), end="")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_best = True
        else:
            save_best = False

        # log
        log_data['epoch'].append(epoch)
        log_data['test accuracy'].append(test_acc)
        log_data['test loss'].append(test_loss)
        log_data['train loss'].append(train_loss)
        logfile = pd.DataFrame(data=log_data)
        logfile.to_csv(log_dir / 'log.csv')

        # save weights
        print('==> Saving model ...')
        Bit = {'epoch': epoch,
               'best_acc': test_acc,
               'Bit_dict': model.state_dict()}

        torch.save(Bit, last)
        if save_best:
            torch.save(Bit, best)
        del Bit
    return


def train(train_loader, model, teacher, criterion, optimizer, device):
    losses = AverageMeter('loss', ':.4e')
    model.train()
    start = time.time()
    if not teacher:
        criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        if teacher:
            with torch.no_grad():
                target = teacher(data)

        loss = criterion(output, target)
        assert (not torch.isnan(loss)) , "loss is nan"
        losses.update(loss.item(), data.size(0))
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

    
    return losses.avg, time.time()-start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--Pretrain', type=str, default='',
                        help='Pretrained model dir')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--logdir', type=str,
                        default='Debug', help='logging directory')

    parser.add_argument('--student', type=str, default='resnet20_cifar',choices=model_names,help='model architecture: ' +' | '.join(model_names))
    parser.add_argument('--teacher', type=str, default='resnet20_cifar',choices=model_names,help='model architecture: ' +' | '.join(model_names))
    parser.add_argument('--teacherDir', type=str, default='',help='Pretrained teacher model dir')
    parser.add_argument('--useTeacher',type= int, default=0)

    parser.add_argument('--data', action='store',
                        default='../data', help='dataset path')
    parser.add_argument('--dataset', action='store',
                        default='Cifar10', help='dataset Cifar10 or Cifar100')  

    parser.add_argument('--lr', default=1e-3,
                        help='the intial learning rate')
    parser.add_argument('--wd', action='store', default=0, #1e-5
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='AdamW',choices=optimizer_names,help='optimizers: ' +' | '.join(model_names))
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--start_epochs', type=int, default=1, help='number of epochs to train_start')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train_end')

    args = parser.parse_args()
    print('==> Options:', args)

    args.logdir = increment_dir(Path(args.logdir)/'exp', '')
    main(args)
