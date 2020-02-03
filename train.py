# coding: utf-8
import torch
import torch.nn
import torch.optim
import torch.nn.functional
from torch.nn import DataParallel
import torchvision.datasets
import torchvision.transforms
import torch.nn as nn

import numpy as np  # this is torch's wrapper for numpy
from matplotlib import pyplot
from matplotlib.pyplot import subplot
from sklearn.metrics import accuracy_score
from model.model import LeNet5
from model.model import Simple_CNN
from model.model import Simple_CNN_3d
from model.model import Simple_CNN_30
from data.data import Dataset
# from data.data_3d import Dataset
import cv2
import pandas as pd
import argparse
import os
import tensorboardX
import time
import csv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--epoch', default=2000, type=int)
    parser.add_argument('--num_worker', default=4, type=int)

    parser.add_argument('--train_csv_path',
                        default=['/home/huiying/luna/11.20/dataset/mhi/0_train.csv',
                                 '/home/huiying/luna/11.20/dataset/mhi/1_aug_train.csv'


    ], type=list)

    parser.add_argument('--val_csv_path',
                        default=['/home/huiying/luna/11.20/dataset/mhi/0_val_96.csv',
                                 '/home/huiying/luna/11.20/dataset/mhi/1_val.csv'

    ], type=list)

    parser.add_argument('--test_csv_path',
                        default=['/home/huiying/luna/11.20/dataset/mhi/0_test_96.csv',
                                 '/home/huiying/luna/11.20/dataset/mhi/1_test.csv'

    ], type=list)

    return parser.parse_args()


def train():
    # samples_path = '/home/huiying/luna/11.11/mhi_resize_to_48_48/z/up/test.csv'

    args = parse_args()
    t = time.strftime("%Y-%m-%d-%H:%M", time.localtime())

    model_save_path = 'batch' + str(args.batch_size) + '_lr' + str(args.lr) + '_weight_decay' + str(args.weight_decay) + '_time' + t
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.backends.cudnn.benchmark = True

    dataset_train = Dataset(args.train_csv_path)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)

    dataset_val = Dataset(args.val_csv_path)
    valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(in_features=512, out_features=2)

    # model = resnet
    model = Simple_CNN_30()
    # model = Simple_CNN()

    # resume the model
    if args.resume_path:
        state = torch.load(args.resume_path, map_location='cpu')
        model.load_state_dict(state, strict=True)

    # multi GPU training
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = tensorboardX.SummaryWriter(model_save_path)


    if args.resume_path:
        epoch_add = int(args.resume_path.split('_')[-3][2:])
    else:
        epoch_add = 0

    saved_model_dict = {10000: 'best_model'}
    t = time.time()

    for epoch in range(args.epoch - epoch_add):
        epoch += epoch_add
        model.train()


        # lr = get_lr(epoch, args)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr


        for batch_num, training_batch in enumerate(train_loader):
            inputs, labels = training_batch
            inputs, labels = inputs.to(device), labels.to(device)
            forward_output = model(inputs)
            loss = loss_func(forward_output, labels)
            writer.add_scalar('train_loss', loss.item(), epoch * len(train_loader) + batch_num)

            if batch_num % 100 == 0:
                print('Epoch:', epoch, '\t\tbatch_num:', batch_num, '\t\tLoss: %.5f' % loss.item(), '\t\tTime: %.5f' % (time.time() - t))

            t = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        loss_list = []
        with torch.no_grad():
            for batch_num, training_batch in enumerate(valid_loader):
                inputs, actual_val = training_batch
                inputs, actual_val = inputs.to(device), actual_val.to(device)
                predicted_val = model(inputs)
                loss = loss_func(predicted_val, actual_val)
                loss_list.append(loss.item())

            loss_mean = np.mean(np.array(loss_list, dtype=np.float32))
            writer.add_scalar('val_loss', loss_mean, epoch)
            save_model(saved_model_dict, loss_mean, model_save_path, epoch, optimizer, model)
            print('Epoch:', epoch, '\t\tValidation mean loss: %.5f' % loss_mean)


def get_lr(epoch, args):
    if epoch <= args.epoch * 0.25:
        lr = args.lr
    elif epoch <= args.epoch * 0.5:
        lr = args.lr * 0.1
    elif epoch <= args.epoch * 0.75:
        lr = args.lr * 0.01
    else:
        lr = args.lr * 0.001
    return lr



def save_model(saved_model_dict, loss_current, model_save_path, epoch, optimizer, model):
    sort = sorted(saved_model_dict.keys(), reverse=True)
    if sort[0] >= loss_current:
        key_del = sort[0]
        model_del = saved_model_dict[key_del]
        path_model_del = os.path.join(model_save_path, model_del)
        if os.path.exists(path_model_del):
            os.remove(path_model_del)
        del saved_model_dict[key_del]

        model_add = 'BEST_EP' + str(epoch) + '_LOSS%.4f' % loss_current
        saved_model_dict[loss_current] = model_add
        if torch.cuda.device_count() == 1:
            torch.save({'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(model_save_path, model_add))
        else:
            torch.save({'net': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(model_save_path, model_add))


# train()

