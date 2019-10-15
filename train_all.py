import argparse
import glob
import itertools
import os
import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
#import torchsample
from PIL import Image
from torch.autograd import Variable

#import imagenet_utils
#import common
#from pretrained_utils import get_relevant_classes
import pytorch_resnet
from pytorch_utils import *
import torch.utils.model_zoo as model_zoo
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import functools
import random 
import csv
import sys

data_path = "/home/leander/hcc/prunWeight/data4"

TRAIN_PATH = os.path.join(data_path, 'train/')
#TRAIN_PATH = "/lfs/raiders3/1/ddkang/imagenet/ilsvrc2012/ILSVRC2012_img_train"
VAL_PATH = os.path.join(data_path, 'val/')
#VAL_PATH = "/lfs/raiders3/1/ddkang/imagenet/ilsvrc2012/ILSVRC2012_img_val"

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help="Small model")
parser.add_argument('--resol', default=224, type=int, help="Resolution")
parser.add_argument('--temp', required=True, help="Softmax temperature")
parser.add_argument('--gpu', default=None, type=int,
                help='GPU id to use.')  
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')

def main():
  
	global args
	args = parser.parse_args()
	pytorch_models = {
    	'resnet18': models.resnet18(pretrained=True),
    	'resnet34': models.resnet34(pretrained=True),
    	'resnet50': models.resnet50(pretrained=True),
    	'resnet101': models.resnet101(pretrained=True),
    	'resnet152': models.resnet152(pretrained=True)
    }
	model_params = [
    		('trn2', []),
    		('trn4', [1]),
    		('trn6', [1, 1]),
    		('trn8', [1, 1, 1]),
    		('trn10', [1, 1, 1, 1]),
    		('trn18', [2, 2, 2, 2]),
    		('trn34', [3, 4, 6, 3])]
	name_to_params = dict(model_params)

	big_model = pytorch_models['resnet18']

	for p in big_model.parameters():
		p.requires_grad=False
		p.cuda(args.gpu)
		
	if args.model.startswith('trn'):
		small_model = pytorch_resnet.rn_builder(name_to_params[args.model],num_classes=4,
			conv1_size=3, conv1_pad=1, nbf=16,downsample_start=False)
	else:
		small_model = models.__dict__[args.model]()

	if args.gpu is not None:
		big_model = big_model.cuda(args.gpu) 
		num_ftrs = big_model.fc.
        print("ok")
		big_model.fc = nn.Linear(num_ftrs, 4)

        if args.model.startswith('alexnet'):
            small_model = small_model.cuda(args.gpu) 
            small_model.cuda()
            num_ftrs = small_model.classifier[6].in_features
            small_model.classifier[6] = nn.Linear(num_ftrs, 4)
        else:
            small_model = small_model.cuda(args.gpu) 

	else:
		big_model = torch.nn.DataParallel(big_model).cuda()
		num_ftrs = big_model.module.fc.in_features
		big_model.module.fc = nn.Linear(num_ftrs, 4)

		if args.model.startswith('alexnet'):
			small_model.features = torch.nn.DataParallel(small_model.features)
			small_model.cuda()
			num_ftrs = small_model.classifier[6].in_features
			small_model.classifier[6] = nn.Linear(num_ftrs, 4)
		else:
			small_model = torch.nn.DataParallel(small_model).cuda()


    ##################
	
	optimizer = optim.Adam(big_model.parameters(),lr=0.001)
	criterion = nn.CrossEntropyLoss().cuda(args.gpu)
	train_loader, val_loader = get_datasets()#train_fnames, val_fnames)
	
	big_model.train(True)
	big_model.cuda(args.gpu)
	for epoch in range(0, args.epochs):
		print("===epoc===%d"%epoch)
		for i,(data,y) in enumerate(train_loader):
			data=Variable(data,requires_grad=True)
            #y=Variable(y,requires_grad=True)

			#if args.gpu is not None:
			data = data.cuda(args.gpu, non_blocking=True)
			y = y.cuda(args.gpu, non_blocking=True)
			
			out = big_model(data)
			loss=criterion(out,y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('loss:',loss,loss.item())

	big_model.train(False)

	test_acc0 = validate(val_loader, big_model, criterion)

    ##################


	train(big_model, small_model, args)


def load_all_data(path):
    """
    return list of all img files and labels
    """
    all_classes = os.listdir(path)
    all_data = []
    for c in all_classes:
        class_path = os.path.join(path, c)
        image_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
        all_data.append(image_files)
    random.shuffle(all_data)
    return all_data


def train(big_model, small_model, args):

    RESOL = args.resol
    NB_CLASSES = 4
    print("Loading images...")
    #train_fnames = load_all_data(TRAIN_PATH)
    #val_fnames = load_all_data(VAL_PATH)

    # BASE_DIR = FILE_BASE
    # if not os.path.exists(BASE_DIR):
    #     try:
    #         os.mkdir(BASE_DIR)
    #     except:
    #         pass

    SMALL_MODEL_NAME = args.model
    TEMPERATURE = int(args.temp)
    train_loader, val_loader = get_datasets()#train_fnames, val_fnames)
    s1 = '%s-%d-%d-epoch{epoch:02d}-sgd-cc.t7' % (SMALL_MODEL_NAME, TEMPERATURE, RESOL)
    s2 = '%s-%d-%d-best-sgd-cc.t7' % (SMALL_MODEL_NAME, TEMPERATURE, RESOL)

    #big_model = nn.Sequential(*list(big_model.features.children())[:-1])
    #print (big_model)
    #print ""
    #small_model = nn.Sequential(*list(small_model.features.children())[:-1])
    #print (small_model)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(small_model.parameters(), 0.1,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    best_acc = trainer(big_model, small_model, TEMPERATURE, criterion, optimizer, scheduler,
                        (train_loader, val_loader),
                        nb_epochs=100, model_ckpt_name=s1, model_best_name=s2,
                        scheduler_arg='loss', save_every=10)
    #best_f1_val, best_f1_epoch = best_f1
    best_acc_val, best_acc_epoch = best_acc

    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.model, args.resol, args.temp, best_acc_val, best_acc_epoch])

    # Touch file at end
    #open('%s.txt' % FILE_BASE, 'a').close()

def validate(val_loader, model, criterion):
    # AverageMeter() : Computes and stores the average and current value
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True) 
            output = model(input)
            loss = criterion(output, target)

            prec1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    #view() means resize() -1 means 'it depends'
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True) 
        pred = pred.t() # a zhuanzhi transpose xcol 5row
        correct = pred.eq(target.view(1, -1).expand_as(pred)) #expend target to pred
        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        return acc

if __name__ =='__main__':
    main()
