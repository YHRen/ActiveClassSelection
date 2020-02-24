"""
    Experiments on data sufficiency test. 
    Each Stage has 5000 samples
    5 Stages in total
    First stage is randomly sampled.

"""


import json
import time
import torch, torchvision
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import collections
import itertools as its
import argparse
from pathlib import Path

from dataset_utils import *
from procedure import train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment-name", help="experiment name",\
                          type=str, required=True)
    parser.add_argument("-b", "--batch-size", help="batch size", type=int, required=True)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, required=True)
    parser.add_argument("--combo-class-budget", help="number of samples after\
                        1st stage of the combo class, default=1000 \
                        (at random), [0, 5000]", type=int, default=1000)
    parser.add_argument("--gpu-device-id", help="GPU device id [=0]", type=int,
                          required=False, default=0)
    parser.add_argument("--output", help="directory to output model results",
                          type=str, default="./output/", required=False)
    parser.add_argument("--randseed", help="random seed for selecting a subset of the \
                          training data.", type=int, default=None, required=False)
    parser.add_argument("--augment", help="use data augmentation (random crop)\
                        ", action='store_true')
    args = parser.parse_args()
    (Path(args.output)/args.experiment_name).mkdir(parents=True, exist_ok=True)
    record_file = Path(args.output)/args.experiment_name/"result.json"
    record = collections.defaultdict(list)

    record['args'] = vars(args)
    # setup cifar10 data
    bsz = args.batch_size
    epochs = args.epochs
    device = torch.device('cuda', args.gpu_device_id)\
        if torch.cuda.is_available() else torch.device('cpu')

    ## Setup the DataLoader
    if not args.augment:
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    ## Setup the DataLoader
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)
    
    target_mapping = {0:0, 1:1, 2:2, 3:4, 4:3, 5:4, 6:4, 7:4, 8:4, 9:4}
    tt_train_set = TargetTransformDataset(train_set, target_mapping=target_mapping)
    tt_test_set = TargetTransformDataset(test_set, target_mapping=target_mapping)
    
    test_sampler = StatefulDataSampler(tt_test_set)
    test_sampler.add_samples(dict(zip(range(5), its.repeat(1000)))) ## for testing balanced class
    test_data = Subset(tt_test_set, test_sampler.get_samples())
    test_loader = DataLoader(test_data, batch_size=bsz, shuffle=False, num_workers=4)

    ## Setup the Active Learning
    train_sampler = StatefulDataSampler(tt_train_set,
                                        random_seed=args.randseed)
    marginal_increment = 1000
    train_loss_per_stage = []
    test_loss_per_stage = []
    test_acc_per_stage = []
    test_acc_best_per_stage = []
    record['time'].append(time.perf_counter())
    for stage in range(5): 
        if stage == 0: ## initial stage, random sampling
            train_sampler.add_samples(dict(zip(range(5), its.repeat(marginal_increment))))
        else:
            budget_per_class = [(5000-args.combo_class_budget)//4]*4 + [args.combo_class_budget]
            train_sampler.add_samples(dict(zip(range(5), budget_per_class)))
        train_data = Subset(tt_train_set, train_sampler.get_samples())
        #print(f"stage = {stage} with training data = {len(train_data)}")
        train_loader = DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=4)
        
        ## setup model, optim, loss_fn, summary_writer
        model = models.resnet18().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        summary_writer = SummaryWriter(args.output+"/"+args.experiment_name+"/"+f"stage_{stage}")
        best_test_acc = 0
        test_acc_this_stage = []
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, loss_fn, epoch, device, event_writer=summary_writer, log_interval=50)
            _, train_acc = test(model, train_loader, loss_fn, epoch, device, event_writer=summary_writer, test_type='train')
            test_loss, test_acc = test(model, test_loader, loss_fn, epoch, device, event_writer=summary_writer)
            test_acc_this_stage.append(test_acc)
            best_test_acc = max(best_test_acc, test_acc)
            #print(f">>> |epoch = {epoch} | train_loss = {train_loss} | train_acc = {train_acc} | test_loss = {test_loss} | test_acc = {test_acc} | ")
    
        train_loss_per_stage.append(train_loss)
        test_loss_per_stage.append(test_loss)
        test_acc_per_stage.append(test_acc)
        test_acc_best_per_stage.append(best_test_acc)
        record['test_acc_per_stage'].append(test_acc)
        record['test_best_acc_per_stage'].append(best_test_acc)
        record[f'stage{stage}_test_accuracy'] = test_acc_this_stage
        record['time'].append(time.perf_counter())


    with open(record_file, 'w') as fp:
        json.dump(record, fp)

