import torch
import time
import numpy as np
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import gc
from utils.map import MAP


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:] = -float('inf')


def evaluate_cprompt(model: ContinualModel, dataset: ContinualDataset, 
                        args: Namespace, t, last=False) -> Tuple[list, list]:
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    # using ema
    if args.use_ema or args.use_ema_c:
        model.ema_before_eval(t)

    if args.dataset == "domain-net":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-c":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-r":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-cr":
        test_loaders = dataset.get_test_dataloaders(t)

    with torch.no_grad():
        for k, test_loader in enumerate(test_loaders):
            if last and k < len(dataset.test_loaders) - 1:
                continue
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            tmp = 0
            for data in test_loader:
                tmp += 1
                if tmp%50 == 0:
                    print(tmp," / ", len(test_loader))
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model.forward_model(inputs)             
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()     
                
                total += labels.shape[0]
                # print("correct:", correct)
                # print("total:", total)`
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
            
            accs.append(correct / total * 100)
            accs_mask_classes.append(correct_mask_classes / total * 100)
    
    model.net.train(status)

    # using ema
    if args.use_ema or args.use_ema_c:
        model.ema_after_eval()

    l = len(accs)
    for i in range(l):
        accs[i] = round(accs[i], 2)
        accs_mask_classes[i] = round(accs_mask_classes[i], 2)

    return accs, accs_mask_classes






def train_cprompt(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:

    model.net.to(model.device)
    print(model.device)
    
    results, results_mask_classes = [], []
    
    model_stash = create_stash(model, args, dataset)
    
    print(file=sys.stderr)
    start_time = time.time()
    for t in range(dataset.N_TASKS):
        gc.collect()
        model.net.train()
        if args.dataset == "domain-net":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-c":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-r":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-cr":
            train_loader = dataset.get_data_loaders(task_id=t)

        model.init_opt(args, t)

        
        for epoch in range(args.n_epochs):
            epoch_start = time.time()
            for i, data in enumerate(train_loader):
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                loss = model.observe(inputs, labels,dataset,t)

                if args.adapt_ema_c:
                    model.cal_g_change()

                # update ema_c
                if args.use_ema_c and args.update_ema_c == 'batch':
                    model.ema_c.update()
                if args.use_ema and args.update_ema_g == 'batch':
                    model.update_ema_g()

                progress_bar(i, len(train_loader), epoch, t, loss, args)
                
                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0


            if args.use_ema_c and args.update_ema_c == 'epoch':
                model.ema_c.update()
            if args.use_ema and args.update_ema_g == 'epoch':
                model.update_ema_g()


        if args.use_ema_c and args.update_ema_c == 'task':
            model.ema_c.update()
        if args.use_ema and args.update_ema_g == 'task':
            model.update_ema_g()


        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0
        

        # save the prompt and model
        if args.save_prompt and args.model != 'finetune_vit' and args.model != "classifier_vit":
            model.save_prompt_to_file(t)
        if args.save_model:
            model.save_model_to_file(t)
        if args.use_ema_c and args.save_ema:
            model.save_ema_to_file(t)
            
        torch.cuda.empty_cache()

        accs = evaluate_cprompt(model, dataset, args, t)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        
        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(args, mean_acc, t + 1, dataset.SETTING)
        save_mean_accuracy_to_file(mean_acc, t + 1, dataset.SETTING, args, t, accs, loss)
        
        # freeze G Prompt
        if t == 0 and args.freeze_g_prompt:
            model.set_g_prompt_not_update()

        model_stash['mean_accs'].append(mean_acc)
        tmp = time.time() - start_time

    running_time = time.time() - start_time

