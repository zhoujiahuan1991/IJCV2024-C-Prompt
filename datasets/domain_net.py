# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from transformers import  ViTForImageClassification
import torch
import torch.nn as nn
import os
import random
import numpy as np
from timm.models import create_model

class DomainNet(Dataset):
    """
    Overrides the dataset to change the getitem function.
    """
    def __init__(self, domain_id=0, mode='train', transform=None,
                 target_transform=None) -> None:
        super().__init__()
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform
        self.target_transform = target_transform

        self.mode = mode
        self.data_root = "your_data_path/domainnet"
        self.image_list_root = "./datasets/image_list/domainnet_list"
        self.task_name = ['real','quickdraw','painting','sketch','infograph','clipart']
        self.domain = self.task_name[domain_id]
        self.image_list_path = os.path.join(self.image_list_root, self.domain + "_" + self.mode + ".txt")
        self.path = self.get_path()
        self.length = len(self.path)
        

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.path[index]
        img_path = os.path.join(self.data_root, img_path)
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.mode == 'train':
            return img, target, 1
        elif self.mode == 'test':
            return img, target
    
    def get_path(self):
        images = []
        image_list = open(self.image_list_path).readlines()
        images += [(val.split()[0], int(val.split()[1])) for val in image_list]
        if self.mode == 'train':
            random.shuffle(images)  
        return images


class DomainNetAll(Dataset):
    def __init__(self, mode='train', transform=None) -> None:
        super().__init__()
        self.transform = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor()]
            )
        self.mode = mode
        self.data_root = "your_data_path/domainnet"
        self.image_list_root = "./datasets/image_list/domainnet_list"
        self.task_name = ['real','quickdraw','painting','sketch','infograph','clipart']
        self.path = self.get_path()
        self.length = len(self.path)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:

        img_path, target, domain = self.path[index]
        img_path = os.path.join(self.data_root, domain, img_path)
        img = Image.open(img_path)
        # original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.mode == 'train':
            return img, target, 1
        elif self.mode == 'test':
            return img, target
    
    def get_path(self):
        images = []
        for domain in self.task_name:
            image_list_path = os.path.join(self.image_list_root, domain + "_" + self.mode + ".txt")
            image_list = open(image_list_path).readlines()
            images += [(val.split()[0], int(val.split()[1]), domain) for val in image_list]
        if self.mode == 'train':
            random.shuffle(images)  
        return images


class SequentialDomainNet(ContinualDataset):

    NAME = 'domain-net'
    SETTING = 'domain-il'
    N_DOMAINS_PER_TASK = 1
    N_TASKS = 6
    N_CLASSES = 345
    
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])
    TRANSFORM_DOMAIN_NET = transforms.Compose(
        [transforms.Resize([224, 224]),
        # [transforms.Resize([244, 244]),
        # [transforms.Resize([64, 64]),
        transforms.ToTensor()]
    )
    
    def get_trans_train(args):
        return transforms.Compose(
            [transforms.Resize(args.resize_train),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()]
    )
        
    def get_trans_test(args):
        return transforms.Compose(
            [transforms.Resize(args.resize_test),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
    )

    def get_finetune_all_dataloader(self, mode="train"):
        dataset = DomainNetAll(mode=mode)
        if mode == "train":
            shu = True
        else:
            shu = False
        return DataLoader(dataset,
                        batch_size=self.args.batch_size, 
                        shuffle=shu, 
                        num_workers=4, 
                        pin_memory=True)

    def get_test_dataloaders(self, task_id):
        test_loader_all = []
        for i in range(task_id + 1):
            test_dataset = DomainNet(domain_id=i, mode='test', 
                                        transform=SequentialDomainNet.get_trans_test(self.args))
            test_loader = DataLoader(test_dataset,
                                    batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
                                    # batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_all.append(test_loader)
        return test_loader_all

    def get_dataset(self, task_id, mode='train'):
        return DomainNet(domain_id=task_id, mode=mode, 
                                    transform=SequentialDomainNet.TRANSFORM_DOMAIN_NET)

    def get_data_loaders(self, task_id):
        transform = self.TRANSFORM

        train_dataset = DomainNet(domain_id=task_id, mode='train', 
                                    transform=SequentialDomainNet.get_trans_train(self.args))
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialDomainNet.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        model = create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=False,
            num_classes=21843,
            drop_block_rate=None,
        )
        weight_path = '../~/models/vit_base_p16_224_in22k.pth'
        model.load_state_dict(torch.load(weight_path, weights_only=True), False)
        model.head = torch.nn.Linear(768,SequentialDomainNet.N_CLASSES)
        return model

    @staticmethod
    def get_loss():
        return nn.CrossEntropyLoss()

    @staticmethod
    #! https://github.com/aimagelab/mammoth/blob/master/datasets/seq_cifar100.py
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    #! https://github.com/aimagelab/mammoth/blob/master/datasets/seq_cifar100.py
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform


if __name__ == '__main__':
    dataset = DomainNet()
    dataset[0]

