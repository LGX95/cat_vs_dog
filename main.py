#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pytorch 迁移学习分类猫和狗
"""

__author__ = 'LGX95'

import os

import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_data(data_dir):
    """加载数据

    通过 ImageFolder 加载数据，并返回一个 DataLoader

    Args:
        data_dir: 存放数据的文件夹路径，文件夹下应该有 `train` 和 `val` 文件夹，里面以文件夹分类图片
                    eg：datasets/cat_vs_dog/

    Returns:
        以 tuple 的形式返回训练数据和测试数据的 DataLoader
        eg: (train_dataloader, val_dataloader)
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 定义 normalize 数据的参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 加载数据集
    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    # 加载 DataLoader
    train_loader = Data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = Data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, 
    )

    return train_loader, val_loader
