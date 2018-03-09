#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pytorch 迁移学习分类猫和狗
"""

__author__ = 'LGX95'

import os

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils import AverageMeter, accuracy


def main():
    train_loader, val_loader = load_data('./datasets/cat_vs_dog/')

    print("=> creating model ")
    model = models.resnet152(pretrained=True)
    # 冻结模型的参数
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    # 替换全连接层，新构造的模块的 requires_grad 为 True
    model.fc = nn.Linear(in_features, 2)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.fc.parameters(), 1e-3)

    for epoch in range(1):
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)


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
        val_dataset, batch_size=32, shuffle=False
    )

    return train_loader, val_loader


def train(train_loader, model, criterion, optimizer, epoch):
    """训练

    Args:
        train_loader: 训练数据集的 DataLoader
        model: CNN 模型
        criterion: 误差函数
        optimizer: 优化器
        epoch: 当前训练的轮数
    """
    # model switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (image, target) in enumerate(train_loader):
        # target = target.cuda(async=True)
        image_var, target_var = Variable(image), Variable(target)

        # 前馈运算和计算损失
        output = model(image_var)
        loss = criterion(output, target_var)

        # 记录准确率和损失
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], image.size(0))
        top1.update(prec1[0], image.size(0))

        # 反向传播和更新权值
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: [{0}][{1}/{2}]'
              ' | Loss: {loss.val:.4f} ({loss.avg:.4f})'
              ' | top1: {top1.val:.3f} ({top1.avg:.3f})'.format(
                  epoch, i, len(train_loader), loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """验证

    Args:
        val_loader: 验证数据集的 DataLoader
        model: CNN 模型
        criterion: 误差函数

    Return:
        测试准确率的平均值
    """
    # model switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (image, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        image_var = Variable(image, volatile=True)
        target_var = Variable(target, volatile=True)

        # 前馈运算和计算损失
        output = model(image_var)
        loss = criterion(output, target_var)

        # 记录准确率和损失
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], image.size(0))
        top1.update(prec1[0], image.size(0))

        print('Validata: [{0}/{1}]'
              ' | Loss: {loss.val:.4f} ({loss.avg:.4f})'
              ' | Accuracy: {top1.val:.3f} ({top1.avg:.3f})'.format(
                  i, len(val_loader), loss=losses, top1=top1))

    return top1.avg


if __name__ == '__main__':
    main()