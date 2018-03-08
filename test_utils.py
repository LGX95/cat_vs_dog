#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest

import torch

from utils import AverageMeter
from utils import accuracy


class TestUtils(unittest.TestCase):
    """对 utils 的测试"""

    def testAverageMeter(self):
        """test class AverageMeter"""

        meter = AverageMeter()
        for i in range(1, 10):
            meter.update(i)
        self.assertEqual(meter.avg, 5)
        self.assertEqual(meter.count, 9)
        self.assertEqual(meter.val, 9)

    def testAccuracy(self):
        """test function accuracy"""

        target = torch.ones(32)
        target = target.type(torch.LongTensor)
        test1 = torch.stack([torch.FloatTensor([0.7, 0.3])] * 32, dim=0)
        prec1 = accuracy(test1, target)[0]
        self.assertEqual(prec1[0], 0)

        test2 = torch.stack([torch.FloatTensor([0.2, 0.8])] * 32, dim=0)
        prec1 = accuracy(test2, target)[0]
        self.assertEqual(prec1[0], 100)


if __name__ == '__main__':
    unittest.main()
