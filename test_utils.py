#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest

from utils import AverageMeter


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


if __name__ == '__main__':
    unittest.main()