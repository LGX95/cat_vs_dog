#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest

from main import TestImageFolder


class TestMain(unittest.TestCase):
    """对 main.py 的测试"""

    # Fixme 引入命令行参数后失效
    # def testLoadData(self):
    #     """测试 load_data"""
    #     loaders = load_data('./datasets/cat_vs_dog')

    #     self.assertEqual(len(loaders), 2)
    #     self.assertNotEqual(len(loaders[0]), 0)
    #     self.assertNotEqual(len(loaders[1]), 0)

    def testTestImageFolder(self):
        """测试 class TestImageFolder"""
        test_dataset = TestImageFolder('./datasets/cat_vs_dog/test')

        self.assertIsInstance(test_dataset[0], tuple)
        self.assertIsInstance(test_dataset[0][1], str)
        self.assertNotEqual(len(test_dataset), 0)


if __name__ == '__main__':
    unittest.main()
