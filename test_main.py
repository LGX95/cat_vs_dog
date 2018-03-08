#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest

from main import load_data


class TestMain(unittest.TestCase):
    """对 main.py 的测试
    """

    def testLoadData(self):
        """测试 load_data
        """
        loaders = load_data('./datasets/cat_vs_dog')

        self.assertEqual(len(loaders), 2)
        self.assertNotEqual(len(loaders[0]), 0)
        self.assertNotEqual(len(loaders[1]), 0)


if __name__ == '__main__':
    unittest.main()