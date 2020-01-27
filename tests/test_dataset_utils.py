import unittest
from dataset_utils import *
import torch, torchvision

class TestTargetTransformDataset(unittest.TestCase):

    def setUp(self):
        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        self.target_mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3, 8:4, 9:0}
        self.tt_set = TargetTransformDataset(self.train_set, \
                                             target_mapping=self.target_mapping)

    def tearDown(self):
        pass

    def test_getitem(self):
        for i in range(0,1000,5):
            self.assertEqual(self.target_mapping[self.train_set[i][1]], self.tt_set[i][1])

    def test_len(self):
        self.assertEqual(len(self.train_set), len(self.tt_set))

class TestStatefulDataSampler(unittest.TestCase):

    def setUp(self):
        self.train_set = torchvision.datasets.CIFAR10(root='./data',\
                                                      train=True, download=True)

    def test_init(self):
        sampler = StatefulDataSampler(self.train_set, random_seed=19)
        self.assertEqual(sampler.state[3]['data'][0], 12973)

    def test_len(self):
        sampler = StatefulDataSampler(self.train_set, random_seed=19)
        self.assertEqual(len(sampler), 0)
        _ = sampler.get_next_samples(100)
        self.assertEqual(len(sampler), 100)
        _ = sampler.get_next_samples(1000)
        self.assertEqual(len(sampler), 1100)

if __name__=='__main__':
    unittest.main()
