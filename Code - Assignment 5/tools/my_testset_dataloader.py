import os
import os.path
import numpy as np
import sys
import torch

from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torch.utils.data import Dataset as VisionDataset
from tools.utils import check_integrity, download_and_extract_archive

import shutil

class TESTSET():
    def __init__(self, root, transform=None, target_transform=None):
        self.transform = transform
        self.root = root
        self.data = []
        img_name = os.path.join(root, "cifar10_test/cifar10-batches-images-test.npy")
        self.data = np.load(img_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: image
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)