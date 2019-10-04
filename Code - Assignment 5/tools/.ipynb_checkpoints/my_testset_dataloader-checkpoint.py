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
            tuple: (image, target) where target is index of the target class.
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


    def download(self):
        try:
            download_and_extract_archive(self.url, self.root, filename=self.filename)
        except Exception as e:
            print("Interrupted during dataset downloading. "
                  "Cleaning up...")
            # Clean up
            cwd = os.getcwd()
            rm_path = os.path.join(cwd, self.root, "cifar10_test")
            shutil.rmtree(rm_path)
            raise e

        print('Files already downloaded and verified')
