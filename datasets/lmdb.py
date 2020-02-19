import cv2
import json
import numpy as np
import sys

import torch
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    def __init__(self, root=None):
        import lmdb
        self.env = lmdb.open(root,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)

        if not self.env:
            print('cannot load lmdb from {}'.format(root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.meta_info = json.loads(txn.get('meta_info'.encode()).decode('utf-8'))

    def __len__(self):
        return self.meta_info['num_samples']

    def __getitem__(self, index):
        assert index < len(self)

        with self.env.begin(write=False) as txn:
            image_key = '{}/{:08d}'.format(self.meta_info['image_key_prefix'], index)
            label_key = '{}/{:08d}'.format(self.meta_info['label_key_prefix'], index)

            image_buf = txn.get(image_key.encode())
            label = txn.get(label_key.encode()).decode('utf-8')

            try:
                image_buf = np.frombuffer(image_buf, dtype=np.uint8)
                image = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
            except Exception:
                print('Corrupted image for {}'.format(index))
                return self[index + 1]

        return image, label

