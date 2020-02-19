from utils.helper import StrLabelConverter
from utils.keys import plate_keys

import cv2
import numpy as np

import torch

class CRNNCollate(object):
    def __init__(self, h=32, w=100):
        self.h = h
        self.w = w
        self.converter = StrLabelConverter(plate_keys)

    def __call__(self, batch):
        images, labels = zip(*batch)

        assert len(images) == len(labels)
        inputs = np.zeros((len(images), 1, self.h, self.w), dtype=np.float32)
        for i, image in enumerate(images):
            image = cv2.resize(image, (self.w, self.h))
            center=(self.w//2,self.h//2)
            import random
            M = cv2.getRotationMatrix2D(center, random.randint(-15,15), 1.0)  #0,15,10,8
            image = cv2.warpAffine(image, M, (self.w, self.h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.astype(np.float32)
            image -= 127.5
            image /= 127.5
            inputs[i, 0, ...] = image.astype(np.float32)
        labels = [label.replace('#', '') for label in labels]
        targets, lengths = self.converter.encode(labels)

        inputs = torch.FloatTensor(inputs)
        targets = torch.IntTensor(targets)
        lengths = torch.IntTensor(lengths)

        return inputs, labels, targets, lengths
