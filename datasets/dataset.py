import cv2
import json
import os
import numpy as np
import pdb
from torch.utils.data import Dataset
import torch
import time


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0: # poly = []
        return polys, tags
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print(poly)
            # print('invalid poly')
            continue
        if p_area > 0:
            # 坐标原点是左上角, 故面积算出来应该是负值
            # print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :] # 交换对角
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)

def load_annoataion(label):
    '''
    load annotation from json
    Args:
        label: label dict
        "label": [
            {
                "name": "outline",
                "type": "detection",
                "version": "1",
                "data": [
                    {
                    "bbox": [
                        [
                          55.285714,
                          99.642857
                        ],
                        [
                          127.285714,
                          99.642857
                        ],
                        [
                          120.857143,
                          126
                        ],
                        [
                          50.785714,
                          125.357143
                        ]
                    ],
                  "class": "MJ017"
                }
                ]
            },
            ...
        ]
    Return:
        text_polys: np.array, shape=(N, 4, 2)
            if text file if empty, shape=(0,)
        text_tags: np.array, shape=(N,)
    '''
    text_polys = []
    text_tags = []
    plate=[]
    for line in label:
        if line['type'] != 'detection':
            continue
        for anno in line['data']:
            text = anno['class']
            bbox = anno['bbox']
            text_polys.append(bbox)
            plate.append(text)     
            if text == '*' or text == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)

    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool),plate

def rois(img,boxes):
    x1=int(boxes[0][0][0])
    y1=int(boxes[0][0][1])
    x2=int(boxes[0][2][0])
    y2=int(boxes[0][2][1])
    return img[y1:y2,x1:x2]

class Plate(Dataset):

    def __init__(self,root=None):
        files=open(root,'r')
        self.labels=files.readlines()
        files.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if index>=len(self):
            index=index%len(self)
        item=json.loads(self.labels[index])
        url=item['url']
        if url.startswith('qiniu:///supredata-internal-ocr/'):
            url = url.replace('qiniu:///supredata-internal-ocr/', '/data/', 1)
        if url.startswith('qiniu:///personal/hutao/'):
            url = url.replace('qiniu:///personal/hutao/', '/data/plate', 1)
        im = cv2.imread(url)
        if im is None:
            return self[np.random.randint(0,len(self)-1)]
        try:
            text_polys, text_tags,plate = load_annoataion(item['label'])
            img=rois(im,text_polys)
            h,w,_=img.shape
            assert h!=0 and w!=0
        except:
            return self[np.random.randint(0,len(self)-1)]
        
        return img,plate[0]
       

if  __name__=='__main__':
    plate=Plate('/root/my/train.json')
    dataloader=torch.utils.data.DataLoader(plate,batch_size=1,shuffle=True)
    tic=time.time()
    for img,plate in dataloader:
        print(img.shape)
        print(plate)
