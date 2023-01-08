import os
import xml.etree.ElementTree as ET

import numpy as np
import json
import time
import codecs

from os.path import join
from .util import read_image
from utils.config import opt


class COCOBboxDataset:
    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False,):
        id_list_file = join(data_dir, '{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = COCO_BBOX_LABEL_NAMES

        self.instances = json.load(codecs.open(join(data_dir, 'instances.json'), 'r', 'utf-8-sig'))


    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_  = str(self.ids[i])
        anno = self.instances[id_]

        bbox      = list()
        label     = list()
        difficult = list()

        for obj in anno['bbox']:
            difficult.append(0)
            bbox.append(obj['bbox'])
            #print (i, id_, obj['name'], COCO_BBOX_LABEL_NAMES.index(obj['name']), bbox)
            label.append(COCO_BBOX_LABEL_NAMES.index(obj['name']))

        if  len(bbox)<1:
          print ('========================')
          print (i, id_, bbox)
          print ('========================')
        bbox      = np.stack(bbox).astype(np.float32)
        label     = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = join(self.data_dir, 'train2017', '{:012d}.jpg'.format(int(id_)))
        if not os.path.isfile(img_file): img_file = join(self.data_dir, 'val2017',  '{:012d}.jpg'.format(int(id_)))
        if not os.path.isfile(img_file): img_file = join(self.data_dir, 'test2017', '{:012d}.jpg'.format(int(id_)))

        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example

if opt.data == 'coco':
  COCO_BBOX_LABEL_NAMES = tuple(json.load(codecs.open(join(opt.data_dir, 'object_id.json'), 'r', 'utf-8-sig')).keys())
else:
  COCO_BBOX_LABEL_NAMES = tuple()
