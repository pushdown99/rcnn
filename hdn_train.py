from __future__ import  absolute_import
import os, time

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config_hdn import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
from hdn_trainer import HDNTrainer

#from data.dataset import Dataset, TestDataset, inverse_normalize
#from model import FasterRCNNVGG16
#from torch.utils import data as data_
#from trainer import FasterRCNNTrainer
#from utils import array_tool as at
#from utils.vis_tool import visdom_bbox
#from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(**kwargs):
    opt._parse(kwargs)

def train(**kwargs):
  opt._parse(kwargs)

  dataset = Dataset(opt)
  print('load data')
  dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers)
  testset = TestDataset(opt)
  test_dataloader = data_.DataLoader(testset, batch_size=1, num_workers=opt.test_num_workers, shuffle=False, pin_memory=True)

  faster_rcnn = FasterRCNNVGG16()
  print('model construct completed')
  trainer = HDNTrainer(faster_rcnn).cuda()
  if opt.load_path:
    trainer.load(opt.load_path)
    print('load pretrained model from %s' % opt.load_path)
  trainer.vis.text(dataset.db.label_names, win='labels')

  best_map = 0
  lr_ = opt.lr
  for epoch in range(opt.epoch):
    trainer.reset_meters()


if __name__ == '__main__':
  _t = time.process_time()
  import fire
  fire.Fire()
  _elapsed = time.process_time() - _t

  print ('')
  print ('elapsed: {:.2f} sec'.format(_elapsed))
