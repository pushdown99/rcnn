from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config_hdn import opt
from model.msdn import Hierarchical_Descriptive_Model

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

if __name__ == '__main__':
    import fire

    fire.Fire()
