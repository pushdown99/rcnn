import os
import torch as t
from utils.config import opt
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

#trainer.load('pretrained/chainer_best_model_converted_to_pytorch_0.7053.pth')
trainer.load('checkpoints/nia_rcnn_01051419_0.6983099462560374')
#trainer.load('pretrained/chainer_best_model_converted_to_pytorch_0.7053.pth')

opt.caffe_pretrain=False # this model was trained from caffe-pretrained model

img = read_image('dataset/NIA/images/IMG_0651112_apple(apple).jpg')
img = t.from_numpy(img)[None]

_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
vis_bbox(at.tonumpy(img[0]),
       at.tonumpy(_bboxes[0]),
       at.tonumpy(_labels[0]).reshape(-1),
       at.tonumpy(_scores[0]).reshape(-1))
