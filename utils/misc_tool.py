import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_image_box (path, bbox):
  im = Image.open(path)
  fig, ax = plt.subplots()
  ax.imshow(im)
  rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
  ax.add_patch(rect)
  plt.show()

def load_image(name, path):
  img_path = path + name + '.jpg'
  img = cv2.imread(img_path)
  return img

def plot_image(img):
  plt.imshow(img)
  plt.title(img.shape)
    
def plot_grid(img_names, img_root, rows=5, cols=5):
  fig = plt.figure(figsize=(25,25))
    
  for i,name in enumerate(img_names):
    fig.add_subplot(rows,cols,i+1)
    img = load_image(name, img_root)
    plot_image(img)
        
  plt.show()

