
# -*- coding: utf-8 -*-
#
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.model_selection import train_test_split
from PIL import Image
# # Import some data to play with

# img_path = '/home/huiying/luna/11.19/mhi/z/up/0/2.156.112605.14038001579812.20180614022315.3.5960.7.201861732248_407_287_152.jpg'
# img = Image.open(img_path)
# high, width = img.size
# coord_x1 = random.choice(range(round((width - width *0.9)/2)))
# coord_y1 = random.choice(range(round((high - high *0.9)/2)))
# coord_x2 = coord_x1 + int(width *0.9)
# coord_y2 = coord_y1 + int(high *0.9)
# img_crop = img.crop([coord_x1,coord_y1,coord_x2,coord_y2])
# img_arry = np.array(img)
#
# print('tt')
import shutil
import os
def copyDirectory(src, dest):
    if os.path.exists(PF):
        shutil.rmtree(PF)
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)
    print(os.path.exists('.DS_Store')) # delete the generated hidden file

CF = 'remote/data_train'  # Copied FolderName (Old)
PF = 'remote/pre_train'  # Pasted FolderName (New)

copyDirectory(CF, PF)





