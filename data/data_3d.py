import torch.utils.data
import numpy as np
import random
from scipy.special import comb
import copy
import csv
import os
from scipy.ndimage import zoom
import cv2
import pandas as pd
from scipy.ndimage import zoom

"""
a.using for generate the image path list and produce the image numpy to tensor
b. before generating the tensor, we can do the data augment and transformate
c. can save the training image for check whether the input image haves problem

note: the return statement is different in test process and train process, so we should change it in different process.
"""


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_csv_list):
        super(Dataset, self).__init__()
        self.path_list = []
        for i in range(len(path_csv_list)):
            csv_file = path_csv_list[i]

            path_csv = pd.read_csv(csv_file)
            for index, row in path_csv.iterrows():
                self.path_list.append((row['filename'], row['index']))


    def __getitem__(self, index):
        path, label = self.path_list[index]
        img = np.load(path)
        # transform img
        if random.random() <= 0.4:
            img = data_flip(img)

        N, H, W = img.shape
        img = zoom(img, (1, 48 / H, 48 / W), order=1)

        img[img < 0] = 0
        img[img > 255] = 255
        img_nor = img / 255
        img_nor = img_nor[np.newaxis, ...]
        img_nor = np.array(img_nor, dtype=np.float32)
        label = np.array(label, dtype=np.long)

        # for train
        # return torch.from_numpy(img_nor), torch.from_numpy(label)

        # save the training img for check
        # save_path = '/home/huiying/luna/11.19/mhi/check/'
        # if label == 1:
        #     save_path_sub = save_path + '1/' + img_name + '.jpg'
        #     cv2.imwrite(save_path_sub, img )
        # if label == 0:
        #     save_path_sub = save_path + '0/' + img_name + '.jpg'
        #     cv2.imwrite(save_path_sub, img)
        #
        # # for test
        test_img_path = '/home/huiying/luna/11.19/mhi/np/0/'
        img_name = path[len(test_img_path): -4]
        return torch.from_numpy(img_nor), torch.from_numpy(label), img_name, img


    def __len__(self):
        return len(self.path_list)






def rotate(img):
    (h, w) = img.shape[:2]
    center = (w//2, h//2)
    angle = random.choice([45, 90, 135, 225, 315])
    rota = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, rota, (w, h))
    return img

def flip(img):
    direct = random.choice([-1, 0, 1])
    img = cv2.flip(img,direct)
    return img


def pad(img_zoom, shape_train):
    shape_current = np.array(img_zoom.shape, dtype=np.int32)
    shape_train = np.array(shape_train, dtype=np.int32)
    num_pad = shape_train - shape_current
    assert (num_pad >= 0).all(), 'num_pad < 0'
    img_zoom = np.pad(img_zoom, ((0, num_pad[0]), (0, num_pad[1]), (0, num_pad[2])), mode='constant', constant_values=0.)
    return img_zoom


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_flip(x, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        cnt = cnt - 1

    return x


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    # _, img_rows, img_cols, img_deps = x.shape
    img_deps, img_cols, img_rows = x.shape
    num_block = 500
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        block_noise_size_z = random.randint(1, img_deps // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        noise_z = random.randint(0, img_deps - block_noise_size_z)
        window = orig_image[noise_z:noise_z + block_noise_size_z, noise_y:noise_y + block_noise_size_y, noise_x:noise_x + block_noise_size_x]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_z, block_noise_size_y, block_noise_size_x))
        image_temp[noise_z:noise_z + block_noise_size_z, noise_y:noise_y + block_noise_size_y, noise_x:noise_x + block_noise_size_x] = window

    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x):
    # _, img_rows, img_cols, img_deps = x.shape
    img_deps, img_cols, img_rows = x.shape
    # block_noise_size_x = random.randint(10, 20)
    # block_noise_size_y = random.randint(10, 20)
    # block_noise_size_z = random.randint(10, 20)
    block_noise_size_x = random.randint(40, 80)
    block_noise_size_y = random.randint(40, 80)
    block_noise_size_z = random.randint(20, 40)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
    x[noise_z:noise_z + block_noise_size_z, noise_y:noise_y + block_noise_size_y, noise_x:noise_x + block_noise_size_x] = random.random()
    return x


def image_out_painting(x):
    # _, img_rows, img_cols, img_deps = x.shape
    img_deps, img_cols, img_rows = x.shape
    # block_noise_size_x = img_rows - random.randint(10, 20)
    # block_noise_size_y = img_cols - random.randint(10, 20)
    # block_noise_size_z = img_deps - random.randint(10, 20)
    block_noise_size_x = img_rows - random.randint(40, 80)
    block_noise_size_y = img_cols - random.randint(40, 80)
    block_noise_size_z = img_deps - random.randint(20, 40)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2]) * 1.0
    x[noise_z:noise_z + block_noise_size_z, noise_y:noise_y + block_noise_size_y, noise_x:noise_x + block_noise_size_x] = image_temp[noise_z:noise_z + block_noise_size_z, noise_y:noise_y + block_noise_size_y, noise_x:noise_x + block_noise_size_x]
    return x