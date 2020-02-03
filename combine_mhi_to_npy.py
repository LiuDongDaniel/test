import os
import cv2
import numpy as np


def combine_mhi_to_npy(up_mhi_path, down_mhi_path, to_up_mhi_path, to_down_mhi_path, save_path):
    file_list = os.listdir(up_mhi_path)
    for i in range(len(file_list)):
        item = file_list[i]
        up_mhi = cv2.imread(up_mhi_path + item)[:, :, 1]
        down_mhi = cv2.imread(down_mhi_path + item)[:, :, 1]
        to_up_mhi = cv2.imread(to_up_mhi_path + item)[:, :, 1]
        to_down_mhi = cv2.imread(to_down_mhi_path + item)[:, :, 1]

        h, w = up_mhi.shape
        block = np.zeros(shape=(4, h, w), dtype=np.float)

        block[0] = up_mhi
        block[1] = down_mhi
        block[2] = to_up_mhi
        block[3] = to_down_mhi

        np.save(save_path + item[:-4] + '.npy', block)

up_mhi_path = '/home/huiying/luna/11.19/mhi/z/up/1/'
down_mhi_path = '/home/huiying/luna/11.19/mhi/z/down/1/'
to_up_mhi_path = '/home/huiying/luna/11.19/mhi/z/center_to_up/1/'
to_down_mhi_path = '/home/huiying/luna/11.19/mhi/z/center_to_down/1/'
save_path = '/home/huiying/luna/11.19/mhi/z/npy/1/'

combine_mhi_to_npy(up_mhi_path, down_mhi_path, to_up_mhi_path, to_down_mhi_path, save_path)