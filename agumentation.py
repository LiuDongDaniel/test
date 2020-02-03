
import random
import cv2
from PIL import Image
import pandas as pd
import math



def crop(img):
    high, width = img.shape
    coord_x1 = random.choice(range(math.ceil((width - width * 0.9) / 2)))
    coord_y1 = random.choice(range(math.ceil((high - high * 0.9) / 2)))
    coord_x2 = coord_x1 + int(width * 0.9)
    coord_y2 = coord_y1 + int(high * 0.9)
    img_crop = img[coord_y1: coord_y2, coord_x1: coord_x2]
    return img_crop




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


def agumentation(samples_csv_path,samples_save_path,augment_number,samples_path):

    samples_list = pd.read_csv(samples_csv_path)
    file_name_list = samples_list['filename']
    for i in range(len(file_name_list)):
        sample_path = file_name_list[i]
        img = cv2.imread(sample_path)[:, :, 1]

        for j in range(augment_number):
            img_name = sample_path[len(samples_path): -4]
            img_name = img_name + '_' + str(j)
            if img.shape[0] >= 12 and img.shape[1] >= 12:
                if random.random() <= 0.35:
                    img_name = img_name + '_crop'
                    img = crop(img)
            if random.random() <= 0.35:
                img = flip(img)
                img_name = img_name + '_flip'
            if random.random() <= 0.35:
                img = rotate(img)
                img_name = img_name + '_rotate'

            cv2.imwrite(samples_save_path + img_name + '.jpg', img)


samples_csv_path = '/home/huiying/luna/11.20/dataset/mhi/1_train.csv'
samples_save_path = '/home/huiying/luna/11.20/dataset/mhi/1_aug/'
augment_number = 5
samples_path = '/home/huiying/luna/11.20/dataset/mhi/1/'
agumentation(samples_csv_path, samples_save_path,augment_number,samples_path)
