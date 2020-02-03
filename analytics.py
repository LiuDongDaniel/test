from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import os
from glob import glob
import pandas as pd
import cv2

tqdm = lambda x: x


"""
using for get the information(coord, diameterX, diameterY, diameterZ, label) about the test image
"""

# get the samples information(diameter, label)
def get_samples_information(samples_csv_path, dataset_info_csv_path, data_path, samples_info_csv_path):
    samples_path_list = pd.read_csv(samples_csv_path)
    samples_name_list = []
    for i in range(len(samples_path_list)):
        sample_path = samples_path_list.ix[i][1]
        sample_name = sample_path[len(data_path): -4]
        samples_name_list.append(sample_name)

    df_node = pd.read_csv(dataset_info_csv_path)

    out = open(samples_info_csv_path, 'w')
    out.writelines('name' + ',' + 'diameterX' + ',' + 'diameterY' + ',' + 'diameterZ' + ',' + 'fps' + '\n')
    for index, row in df_node.iterrows():
        name = row['seriesuid'] + '_' + str(int(row['coordX'])) + '_' + str(int(row['coordY']))  + '_' + str(int(row['coordZ']))
        if name in samples_name_list:
            out.writelines(name + ',' + str(int(row['diameterX'])) + ',' + str(int(row['diameterY']))+ ',' + str(int(row['diameterZ']))+ ',' + str(row['fps']) + '\n' )


samples_csv_path = '/home/huiying/luna/11.13/mhi_full/z/up/1_test.csv'
dataset_info_csv_path = '/home/huiying/luna/fps.csv'
data_path = '/home/huiying/luna/11.13/mhi_full/z/up/1/'
samples_info_csv_path = '/home/huiying/luna/11.13/analytics/1_test_info.csv'
get_samples_information(samples_csv_path, dataset_info_csv_path, data_path, samples_info_csv_path)
