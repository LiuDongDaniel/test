import os
import pandas as pd
import numpy as np

"""
using for randomly select item form csv file, we should input the the original csv file, the save path and the number of items we want to save 
"""

def get_random_list(number):
    idx = list(range(number))
    np.random.shuffle(idx)
    return idx

def split_samples_to_three_csv(samples_csv_path, train_path, val_path, test_path, precent_for_train, precent_for_val):
    precent_for_val = precent_for_train + precent_for_val

    build_train = open(train_path, 'w')
    build_train.writelines("index" + "," + "filename" + "\n")
    build_val = open(val_path, 'w')
    build_val.writelines("index" + "," + "filename" + "\n")
    build_test = open(test_path, 'w')
    build_test.writelines("index" + "," + "filename" + "\n")

    samples_list = pd.read_csv(samples_csv_path)
    random_list = get_random_list(len(samples_list))
    save_num_list_train = random_list[: int(precent_for_train * len(random_list))]
    save_num_list_val = random_list[int(precent_for_train * len(random_list)): int(precent_for_val * len(random_list))]
    save_num_list_test = random_list[int(precent_for_val * len(random_list)) :]

    for i, item in enumerate(save_num_list_train):
        sample = samples_list.iloc[item].tolist()
        build_train.writelines(str(sample[0]) + "," + sample[1] + "\n")

    for i, item in enumerate(save_num_list_val):
        sample = samples_list.iloc[item].tolist()
        build_val.writelines(str(sample[0]) + "," + sample[1] + "\n")

    for i, item in enumerate(save_num_list_test):
        sample = samples_list.iloc[item].tolist()
        build_test.writelines(str(sample[0]) + "," + sample[1] + "\n")



def random_select_samples(samples_csv_path, save_path, save_number):
    build = open(save_path, 'w')
    build.writelines("index" + "," + "filename" + "\n")

    samples_list = pd.read_csv(samples_csv_path)
    random_list = get_random_list(len(samples_list))
    if save_number <= len(random_list):
        save_num_list = random_list[:save_number]
    else:
        save_num_list = random_list
    for i, item in enumerate(save_num_list):
        sample = samples_list.iloc[item].tolist()
        build.writelines(str(sample[0]) + "," + sample[1] + "\n")




def random_select_files(samples_csv_path, save_path_1,save_path_2, save_precent_for_1):
    build_1 = open(save_path_1, 'w')
    build_1.writelines("filename" + "\n")

    build_2 = open(save_path_2, 'w')
    build_2.writelines("filename" + "\n")


    samples_list = pd.read_csv(samples_csv_path)
    random_list = get_random_list(len(samples_list))
    save_list_1 = random_list[: int(save_precent_for_1 * len(random_list))]
    save_list_2 = random_list[int(save_precent_for_1 * len(random_list)) : ]

    for i, item in enumerate(save_list_1):
        sample = samples_list.iloc[item].tolist()
        build_1.writelines(str(sample[0]) + "\n")

    for i, item in enumerate(save_list_2):
        sample = samples_list.iloc[item].tolist()
        build_2.writelines(str(sample[0]) + "\n")


samples_csv_path = '/mnt/sdb1/Datas/npz/301_series.csv'
save_pat_1 = '/mnt/sdb1/Datas/npz/301_train.csv'
save_path_2 = '/mnt/sdb1/Datas/npz/301_val.csv'
save_precent_for_1 = 0.85
random_select_files(samples_csv_path, save_pat_1,save_path_2, save_precent_for_1)


def random_select_candidate(samples_csv_path, save_path, save_number):
    build = open(save_path, 'w')
    build.writelines("seriesuid" + "," + "coordX" + "," + "coordY" + "," + "coordZ" + "," + "class" + "\n")

    samples_list = pd.read_csv(samples_csv_path)
    samples_list_negative = samples_list[samples_list['class'] == 0]
    random_list = get_random_list(len(samples_list_negative))
    if save_number <= len(random_list):
        save_num_list = random_list[:save_number]
    else:
        save_num_list = random_list
    for i, item in enumerate(save_num_list):
        sample = samples_list_negative.iloc[item].tolist()
        build.writelines(str(sample[0]) + "," + str(sample[1]) + "," + str(sample[2]) + "," + str(sample[3]) + "," + str(sample[4]) + "\n")


def cal_files_num(path):
    files_list = os.listdir(path)
    num = len(files_list)
    return num



# samples_csv_path =  "/home/huiying/luna/11.25/z/0_s.csv"
# save_path =  "/home/huiying/luna/11.25/z/0_5075.csv"
# save_number = 5075
# random_select_files(samples_csv_path, save_path, save_number)
# #
# samples_csv_path = '/home/huiying/luna/11.19/mhi/z/npy/1.csv'
# save_path = '/home/huiying/luna/11.19/mhi/z/npy/1_s.csv'
# random_select_samples(samples_csv_path, save_path, 2371)
# samples_csv_path = '/home/huiying/luna/11.20/dataset/mhi/0_test.csv'
# save_path = '/home/huiying/luna/11.20/dataset/mhi/0_test_96.csv'
#
# random_select_samples(samples_csv_path, save_path, 96)
#
# # #
# samples_csv_path = '/home/huiying/luna/11.20/dataset/mhi/0.csv'
# train_path = '/home/huiying/luna/11.20/dataset/mhi/0_train.csv'
# val_path = '/home/huiying/luna/11.20/dataset/mhi/0_val.csv'
# test_path = '/home/huiying/luna/11.20/dataset/mhi/0_test.csv'
# split_samples_to_three_csv(samples_csv_path, train_path, val_path, test_path, 0.83, 0.085)
# #
# samples_csv_path = '/home/huiying/luna/11.19/mhi/x/up/1_s.csv'
# train_path = '/home/huiying/luna/11.19/mhi/x/up/1_train.csv'
# val_path = '/home/huiying/luna/11.19/mhi/x/up/1_val.csv'
# test_path = '/home/huiying/luna/11.19/mhi/x/up/1_test.csv'
# split_samples_to_three_csv(samples_csv_path, train_path, val_path, test_path, 0.8, 0.1)
