from __future__ import division
from glob import glob
import os

"""
using for generate the csv file for the jpg image, we should input the path of the .jpg image, the save csv path, and label of each image
"""


# save .jpg file path to csv
def save_jpg2csv(path, name, labelnum=1):
    build = open(name, 'w')
    file_list = glob(path + "*.jpg")
    build.writelines("index" + "," + "filename" + "\n")
    for index in range(len(file_list)):
        build.writelines(str(labelnum) + "," + file_list[index] + "\n")

# save .jpg file path to csv
def save_jpg2csv_no_label(path, name):
    build = open(name, 'w')
    file_list = glob(path + "*.jpg")
    build.writelines("filename" + "\n")
    for index in range(len(file_list)):
        build.writelines(file_list[index] + "\n")

# save_jpg2csv("/home/huiying/luna/11.20/dataset/mhi/0/", "/home/huiying/luna/11.20/dataset/mhi/0.csv", 0)

# save .jpg file path to csv
def save_file2csv_forsingle(dir, name):

    build = open(name, 'w')
    build.writelines("filename" + "\n")

    for root, dirs, files in os.walk(dir):
        if len(dirs):
            print("sub_dirs:", dirs)
            sub_dirs = dirs

    for index in range(len(sub_dirs)):
        build.writelines(dir  + sub_dirs[index] + "\n")

save_file2csv_forsingle("/home/huiying/luna/11.25/z/0/", "/home/huiying/luna/11.25/z/0_s.csv")

# def save_npy2csv_no_label(path, name):
#     build = open(name, 'w')
#     file_list = glob(path + "*.npy")
#     build.writelines("filename" + "\n")
#     for index in range(len(file_list)):
#         build.writelines(file_list[index] + "\n")


# save .npy file path to csv
# def save_npy2csv(path, name, labelnum=1):
#     build = open(name, 'w')
#     file_list = glob(path + "*.npy")
#     build.writelines("index" + "," + "filename" + "\n")
#     for index in range(len(file_list)):
#         build.writelines(str(labelnum) + "," + file_list[index] + "\n")


#
# save_npy2csv("/home/huiying/luna/11.19/mhi/z/npy/0/", "/home/huiying/luna/11.19/mhi/z/npy/0.csv", 0)
# save_npy2csv("/home/huiying/luna/11.19/mhi/z/npy/1/", "/home/huiying/luna/11.19/mhi/z/npy/1.csv", 1)

# save_jpg2csv("/home/huiying/luna/11.19/mhi/x/down/0/", "/home/huiying/luna/11.19/mhi/x/down/0.csv", 0)
# save_jpg2csv("/home/huiying/luna/11.19/mhi/x/down/1/", "/home/huiying/luna/11.19/mhi/x/down/1.csv", 1)
#
# save_jpg2csv("/home/huiying/luna/11.19/mhi/x/center_to_up/0/", "/home/huiying/luna/11.19/mhi/x/center_to_up/0.csv", 0)
# save_jpg2csv("/home/huiying/luna/11.19/mhi/x/center_to_up/1/", "/home/huiying/luna/11.19/mhi/x/center_to_up/1.csv", 1)
#
# save_jpg2csv("/home/huiying/luna/11.19/mhi/x/center_to_down/0/", "/home/huiying/luna/11.19/mhi/x/center_to_down/0.csv", 0)
# save_jpg2csv("/home/huiying/luna/11.19/mhi/x/center_to_down/1/", "/home/huiying/luna/11.19/mhi/x/center_to_down/1.csv", 1)

# save_npy2csv_no_label("/Users/liudong/Desktop/project/LUNA16/challenge/classsification/0_aug/","/Users/liudong/Desktop/project/LUNA16/challenge/classsification/aug.csv")

