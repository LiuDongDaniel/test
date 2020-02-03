from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import os
from glob import glob
import pandas as pd
import cv2

"""
using for cut the cube form the 3D CT images(mhd, dicom)
we can change the cube size, the save form(bmp, npy)
"""




tqdm = lambda x: x


def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(filename)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    srcitkimagearray = sitk.GetArrayFromImage(image)  # z, y, x
    origin = image.GetOrigin()  # x, y, z
    spacing = image.GetSpacing()  # x, y, z


    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower

    nor_image = (srcitkimagearray - lower) / (upper - lower)
    nor_image[nor_image <0] = 0
    nor_image[nor_image >1] = 1
    nor_image = (nor_image *255).astype('uint8')

    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(nor_image)
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)

    return sitktructedimage


# Some helper functions

def get_cube_from_img(img3d, center, diameter_x,diameter_y,diameter_z):
    # get roi(z,y,z) image and in order the out of img3d(z,y,x)range
    center_z = center[0]
    center_y = center[1]
    center_x = center[2]

    # double size
    start_x = max(center_x - diameter_x, 0)
    if start_x + diameter_x *2  > img3d.shape[2]:
        start_x = img3d.shape[2] - diameter_x *2
    start_y = max(center_y - diameter_y, 0)
    if start_y + diameter_y *2 > img3d.shape[1]:
        start_y = img3d.shape[1] - diameter_y *2
    start_z = max(center_z - diameter_z/2 , 0)
    if start_z + diameter_z > img3d.shape[0]:
        start_z = img3d.shape[0] - diameter_z
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    roi_img3d = img3d[start_z:start_z + diameter_z , start_y:start_y + diameter_y *2, start_x:start_x + diameter_x*2]


    #original size
    # start_x = max(center_x - diameter_x/2, 0)
    # if start_x + diameter_x  > img3d.shape[2]:
    #     start_x = img3d.shape[2] - diameter_x
    # start_y = max(center_y - diameter_y/2, 0)
    # if start_y + diameter_y > img3d.shape[1]:
    #     start_y = img3d.shape[1] - diameter_y
    # start_z = max(center_z - diameter_z/2 , 0)
    # if start_z + diameter_z > img3d.shape[0]:
    #     start_z = img3d.shape[0] - diameter_z
    # start_z = int(start_z)
    # start_y = int(start_y)
    # start_x = int(start_x)
    # roi_img3d = img3d[start_z:start_z + diameter_z , start_y:start_y + diameter_y, start_x:start_x + diameter_x]

    return roi_img3d


# get 1.5 times
def get_cube_from_img_1_5(img3d, center, diameter_x,diameter_y,diameter_z):
    # get roi(z,y,z) image and in order the out of img3d(z,y,x)range
    center_z = center[0]
    center_y = center[1]
    center_x = center[2]

    # double size
    start_x = max(center_x - (diameter_x * 0.75), 0)
    if start_x + diameter_x *1.5  > img3d.shape[2]:
        start_x = img3d.shape[2] - diameter_x *1.5
    start_y = max(center_y - (diameter_y *0.75), 0)
    if start_y + diameter_y * 1.5 > img3d.shape[1]:
        start_y = img3d.shape[1] - diameter_y *1.5
    start_z = max(center_z - diameter_z/2 , 0)
    if start_z + diameter_z > img3d.shape[0]:
        start_z = img3d.shape[0] - diameter_z
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    roi_img3d = img3d[start_z:start_z + diameter_z , start_y:start_y + int(diameter_y *1.5), start_x:start_x + int(diameter_x*1.5)]


    #original size
    # start_x = max(center_x - diameter_x/2, 0)
    # if start_x + diameter_x  > img3d.shape[2]:
    #     start_x = img3d.shape[2] - diameter_x
    # start_y = max(center_y - diameter_y/2, 0)
    # if start_y + diameter_y > img3d.shape[1]:
    #     start_y = img3d.shape[1] - diameter_y
    # start_z = max(center_z - diameter_z/2 , 0)
    # if start_z + diameter_z > img3d.shape[0]:
    #     start_z = img3d.shape[0] - diameter_z
    # start_z = int(start_z)
    # start_y = int(start_y)
    # start_x = int(start_x)
    # roi_img3d = img3d[start_z:start_z + diameter_z , start_y:start_y + diameter_y, start_x:start_x + diameter_x]

    return roi_img3d



# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


def get_node_classify():
    # Getting list of image files and output nuddle 0 and 1
    for subsetindex in range(1):
        # luna_path = "/Users/liudong/Desktop/project/LUNA16/challenge/src/"
        # luna_subset_path = '/mnt/sdb1/fps_test/'
        luna_subset_path = '/home/huiying/574/'
        output_path = "/home/huiying/luna/11.20/dataset/z/"
        # file_list = glob(luna_subset_path + "*.mhd")
        file_list = os.listdir(path=luna_subset_path)

        # The locations of the nodes
        # luna_csv_path = "/Users/liudong/Desktop/project/LUNA16/challenge"
        df_node = pd.read_csv("/home/huiying/luna/fps.csv")
        df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
        df_node = df_node.dropna()
        # Looping over the image files(enumerate for find the index and value)
        for fcount, img_file in enumerate(tqdm(file_list)):
            # get all nodules associate with file
            mini_df = df_node[df_node["file"] == img_file]
            # some files may not have a nodule--skipping those
            if mini_df.shape[0] > 0:
                # load the data once()
                img_file_path = luna_subset_path + img_file
                itk_img = load_itkfilewithtrucation(img_file_path, 600, -1000)
                img_array = sitk.GetArrayFromImage(itk_img)
                # x,y,z  Origin in world coordinates (mm)
                origin = np.array(itk_img.GetOrigin())
                # spacing of voxels in world coor. (mm)
                spacing = np.array(itk_img.GetSpacing())
                # go through all nodes
                index = 0
                for node_idx, cur_row in mini_df.iterrows():
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    diameter_x = cur_row['diameterX']
                    diameter_y = cur_row['diameterY']
                    diameter_z = cur_row['diameterZ']
                    label = cur_row["fps"]
                    # nodule center
                    center = np.array([node_x, node_y, node_z])
                    coor_name = str(int(node_x)) + '_' + str(int(node_y)) + '_' + str(int(node_z))
                    # nodule center in voxel space (still x,y,z ordering)  # clip prevents going out of bounds in Z
                    # v_center = np.rint((center - origin) / spacing)
                    v_center = center
                    # convert x,y,z order v_center to z,y,x order v_center
                    v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
                    # node_cube = get_cube_from_img(img_array, v_center, int(diameter_x),int(diameter_y),int(diameter_z))
                    node_cube = get_cube_from_img_1_5(img_array, v_center, int(diameter_x),int(diameter_y),int(diameter_z))
                    node_cube.astype(np.uint8)
                    save_size = node_cube.shape[0]

                    # save as bmp file
                    x = 0
                    for i in range(save_size):
                        # if label == 1:
                        #     filepath = output_path + "1/" + str(img_file) + "_" + coor_name + "/"
                        #     if not os.path.exists(filepath):
                        #         os.makedirs(filepath)
                        #     n = str(i)
                        #     z = n.zfill(3)
                        #     cv2.imwrite(filepath + z + ".bmp", node_cube[i])
                        # if label == 0:
                        #     filepath = output_path + "0/" + str(img_file) + "_" + coor_name + "/"
                        #     if not os.path.exists(filepath):
                        #         os.makedirs(filepath)
                        #     n = str(i)
                        #     z = n.zfill(3)
                        #     cv2.imwrite(filepath + z + ".bmp", node_cube[i])
                        if save_size <= 11:

                            if label == 1:
                                filepath = output_path + "1/" + str(img_file) + "_" + str(coor_name) + "/"
                                if not os.path.exists(filepath):
                                    os.makedirs(filepath)
                                n = str(i)
                                z = n.zfill(3)
                                cv2.imwrite(filepath + z + ".bmp", node_cube[i])

                            if label == 0:
                                filepath = output_path + "0/" + str(img_file) + "_" + str(coor_name) + "/"
                                if not os.path.exists(filepath):
                                    os.makedirs(filepath)
                                n = str(i)
                                z = n.zfill(3)
                                cv2.imwrite(filepath + z + ".bmp", node_cube[i])

                        if save_size > 11:
                            num = int((save_size - 11) /2)
                            if i >= num  and i <= (save_size -num):
                                if label == 1:
                                    filepath = output_path + "1/" + str(img_file) + "_" + str(coor_name) + "/"
                                    if not os.path.exists(filepath):
                                        os.makedirs(filepath)
                                    n = str(x)
                                    z = n.zfill(3)
                                    cv2.imwrite(filepath + z + ".bmp", node_cube[i])

                                if label == 0:
                                    filepath = output_path + "0/" + str(img_file) + "_" + str(coor_name) + "/"
                                    if not os.path.exists(filepath):
                                        os.makedirs(filepath)
                                    n = str(x)
                                    z = n.zfill(3)
                                    cv2.imwrite(filepath + z + ".bmp", node_cube[i])

                                x = x +1

                    index += 1
                    # save as npy file
                    # if label == 1:
                    #     filepath = output_path + "1/"
                    #     if not os.path.exists(filepath):
                    #         os.makedirs(filepath)
                    #     filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                    #     np.save(filepath + filename + ".npy", node_cube)
                    # if label == 0:
                    #     filepath = output_path + "0/"
                    #     if not os.path.exists(filepath):
                    #         os.makedirs(filepath)
                    #     filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                    #     np.save(filepath + filename + ".npy", node_cube)
                    # index += 1


get_node_classify()
