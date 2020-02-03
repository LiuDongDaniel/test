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

    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower

    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    sitktructedimage.SetOrigin(origin)
    sitktructedimage.SetSpacing(spacing)

    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage



def get_cube_from_img(img3d, center, diameter_x,diameter_y,diameter_z):
    center_z = center[0]
    center_y = center[1]
    center_x = center[2]

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

    return roi_img3d


# get 1.5 times
def get_cube_from_img_1_5(img3d, center, diameter_x,diameter_y,diameter_z):
    center_z = center[0]
    center_y = center[1]
    center_x = center[2]

    start_x = max(center_x - (diameter_x * 0.75), 0)
    if start_x + diameter_x *1.5  > img3d.shape[2]:
        start_x = img3d.shape[2] - diameter_x *1.5
    start_y = max(center_y - (diameter_y *0.75), 0)
    if start_y + diameter_y * 1.5 > img3d.shape[1]:
        start_y = img3d.shape[1] - diameter_y *1.5
    start_z = max(center_z - diameter_z/2 , 0)
    if start_z + diameter_z > img3d.shape[0]:
        start_z = img3d.shape[0] - diameter_z
    start_z = int(round(start_z))
    start_y = int(round(start_y))
    start_x = int(round(start_x))

    roi_img3d = img3d[start_z:int(start_z + diameter_z), start_y:int(start_y + diameter_y * 1.5 + 1), start_x:int(start_x + diameter_x * 1.5 + 1)]

    return roi_img3d



# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f[0]:
            return (f[0])


def get_node_classify():
    # luna_train_path_csv = pd.read_csv('/home/huiying/luna/11.22/val_cube/luna_val_45.csv')
    luna_train_path_csv = pd.read_csv('/home/huiying/luna/11.30/test/luna_test.csv')
    output_path = "/home/huiying/luna/11.30/test/"
    luna_train_path_arr = np.array(luna_train_path_csv)
    file_list = luna_train_path_arr.tolist()
    imgs_path = '/mnt/sdb1/luna_raw/'
    # imgs_path = "/home/huiying/luna/11.22/annotation_mask/"

    # The locations of the nodes
    df_node = pd.read_csv("/home/huiying/luna/11.30/test/iou_test.csv")

    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()
    # Looping over the image files(enumerate for find the index and value)
    for fcount, img_name in enumerate(tqdm(file_list)):
        # get all nodules associate with file
        mini_df = df_node[df_node["file"] == img_name[0]]
        # some files may not have a nodule--skipping those
        img_path = imgs_path + img_name[0] + '.mhd'
        if mini_df.shape[0] > 0:
            # load the data once()
            itk_img = load_itkfilewithtrucation(img_path, 600, -1000)
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
                diameter_x = cur_row["diameterX"]
                diameter_y = cur_row["diameterY"]
                diameter_z = cur_row["diameterZ"]

                diameter_x = round(diameter_x / spacing[0])
                diameter_y = round(diameter_y / spacing[1])
                diameter_z = round(diameter_z / spacing[2])
                # diameter = 12
                # diameter_x = round(diameter / spacing[0])
                # diameter_y = round(diameter / spacing[1])
                # diameter_z = round(diameter / spacing[2])
                label = cur_row["label"]

                # label =1




                # nodule center
                center = np.array([node_x, node_y, node_z])
                coor_name = str(int(node_x)) + '_' + str(int(node_y)) + '_' + str(int(node_z))
                # nodule center in voxel space (still x,y,z ordering)  # clip prevents going out of bounds in Z
                v_center = np.rint((center - origin) / spacing)
                # convert x,y,z order v_center to z,y,x order v_center
                v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
                # node_cube = get_cube_from_img(img_array, v_center, int(diameter_x),int(diameter_y),int(diameter_z))
                node_cube = get_cube_from_img_1_5(img_array, v_center, diameter_x, diameter_y, diameter_z)
                node_cube.astype(np.uint8)
                save_size = node_cube.shape[0]

                # save as bmp file
                x = 0
                for i in range(save_size):

                    if save_size <= 11:

                        if label == 1:
                            filepath = output_path + "1/" + str(img_name[0]) + "_" + str(coor_name) + "/"
                            if not os.path.exists(filepath):
                                os.makedirs(filepath)
                            n = str(i)
                            z = n.zfill(3)
                            cv2.imwrite(filepath + z + ".bmp", node_cube[i])

                        if label == 0:
                            filepath = output_path + "0/" + str(img_name[0]) + "_" + str(coor_name) + "/"
                            if not os.path.exists(filepath):
                                os.makedirs(filepath)
                            n = str(i)
                            z = n.zfill(3)
                            cv2.imwrite(filepath + z + ".bmp", node_cube[i])

                    if save_size > 11:
                        num = int((save_size - 11) / 2)
                        if i >= num and i <= (save_size - num):
                            if label == 1:
                                filepath = output_path + "1/" + str(img_name[0]) + "_" + str(coor_name) + "/"
                                if not os.path.exists(filepath):
                                    os.makedirs(filepath)
                                n = str(x)
                                z = n.zfill(3)
                                cv2.imwrite(filepath + z + ".bmp", node_cube[i])

                            if label == 0:
                                filepath = output_path + "0/" + str(img_name[0]) + "_" + str(coor_name) + "/"
                                if not os.path.exists(filepath):
                                    os.makedirs(filepath)
                                n = str(x)
                                z = n.zfill(3)
                                cv2.imwrite(filepath + z + ".bmp", node_cube[i])

                            x = x + 1

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
