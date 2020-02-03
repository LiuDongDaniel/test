from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import os
from glob import glob
import pandas as pd
import cv2

"""
using the seriesuid csv, annotation_csv and headnorm_csv to calculate the iou between annotation_csv and headnorm csv.
the parameter introduction:
-- imgs_path: the path of the original ct files( form as mhd) which is needed to read the ct information such as coordinateX,Y,Z and diameterX,Y,Z
-- inter_union_save_path: the path for saving the final output which is a csv form file
-- annotation_node: the annotation file which is a csv form file including the seriesuid, coordinateX,Y,Z and diameterX,Y,Z
-- headnorm_node: the form is as same as annotation_node
"""


tqdm = lambda x: x

def get_filename(file_list, case):
    for f in file_list:
        if case in f[0]:
            return (f[0])


# get mask_region from csv data
def make_mask(mask, v_center, diameter_x, diameter_y, diameter_z):
    radius_x = np.rint(diameter_x / 2)
    radius_y = np.rint(diameter_y / 2)
    radius_z = np.rint(diameter_z / 2)
    center_z = v_center[0]
    center_y = v_center[1]
    center_x = v_center[2]

    z_min = int(center_z - radius_z)
    z_max = int(center_z + radius_z + 1)
    y_min = int(center_y - radius_y)
    y_max = int(center_y + radius_y + 1)
    x_min = int(center_x - radius_x)
    x_max = int(center_x + radius_x + 1)
    mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1.0

def calculate_nodule_mask(cur_row, spacing, origin,mask):
    node_x = cur_row["coordX"]
    node_y = cur_row["coordY"]
    node_z = cur_row["coordZ"]
    diameter_x = cur_row["diameterX"]
    diameter_y = cur_row["diameterY"]
    diameter_z = cur_row["diameterZ"]

    diameter_x = round(diameter_x / spacing[0])
    diameter_y = round(diameter_y / spacing[1])
    diameter_z = round(diameter_z / spacing[2])

    center = np.array([node_x, node_y, node_z])
    # nodule center
    v_center = np.rint((center - origin) / spacing)
    # convert x,y,z order v_center to z,y,x order v_center
    v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
    make_mask(mask, v_center, diameter_x, diameter_y, diameter_z)

def calculate_IOU(anno_mask,head_mask):
    intersection = np.multiply(anno_mask,head_mask)
    inter = intersection.sum()

    union = anno_mask + head_mask
    union[union < 0] = 0
    union[union > 0] = 1
    union = union.sum()
    anno_sum = anno_mask.sum()
    head_sum = head_mask.sum()
    iou = 0
    if union != 0:
        iou = float('%.4f' % (inter / union))
    return anno_sum, head_sum, inter, union, iou


def gene_inter_union_csv(imgs_path, inter_union_save_path, luna_train_path_csv, annotation_node, headnorm_node):
    # get the seriesuid list
    luna_train_path_csv = pd.read_csv(luna_train_path_csv)
    luna_train_path_arr = np.array(luna_train_path_csv)
    file_list = luna_train_path_arr.tolist()

    # get the annotation list and map with seriesuid
    annotation_node = pd.read_csv(annotation_node)
    annotation_node["file"] = annotation_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    annotation_node = annotation_node.dropna()

    # get the headnorm list and map with seriesuid
    headnorm_node = pd.read_csv(headnorm_node)
    headnorm_node["file"] = headnorm_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    headnorm_node = headnorm_node.dropna()

    # build the save csv
    build = open(inter_union_save_path, 'w')
    build.writelines("seriesuid" + "," + "coordX" + "," + "coordY" + "," + "coordZ" + "," + "diameterX" + "," + "diameterY" + "," + "diameterZ" + "," + "score" + "," +
                     "anno_coordX" + "," + "anno_coordY" + "," + "anno_coordZ" + "," + "diameterX" + "," + "diameterY" + "," + "diameterZ" + "," +
                     "anno" + "," + "head" + "," + "inter" + "," + "union" + ","+"iou" + "\n")


    # loop the file_list
    for fcount, img_name in enumerate(tqdm(file_list)):
        # choose the corresponding seriesuid in annotation and headnorm
        img_name = img_name[0]
        seriesuid_annotation_node = annotation_node[annotation_node["file"] == img_name]
        seriesuid_headnorm_node = headnorm_node[headnorm_node["file"] == img_name]
        img_path = imgs_path + img_name + '.mhd'

        # get the information of the ct
        itk_img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(itk_img)
        num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())

        #loop the headnorm_node
        if seriesuid_headnorm_node.shape[0] > 0:
            for node_idx, head_row in seriesuid_headnorm_node.iterrows():
                head_mask = np.zeros(shape=(num_z, height, width), dtype=np.float)
                calculate_nodule_mask(head_row, spacing, origin, head_mask)

                # save the value
                node_x = head_row["coordX"]
                node_y = head_row["coordY"]
                node_z = head_row["coordZ"]
                diameter_x = head_row["diameterX"]
                diameter_y = head_row["diameterY"]
                diameter_z = head_row["diameterZ"]
                score = head_row["probability"]


                # there have true nodules in this seriesuid
                if seriesuid_annotation_node.shape[0] > 0:
                    # loop the annotaion nodles and build the mask for each annotation nodule
                    anno_list = []
                    for node_idx, cur_row in seriesuid_annotation_node.iterrows():
                        anno_mask = np.zeros(shape=(num_z, height, width), dtype=np.float)
                        calculate_nodule_mask(cur_row, spacing, origin, anno_mask)

                        # calculate inter, union
                        anno_sum, head_sum, inter, union, iou = calculate_IOU(anno_mask, head_mask)

                        # mark the coord for the anno_node
                        anno_node_x = cur_row["coordX"]
                        anno_node_y = cur_row["coordY"]
                        anno_node_z = cur_row["coordZ"]
                        anno_diameter_x = cur_row["diameterX"]
                        anno_diameter_y = cur_row["diameterY"]
                        anno_diameter_z = cur_row["diameterZ"]
                        anno_anno = str(anno_sum)
                        anno_head = str(head_sum)
                        anno_inter = str(inter)
                        anno_union = str(union)
                        anno_iou = iou

                        anno = [anno_node_x, anno_node_y, anno_node_z, anno_diameter_x, anno_diameter_y, anno_diameter_z, anno_anno, anno_head, anno_inter, anno_union, anno_iou]
                        anno_list.append(anno)
                        if len(anno_list) > 1:
                            if anno_list[0][10] <= anno_list[1][10]:
                                anno_list.pop(0)
                            else:
                                anno_list.pop(1)

                    build.writelines(
                        str(img_name) + "," + str(node_x) + "," + str(node_y) + "," + str(node_z) + "," + str(diameter_x) + "," + str(diameter_y) + "," + str(diameter_z) + "," + str(score) + "," +
                        str(anno_list[0][0]) + "," + str(anno_list[0][1]) + "," + str(anno_list[0][2]) + "," + str(anno_list[0][3]) + "," + str(anno_list[0][4]) + "," + str(anno_list[0][5]) + "," + str(anno_list[0][6]) + "," +
                        str(anno_list[0][7]) + "," + str(anno_list[0][8]) + "," + str(anno_list[0][9]) + "," + str(anno_list[0][10]) + "\n")
                # there dont have true nodules in this seriesuid
                if seriesuid_annotation_node.shape[0] == 0:
                    build.writelines(
                        str(img_name) + "," + str(node_x) + "," + str(node_y) + "," + str(node_z) + "," + str(diameter_x) + "," + str(diameter_y) + "," + str(diameter_z) + "," + str(score) + "," +
                        "_" + "," + "_" + "," + "_" + "," + "_" + "," + "_" + "," + "_" + "," + "_" + "," +
                        "_" + "," + "_" + "," + "_" + "," + "0" + "\n")










        # there have true nodules in this seriesuid
        # if seriesuid_annotation_node.shape[0] > 0:
        #     #loop the annotaion nodles and build the mask for each annotation nodule
        #     for node_idx, cur_row in seriesuid_annotation_node.iterrows():
        #         anno_mask = np.zeros(shape=(num_z, height, width), dtype=np.float)
        #         calculate_nodule_mask(cur_row, spacing, origin,anno_mask)
        #
        #         if seriesuid_headnorm_node.shape[0] > 0:
        #             # loop the headnorm nodules, building the mask for each headnorm nodules,
        #             # and compare with annotation nodule, calculating the IOU
        #             for node_idx, head_row in seriesuid_headnorm_node.iterrows():
        #                 head_mask = np.zeros(shape=(num_z, height, width), dtype=np.float)
        #                 calculate_nodule_mask(head_row, spacing, origin, head_mask)
        #
        #                 # calculate inter, union
        #                 anno_sum, head_sum, inter, union = calculate_IOU(anno_mask,head_mask)
        #                 iou = 0
        #                 if union != 0:
        #                     iou = float('%.4f' % (inter/union))
        #                 # save the value
        #                 node_x = head_row["coordX"]
        #                 node_y = head_row["coordY"]
        #                 node_z = head_row["coordZ"]
        #                 diameter_x = head_row["diameterX"]
        #                 diameter_y = head_row["diameterY"]
        #                 diameter_z = head_row["diameterZ"]
        #                 build.writelines(str(img_name) + "," + str(node_x) + "," + str(node_y) + "," + str(node_z) + "," + str(diameter_x) + ","
        #                                  + str(diameter_y) + "," + str(diameter_z) + "," + str(anno_sum) + "," + str(head_sum) + "," + str(inter)
        #                                  + "," + str(union) + "," + str(iou) + "\n")
        # # there dont have true nodules in this seriesuid
        # if seriesuid_annotation_node.shape[0] == 0:
        #     # all of the nodules in this seriesuid are false
        #     if seriesuid_headnorm_node.shape[0] > 0:
        #         for node_idx, head_row in seriesuid_headnorm_node.iterrows():
        #             # save the value
        #             node_x = head_row["coordX"]
        #             node_y = head_row["coordY"]
        #             node_z = head_row["coordZ"]
        #             diameter_x = head_row["diameterX"]
        #             diameter_y = head_row["diameterY"]
        #             diameter_z = head_row["diameterZ"]
        #             build.writelines(
        #                 str(img_name) + "," + str(node_x) + "," + str(node_y) + "," + str(node_z) + "," + str(
        #                     diameter_x) + ","
        #                 + str(diameter_y) + "," + str(diameter_z) + "," + "_" + "," + "_" + "," + "_"
        #                 + "," + "_" + "," + str(0) + "\n")





imgs_path = '/mnt/sdb1/luna_raw/'
inter_union_save_path = '/home/huiying/luna/11.22/test_cube/luna_test_iou.csv'
luna_train_path_csv = '/home/huiying/luna/11.22/test_cube/luna_test_88.csv'
annotation_node = "/home/huiying/luna/fp_reduction_1115nms/annotations.csv"
headnorm_node = "/home/huiying/luna/fp_reduction_1115nms/fp_per_scan_8.csv"


gene_inter_union_csv(imgs_path, inter_union_save_path, luna_train_path_csv, annotation_node, headnorm_node )



