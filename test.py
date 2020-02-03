import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import sys
from data.data import Dataset
# from data.data_3d import Dataset
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from model.model import Simple_CNN
from model.model import Simple_CNN_3d
from model.model import LeNet5
from model.model import Simple_CNN_30
from train import parse_args
from sklearn.metrics import roc_curve, auc
import numpy as np

"""
using for test the model, can generate some metrics of the model: 
a. split the test image to TP, TN, FP, FN
b. generate the csv file for the predict_value and the label
c. generate the ROC curve
d. generate the FROC curve
e. generate the recall rate and the  precision rate
"""

def gene_recall_and_precison(actual_val_list, predict_val_list, pos_label):
    actual_vals = np.array(actual_val_list)
    predict_vals = np.array(predict_val_list)
    if pos_label == 0:
        combine = actual_vals + predict_vals
        TP = sum(combine==0)
        recall = TP / sum(actual_vals==0)
        precision = TP / sum(predict_vals==0)
    if pos_label == 1:
        combine = actual_vals * predict_vals
        TP = sum(combine==1)
        recall = TP / sum(actual_vals==1)
        precision = TP / sum(predict_vals==1)
    print('recall: %.4f' % recall)
    print( 'precision: %.4f' % precision)







def gene_free_roc_curve(actual_val_list, score_list, pos_label, total_number_of_image):
    fpr, tpr, threshold = roc_curve(actual_val_list, score_list, pos_label)
    if pos_label == 0:
        false_positive_numbers = sum(actual_val_list)
    if pos_label == 1:
        false_positive_numbers = len(actual_val_list) - sum(actual_val_list)
    avg_num_false_positive = fpr * false_positive_numbers / total_number_of_image

    plt.plot(avg_num_false_positive, tpr, color='b', lw=2)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.125, 8])
    plt.ylim([0, 1.1])
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('True Positive Rate')
    plt.title('FROC performence')
    plt.show()





def gene_roc_curve(actual_val_list, score_list, pos_label):
    fpr, tpr, threshold = roc_curve(actual_val_list, score_list, pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # set the size of letter
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)


    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()



def img_split_save(test_img_save_path, img_name, img, actual_val, idx):
    if actual_val.numpy()[0] == 1 and idx.numpy()[0] == 1:
        save_path_sub = test_img_save_path + 'TP/'
        if not os.path.exists(save_path_sub):
            os.makedirs(save_path_sub)
        cv2.imwrite(save_path_sub + img_name[0] + '.jpg', img.numpy()[0])
    if actual_val.numpy()[0] == 1 and idx.numpy()[0] == 0:
        save_path_sub = test_img_save_path + 'FN/'
        if not os.path.exists(save_path_sub):
            os.makedirs(save_path_sub)
        cv2.imwrite(save_path_sub + img_name[0] + '.jpg', img.numpy()[0])
    if actual_val.numpy()[0] == 0 and idx.numpy()[0] == 1:
        save_path_sub = test_img_save_path + 'FP/'
        if not os.path.exists(save_path_sub):
            os.makedirs(save_path_sub)
        cv2.imwrite(save_path_sub + img_name[0] + '.jpg', img.numpy()[0])
    if actual_val.numpy()[0] == 0 and idx.numpy()[0] == 0:
        save_path_sub = test_img_save_path + 'TN/'
        if not os.path.exists(save_path_sub):
            os.makedirs(save_path_sub)
        cv2.imwrite(save_path_sub + img_name[0] + '.jpg', img.numpy()[0])



def test_model(model_path, predict_csv_save_name, test_img_save_path):
    args = parse_args()
    # load test data
    dataset_test = Dataset(args.test_csv_path)
    test_loader = torch.utils.data.DataLoader(dataset_test)
    # load model
    # model = Simple_CNN()
    # model = LeNet5()
    model = Simple_CNN_30()
    # resnet = torchvision.models.resnet18(pretrained=True)
    # resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    # resnet.fc = nn.Linear(in_features=512, out_features=2)
    # model = resnet

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])
    # for save value to csv
    out = open(predict_csv_save_name, 'w')
    out.writelines("name" + "," + "label" + "," + "predict" + "\n")
    # for roc
    pos_score_list = []
    actual_val_list  = []
    predict_label_list = []

    for num, test_item in enumerate(test_loader):
        inputs, actual_val, img_name, img = test_item
        predicted_val = model(inputs)
        predicted_val = predicted_val.data
        max_score, predict_label = torch.max(predicted_val, 1)

        # for save_img
        # img_split_save(test_img_save_path, img_name, img, actual_val, predict_label)
        #save predict to csv
        # out.writelines(img_name[0] + "," +str(actual_val.numpy()[0]) + "," + str(predict_label.numpy()[0]) + "\n")

        pos_score = predicted_val.numpy()[0][1]
        pos_score_list.append(pos_score)
        actual_val_list.append(actual_val.numpy()[0])
        predict_label_list.append(predict_label.numpy()[0])

        # print('label:', actual_val.numpy()[0], 'predict:', predict_label.numpy()[0], max_score.numpy()[0])
    gene_roc_curve(actual_val_list, pos_score_list, 1)
    # gene_free_roc_curve(actual_val_list, pos_score_list, 1, 163)
    gene_recall_and_precison(actual_val_list, predict_label_list, 1)







model_path = '/home/huiying/PycharmProjects/test/batch32_lr0.0003_weight_decay0_time2019-11-21-16:22/BEST_EP24_LOSS0.0941'
predict_csv_save_name = '/home/huiying/PycharmProjects/test/batch32_lr0.0003_weight_decay0_time2019-11-21-16:22/predict.csv'
test_img_save_path = '/home/huiying/PycharmProjects/test/batch32_lr0.0003_weight_decay0_time2019-11-21-16:22/'
test_model(model_path, predict_csv_save_name, test_img_save_path)





