import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2 as cv
import re
import SimpleITK as sitk
from scipy.io import loadmat
import h5py

from MeDIT.Visualization import Imshow3D, Imshow3DArray

def find_numfromstr(str):
    f = re.findall('(\d+)', str)
    num = int(f[0])
    return num

def hlabel_to_onehot(label):
    label_all = np.zeros((0, label.shape[0], label.shape[1], 4))
    # print(label.shape)
    for m in range(label.shape[2]):
        print(m)
        label_one = label[:, :, m]
        label_one = np.reshape(label_one, (1, label_one.shape[0],label_one.shape[1], 1))
        label_b = None
        label_1 = np.zeros((1, label_one.shape[1], label_one.shape[2], 1))
        label_2 = np.zeros((1, label_one.shape[1], label_one.shape[2], 1))
        label_3 = np.zeros((1, label_one.shape[1], label_one.shape[2], 1))
        for i in range(label_one.shape[1]):
            for j in range(label_one.shape[2]):
                if label_one[0, i, j, 0] == 4:
                    label_1[0, i, j, 0] = 1
                elif label_one[0, i, j, 0] == 5:
                    label_2[0, i, j, 0] = 1
                elif label_one[0, i, j, 0] == 6:
                    label_3[0, i, j, 0] = 1
                else:
                    label_b = 1 - label_1 - label_2 -label_3
        label_c = np.concatenate((label_b, label_1, label_2, label_3), axis=3)
        # print(label_c.shape)
        # if np.sum(label_c[0, :, :, 0]) != label_c.shape[1]*label_c.shape[2]:
        #     plt.imshow(label_c[0, :, :, 0],cmap='gray')
        #     plt.show()
        # print(label_c.shape)
        label_all = np.concatenate((label_all, label_c), axis=0)
    return label_all
CNN_out_path =r'D:/htfg_all/CNN'
dir1 = r'D:/htfg_all/CNN/'
image_path = r'D:/htfg_all/RESULT1_2'
nums=[]
def creath5():
    for sub_name in os.listdir(CNN_out_path):
        num = find_numfromstr(sub_name)
        sub_path_CNN = os.path.join(CNN_out_path, sub_name)
        sub_path_image = os.path.join(image_path, sub_name)
        print(sub_path_CNN)
        print(sub_path_image)
        image_array = []
        label_array = []
        CNN_array = []

        for filename_image in os.listdir(sub_path_image):
            for filename_CNN in os.listdir(sub_path_CNN):
                if 'M.nii' in filename_image:
                    sub_sub_path_image = os.path.join(sub_path_image, filename_image)
                    print(sub_sub_path_image)
                    image = sitk.ReadImage(sub_sub_path_image)
                    image_array = sitk.GetArrayFromImage(image)
                    image_array = image_array.transpose((0, 2, 1))
                    print(image_array.shape)
                if 'nii.gz' in filename_image:
                    sub_sub_path_label = os.path.join(sub_path_image, filename_image)
                    print(sub_sub_path_label)
                    label = sitk.ReadImage(sub_sub_path_label)
                    label_array = sitk.GetArrayFromImage(label)
                    label_array = label_array.transpose((0, 2, 1))
                    print(label_array.shape)
                if 'nii.gz' in filename_CNN:
                    sub_sub_path_CNN = os.path.join(sub_path_CNN, filename_CNN)
                    print(sub_sub_path_CNN)
                    CNN_out = sitk.ReadImage(sub_sub_path_CNN)
                    CNN_array = sitk.GetArrayFromImage(CNN_out)
                    CNN_array = CNN_array.transpose((0, 2, 1))
                    print(CNN_array.shape)
        print(image_array.shape)
        print(label_array.shape)
        print(CNN_array.shape)
        if num>=26:

            label_onehot = hlabel_to_onehot(label_array[ :, :, 90:190])
            CNN_onehot = hlabel_to_onehot(CNN_array[:, :, 90:190])

            file_train = h5py.File(dir1 + 'test_' + str(num) + '.h5', 'w')
            file_train.create_dataset('test_data', data=image_array[:, :, 90:190])
            file_train.create_dataset('test_label', data=label_onehot)
            file_train.create_dataset('test_CNN', data=CNN_onehot)
            file_train.close()


    # for j in range(image_array.shape[2]):
    #     if np.sum(label_array[j, :, :, 0]) != label_array.shape[1]*label_array.shape[2] \
    #             or np.sum(CNN_array[:, :, j]) != CNN_array.shape[1]*CNN_array.shape[2]:
    #
    #         plt.figure('p', figsize=(12, 12))
    #         plt.subplot(121)
    #         plt.imshow(image_array[:, :, j], cmap='gray', vmin=-0.05, vmax=0.2)
    #         plt.contour(label_array[j, :, :, 1], linewidths=0.19, cmap=ListedColormap('red'))
    #         plt.contour(label_array[j, :, :, 2], linewidths=0.19, cmap=ListedColormap('green'))
    #         plt.contour(label_array[j, :, :, 3], linewidths=0.19, cmap=ListedColormap('yellow'))
    #         plt.subplot(122)
    #         plt.imshow(image_array[:, :, j], cmap='gray', vmin=-0.05, vmax=0.2)
    #         plt.contour(CNN_array[j, :, :, 1], linewidths=0.19, cmap=ListedColormap('red'))
    #         plt.contour(CNN_array[j, :, :, 2], linewidths=0.19, cmap=ListedColormap('green'))
    #         plt.contour(CNN_array[j, :, :, 3], linewidths=0.19, cmap=ListedColormap('yellow'))
    #         plt.show()


creath5()

def pltout():
    for sub_name in os.listdir(CNN_out_path):
        num = find_numfromstr(sub_name)
        if num>=22:
            file1 = h5py.File(dir1 + 'test_' + str(num) + '.h5', 'r')
            image_array = file1['test_data'][:]
            label_array = file1['test_label'][:]
            CNN_array = file1['test_CNN'][:]
            file1.close()
            print(num)
            print(image_array.shape)
            print(label_array.shape)
            print(CNN_array.shape)
            # for j in range(140, image_array.shape[2]):
            #     plt.imshow(label_array[j, :, :, 0], cmap='gray', vmin=-0.05, vmax=0.2)
            #     # plt.contour(label_array[150, :, :, 1], linewidths=0.19, cmap=ListedColormap('red'))
            #     # plt.contour(label_array[150, :, :, 2], linewidths=0.19, cmap=ListedColormap('green'))
            #     # plt.contour(label_array[150, :, :, 3], linewidths=0.19, cmap=ListedColormap('yellow'))
            #     plt.show()
            for j in range(image_array.shape[2]):
                if np.sum(label_array[j, :, :, 0]) != label_array.shape[1]*label_array.shape[2] \
                        or np.sum(CNN_array[j, :, :, 0]) != CNN_array.shape[1]*CNN_array.shape[2]:
                    print(j)
                    plt.figure('p', figsize=(12, 12))
                    plt.subplot(121)
                    plt.imshow(image_array[:, :, j], cmap='gray', vmin=-0.05, vmax=0.2)
                    plt.contour(label_array[j, :, :, 1], linewidths=0.19, cmap=ListedColormap('red'))
                    plt.contour(label_array[j, :, :, 2], linewidths=0.19, cmap=ListedColormap('green'))
                    plt.contour(label_array[j, :, :, 3], linewidths=0.19, cmap=ListedColormap('yellow'))
                    plt.subplot(122)
                    plt.imshow(image_array[:, :, j], cmap='gray', vmin=-0.05, vmax=0.2)
                    plt.contour(CNN_array[j, :, :, 1], linewidths=0.19, cmap=ListedColormap('red'))
                    plt.contour(CNN_array[j, :, :, 2], linewidths=0.19, cmap=ListedColormap('green'))
                    plt.contour(CNN_array[j, :, :, 3], linewidths=0.19, cmap=ListedColormap('yellow'))
                    plt.show()

# pltout()