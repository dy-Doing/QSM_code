import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2 as cv

import SimpleITK as sitk
from scipy.io import loadmat
import h5py
from MeDIT.DataAugmentor import DataAugmentor3D, AugmentParametersGenerator
from MeDIT.Visualization import Imshow3D, Imshow3DArray

d_path = r'D:/htfg_all/LiGaiYing'
dir1 = r'C:/Users/ZHOUFF/PycharmProjects/htfg/data/d_h5/'
target_patch = 128


def get_target_dlabel(sub_path):
    for filename_1 in os.listdir(sub_path):
        if '.nii.gz' in filename_1:
            sub_sub_path_1 = os.path.join(sub_path, filename_1)
            # print(1)
            # print(sub_sub_path_1)
            label = sitk.ReadImage(sub_sub_path_1)
            label_array = sitk.GetArrayFromImage(label)
            transfored_label = label_array.transpose((1,2,0))
            print(transfored_label.shape)
            # for i in range(transfored_label.shape[2]):
            #     if np.sum(transfored_label[:, :, i]) != 0:
            #         print(i)
            #         plt.imshow(transfored_label[:, :, i], cmap='gray')
            #         plt.show()

            label_all = np.zeros((transfored_label.shape[2], transfored_label.shape[0], transfored_label.shape[1],1))
            for m in range(transfored_label.shape[2]):
                label_0 = transfored_label[:, :, m]
                label_new_1 = np.zeros((label_0.shape[0], label_0.shape[1]))
                for i in range(label_0.shape[0]):
                    for j in range(label_0.shape[1]):
                        if label_0[i, j] == 4:
                            label_new_1[i, j] = 1
                        elif label_0[i, j] == 5:
                            label_new_1[i, j] = 2
                        else:
                            pass
                label_new_1 = cv.flip(label_new_1, 0, dst=None)

                # if np.sum(label_new_1) != 0:
                #     print(m)
                #     plt.imshow(label_new_1, cmap='gray')
                #     plt.show()
                label_all_0 = np.reshape(label_new_1, (1, label_new_1.shape[0], label_new_1.shape[1], 1))
                label_all[m, :, :, 0:1] = label_all_0

            return label_all

def get_target_dimage(sub_path):
    for filename_1 in os.listdir(sub_path):
        if '.img' in filename_1:
            sub_sub_path_2 = os.path.join(sub_path, filename_1)
            # print(2)
            # print(sub_sub_path_2)
            image = sitk.ReadImage(sub_sub_path_2)
            image_array = sitk.GetArrayFromImage(image)
            transfored_image = image_array.transpose((1, 2, 0))
            # for i in range(30, transfored_image.shape[2]):
            #     plt.imshow(transfored_image[:, :, i], cmap='gray')
            #     plt.show()
            image_all = np.zeros((transfored_image.shape[2], transfored_image.shape[0], transfored_image.shape[1], 1))
            for j in range(transfored_image.shape[2]):
                image_trans = transfored_image[:, :, j]
                image_all_0 = cv.flip(image_trans, 0, dst=None)

                image_all_0 = np.reshape(image_all_0, (1, image_all_0.shape[0], image_all_0.shape[1], 1))
                image_all[j, :, :, 0:1] = image_all_0
            return image_all

def crop_data_to_target(train_image, train_label, target_patch):
    train_image_crop = train_image[:, train_image.shape[1]//2 - target_patch//2:train_image.shape[1]//2 + target_patch//2,
                       train_image.shape[2]//2 - target_patch//2:train_image.shape[2]//2 + target_patch//2, :]
    train_label_crop = train_label[:, train_label.shape[1]//2 - target_patch//2:train_label.shape[1]//2 + target_patch//2,
                       train_label.shape[2]//2 - target_patch//2:train_label.shape[2]//2 + target_patch//2, :]
    return train_image_crop, train_label_crop

def right_data(k, no_label_number,number_range, transfored_image,label_array):
    label_array_crop_r = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 1))
    transfored_image_crop_r = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 3))
    for i in range(no_label_number):
        j = random.randint(k, k+number_range)

        label_array_crop_r = np.concatenate((label_array_crop_r,label_array[j:j + 1, :, :, :]), axis=0)

        image_trans = np.transpose(transfored_image[j - 1:j + 2, :, :, :], (3, 1, 2, 0))
        transfored_image_crop_r = np.concatenate((transfored_image_crop_r, image_trans), axis=0)

    return transfored_image_crop_r, label_array_crop_r

def left_data(k, no_label_number,number_range, transfored_image, label_array):
    transfored_image_crop_l = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 3))
    label_array_crop_l = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 1))
    for i in range(no_label_number):
        j = random.randint(k-number_range, k)

        label_array_crop_l = np.concatenate((label_array_crop_l, label_array[j:j + 1, :, :, :]), axis=0)

        image_trans = np.transpose(transfored_image[j - 1:j + 2, :, :, :],(3,1,2,0))
        transfored_image_crop_l = np.concatenate((transfored_image_crop_l, image_trans), axis=0)

    return transfored_image_crop_l,label_array_crop_l

def get_train_ddata(transfored_image, label_array, no_label_number,number_range):
    label_array_crop_l = None
    transfored_image_crop_l = None
    label_array_crop_r = None
    transfored_image_crop_r = None

    label_array_crop_h_all = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 1))
    transfored_image_crop_h_all = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 3))

    for k in range(1, label_array.shape[0] - 1):

        if np.sum(label_array[k - 1, :, :, :]) == 0 and np.sum(label_array[k, :, :, :]) != 0:
            transfored_image_crop_l, label_array_crop_l = left_data(k, no_label_number,number_range, transfored_image, label_array)
            print(label_array_crop_l.shape)

        elif np.sum(label_array[k - 1, :, :, :]) != 0 and np.sum(label_array[k, :, :, :]) == 0:
            transfored_image_crop_r, label_array_crop_r = right_data(k, no_label_number, number_range, transfored_image, label_array)
            print(label_array_crop_r.shape)

        elif np.sum(label_array[k, :, :, :]) != 0:

            label_array_crop_h_all = np.concatenate((label_array_crop_h_all, label_array[k:k + 1, :, :, :]), axis=0)

            image_trans_h = np.transpose(transfored_image[k - 1:k + 2, :, :, :], (3, 1, 2, 0))
            transfored_image_crop_h_all = np.concatenate((transfored_image_crop_h_all, image_trans_h), axis=0)
        else:
            pass
    label_0 = np.concatenate((label_array_crop_l, label_array_crop_h_all, label_array_crop_r), axis=0)
    image_0 = np.concatenate((transfored_image_crop_l, transfored_image_crop_h_all, transfored_image_crop_r), axis=0)
    print(label_0.shape)
    print(image_0.shape)
    return image_0, label_0

def dataAug(data, label):
    random_params = {'shear': 0.1, 'shift_x': 10, 'shift_y': 10, 'rotate_z_angle': 5, 'stretch_x': 0.1, 'stretch_y': 0.1}
    param_generator = AugmentParametersGenerator()
    aug_generator = DataAugmentor3D()
    while True:
        param_generator.RandomParameters(random_params)
        aug_generator.SetParameter(param_generator.GetRandomParametersDict())
        data = aug_generator.Execute(source_data=data, interpolation_method='linear', is_clear=False)
        label = aug_generator.Execute(source_data=label, interpolation_method='nearest', is_clear=False)
        return data, label

def input_ddata_to_h5py(d_path):
    count = 0
    for sub_name in os.listdir(d_path):
        sub_path = os.path.join(d_path, sub_name)
        print(sub_path)
        count += 1
        print(count)
        label_array = get_target_dlabel(sub_path)
        transfored_image = get_target_dimage(sub_path)
        for h in range(transfored_image.shape[0]):
            if np.sum(label_array[h, :, :, 0]) != 0:
                print('1')
                plt.imshow(transfored_image[h,:, :, 0], cmap='gray',vmin=-100, vmax=200)
                plt.contour(label_array[h, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                plt.show()

        train_image_crop, train_label_crop = crop_data_to_target(transfored_image,
                                                                 label_array, 128)
        for h in range(transfored_image.shape[0]):
            if np.sum(label_array[h, :, :, 0]) != 0:
                print('2')
                plt.imshow(train_image_crop[h,:, :, 0], cmap='gray',vmin=-100, vmax=200)
                # plt.contour(train_label_crop[h, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                plt.show()

        train_image_crop, train_label_crop = get_train_ddata(train_image_crop,
                                                             train_label_crop, no_label_number=2,
                                                              number_range=5)
        for h in range(train_image_crop.shape[0]):
            if np.sum(train_label_crop[h, :, :, 0]) != 0:

                print('3')
                print('3_1')
                plt.imshow(train_image_crop[h, :, :, 0], cmap='gray', vmin=-100, vmax=200)
                # plt.contour(train_label_crop[h-1, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                plt.show()
                print('3_2')
                plt.imshow(train_image_crop[h, :, :, 1], cmap='gray', vmin=-100, vmax=200)
                # plt.contour(train_label_crop[h, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                plt.show()
                print('3_3')
                plt.imshow(train_image_crop[h, :, :, 2], cmap='gray', vmin=-100, vmax=200)
                # plt.contour(train_label_crop[h + 1, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                plt.show()

                train_image_crop[h, :, :, 0:3],train_label_crop[h, :, :, 0:1] = \
                       dataAug(train_image_crop[h, :, :, 0:3],train_label_crop[h, :, :, 0:1])
                if np.sum(train_label_crop[h, :, :, 0:1]) !=0:
                    print('4')
                    print('4_1')
                    plt.imshow(train_image_crop[h,:, :, 0], cmap='gray',vmin=-100, vmax=200)
                    # plt.contour(train_label_crop[h-1, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                    plt.show()
                    print('4_2')
                    plt.imshow(train_image_crop[h, :, :, 1], cmap='gray', vmin=-100, vmax=200)
                    # plt.contour(train_label_crop[h, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                    plt.show()
                    print('4_3')
                    plt.imshow(train_image_crop[h, :, :, 2], cmap='gray', vmin=-100, vmax=200)
                    # plt.contour(train_label_crop[h + 1, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                    plt.show()

                    train_image_crop_2, train_label_crop_2 = crop_data_to_target(train_image_crop,
                                                                                 train_label_crop, 96)
                    print('5')
                    print('5_1')
                    plt.imshow(train_image_crop_2[h,:, :, 0], cmap='gray',vmin=-100, vmax=200)
                    # plt.contour(train_label_crop_2[h, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                    plt.show()
                    print('5_2')
                    plt.imshow(train_image_crop_2[h, :, :, 1], cmap='gray', vmin=-100, vmax=200)
                    # plt.contour(train_label_crop_2[h, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                    plt.show()
                    print('5_3')
                    plt.imshow(train_image_crop_2[h, :, :, 2], cmap='gray', vmin=-100, vmax=200)
                    # plt.contour(train_label_crop_2[h, :, :, 0], linewidths=0.19, cmap=ListedColormap('red'))
                    plt.show()
def main():

    input_ddata_to_h5py(d_path)


main()
