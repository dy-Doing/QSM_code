import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2 as cv

import SimpleITK as sitk
from scipy.io import loadmat
import h5py

from MeDIT.Visualization import Imshow3D, Imshow3DArray

# d_path = r'D:/htfg_all/QSM-GYLi'
h_path = r'D:/htfg_all/RESULT1_2'
inputs_path = r'D:\htfg_all\QSMseg_0305'
dir1 = r'C:/Users/ZHOUFF/PycharmProjects/htfg/data/h_h5/'
target_patch = 96


def change_hdr_to_nii(d_path):
    for sub_name in os.listdir(d_path):
        sub_path = os.path.join(d_path, sub_name)
        print(sub_path)
        for filename_1 in os.listdir(sub_path):
            if '.hdr' in filename_1:
                sub_sub_path_1 = os.path.join(sub_path, filename_1)
                image = sitk.ReadImage(sub_sub_path_1)
                image_array = sitk.GetArrayFromImage(image)
                image = sitk.GetImageFromArray(image_array)
                path = sub_sub_path_1[:-4]+'.nii.gz'
                print(path)
                sitk.WriteImage(image, path)


def get_target_hlabel(sub_path):
    for filename_1 in os.listdir(sub_path):
        if '.nii.gz' in filename_1:
            sub_sub_path_1 = os.path.join(sub_path, filename_1)
            # print(1)
            # print(sub_sub_path_1)
            label = sitk.ReadImage(sub_sub_path_1)
            label_array = sitk.GetArrayFromImage(label)
            # print('.......')
            # print(label_array.shape)
            transfored_label = label_array.transpose((0, 2, 1))
            # print(transfored_label.shape)
            label_right_all = np.zeros((transfored_label.shape[2], target_patch, target_patch, 1))
            label_left_to_right_all = np.zeros((transfored_label.shape[2], target_patch, target_patch, 1))
            label_all = np.zeros((transfored_label.shape[2], target_patch, target_patch, 1))

            for m in range(transfored_label.shape[2]):
                # print(m)
                label_0 = transfored_label[:, :, m]
                label_new_1 = np.zeros((label_0.shape[0], label_0.shape[1]))
                for i in range(label_0.shape[0]):
                    for j in range(label_0.shape[1]):
                        if label_0[i, j]== 4:
                            label_new_1[i, j] = 1
                        elif label_0[i, j]== 5:
                            label_new_1[i, j] = 2
                        elif label_0[i, j]== 6:
                            label_new_1[i, j] = 3
                        else:
                            pass
                label_new_1 = cv.flip(label_new_1, 0, dst=None)
                # label_left = label_new_1[label_new_1.shape[0] // 2:label_new_1.shape[0] // 2 + target_patch,
                #              label_new_1.shape[1] // 2 - target_patch:label_new_1.shape[1] // 2]
                # label_right = label_new_1[label_0.shape[0] // 2:label_new_1.shape[0] // 2 + target_patch ,
                #               label_new_1.shape[1] // 2:label_new_1.shape[1] // 2 + target_patch]
                # label_left_to_right = cv.flip(label_left, 1, dst=None)

                # label_right = np.reshape(label_right, (1, label_right.shape[0], label_right.shape[1], 1))
                # label_left_to_right = np.reshape(label_left_to_right,
                #                                  (1, label_left_to_right.shape[0], label_left_to_right.shape[1], 1))
                # label_right_all[m, :, :, 0:1] = label_right
                # label_left_to_right_all[m, :, :, 0:1] = label_left_to_right

                label_all_0 = label_new_1[label_new_1.shape[0] // 2 - 15:label_new_1.shape[0] // 2 + target_patch -15,
                             label_new_1.shape[1] // 2 - target_patch//2:label_new_1.shape[1] // 2 + target_patch//2]
                label_all_0 = np.reshape(label_all_0, (1, label_all_0.shape[0], label_all_0.shape[1], 1))
                label_all[m, :, :, 0:1] = label_all_0
            return label_all


def get_target_himage(sub_path):
    for filename_1 in os.listdir(sub_path):
        if 'M.nii' in filename_1 and '.gz' not in filename_1:
            sub_sub_path_2 = os.path.join(sub_path, filename_1)
            image = sitk.ReadImage(sub_sub_path_2)
            image_array = sitk.GetArrayFromImage(image)
            transfored_image = image_array.transpose((0, 2, 1))
            image_right_all = np.zeros((transfored_image.shape[2], target_patch, target_patch, 1))
            image_left_to_right_all = np.zeros((transfored_image.shape[2], target_patch, target_patch, 1))
            image_all = np.zeros((transfored_image.shape[2], target_patch, target_patch, 1))
            for j in range(transfored_image.shape[2]):
                image_trans = transfored_image[:, :, j]
                image_trans = cv.flip(image_trans, 0, dst=None)
                # image_left = image_trans[image_trans.shape[0] // 2 :image_trans.shape[0] // 2 + target_patch,
                #              image_trans.shape[1] // 2 - target_patch:image_trans.shape[1] // 2]
                # image_right = image_trans[image_trans.shape[0] // 2:image_trans.shape[0] // 2 + target_patch,
                #               image_trans.shape[1] // 2:image_trans.shape[1] // 2 + target_patch]
                # image_left_to_right = cv.flip(image_left, 1, dst=None)
                #
                # image_right = np.reshape(image_right, (1, image_right.shape[0], image_right.shape[1], 1))
                # image_left_to_right = np.reshape(image_left_to_right,
                #                                  (1, image_left_to_right.shape[0], image_left_to_right.shape[1], 1))
                #
                # image_right_all[j, :, :, 0:1] = image_right
                # image_left_to_right_all[j, :, :, 0:1] = image_left_to_right
                image_all_0 = image_trans[image_trans.shape[0] // 2 -15:image_trans.shape[0] // 2 + target_patch - 15,
                              image_trans.shape[1] // 2 - target_patch // 2:image_trans.shape[
                                                                                1] // 2 + target_patch // 2]
                image_all_0 = np.reshape(image_all_0, (1, image_all_0.shape[0], image_all_0.shape[1], 1))
                image_all[j, :, :, 0:1] = image_all_0
            return image_all



def right_data(k, no_label_number,transfored_image,label_array):
    label_array_crop_r = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 1))
    transfored_image_crop_r = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 3))
    for i in range(no_label_number):
        j = random.randint(k, k+20)

        label_array_crop_r = np.concatenate((label_array_crop_r,label_array[j:j + 1, :, :, :]), axis=0)

        image_trans = np.transpose(transfored_image[j - 1:j + 2, :, :, :], (3, 1, 2, 0))
        transfored_image_crop_r = np.concatenate((transfored_image_crop_r, image_trans), axis=0)

    return transfored_image_crop_r, label_array_crop_r

def left_data(k, no_label_number,transfored_image, label_array):
    transfored_image_crop_l = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 3))
    label_array_crop_l = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 1))
    for i in range(no_label_number):
        j = random.randint(k-20, k)

        label_array_crop_l = np.concatenate((label_array_crop_l, label_array[j:j + 1, :, :, :]), axis=0)

        image_trans = np.transpose(transfored_image[j - 1:j + 2, :, :, :],(3,1,2,0))
        transfored_image_crop_l = np.concatenate((transfored_image_crop_l, image_trans), axis=0)

    return transfored_image_crop_l,label_array_crop_l

def get_train_hdata(transfored_image, label_array, no_label_number):
    label_array_crop_l = None
    transfored_image_crop_l = None
    label_array_crop_r = None
    transfored_image_crop_r = None

    label_array_crop_h_all = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 1))
    transfored_image_crop_h_all = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 3))

    for k in range(1, label_array.shape[0] - 1):

        if np.sum(label_array[k - 1, :, :, :]) == 0 and np.sum(label_array[k, :, :, :]) != 0:
            transfored_image_crop_l, label_array_crop_l = left_data(k, no_label_number, transfored_image, label_array)

        elif np.sum(label_array[k - 1, :, :, :]) != 0 and np.sum(label_array[k, :, :, :]) == 0:
            transfored_image_crop_r, label_array_crop_r = right_data(k, no_label_number, transfored_image, label_array)

        elif np.sum(label_array[k, :, :, :]) != 0:

            label_array_crop_h_all = np.concatenate((label_array_crop_h_all, label_array[k:k + 1, :, :, :]), axis=0)

            image_trans_h = np.transpose(transfored_image[k - 1:k + 2, :, :, :], (3, 1, 2, 0))
            transfored_image_crop_h_all = np.concatenate((transfored_image_crop_h_all, image_trans_h), axis=0)
        else:
            pass
    label_0 = np.concatenate((label_array_crop_l, label_array_crop_h_all, label_array_crop_r), axis=0)
    image_0 = np.concatenate((transfored_image_crop_l, transfored_image_crop_h_all, transfored_image_crop_r), axis=0)
    return image_0, label_0

def get_3d_data(transfored_image,label_array):
    label_array_crop_h_all = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 1))
    transfored_image_crop_h_all = np.zeros((0, transfored_image.shape[1], transfored_image.shape[2], 3))

    for k in range(1, transfored_image.shape[0] - 2):
        label_array_crop_h_all = np.concatenate((label_array_crop_h_all, label_array[k:k + 1, :, :, :]), axis=0)

        image_trans_h = np.transpose(transfored_image[k - 1:k + 2, :, :, :], (3, 1, 2, 0))
        transfored_image_crop_h_all = np.concatenate((transfored_image_crop_h_all, image_trans_h), axis=0)
    return transfored_image_crop_h_all, label_array_crop_h_all


def crop_data_to_target(train_image, train_label, target_patch):
    train_image_crop = train_image[:, train_image.shape[1]//2 - target_patch//2:train_image.shape[1]//2 + target_patch//2,
                       train_image.shape[2]//2 - target_patch//2:train_image.shape[2]//2 + target_patch//2, :]
    train_label_crop = train_label[:, train_label.shape[1]//2 - target_patch//2:train_label.shape[1]//2 + target_patch//2,
                       train_label.shape[2]//2 - target_patch//2:train_label.shape[2]//2 + target_patch//2, :]
    return train_image_crop, train_label_crop


def hlabel_to_onehot(label):
    label_all = np.zeros((0, label.shape[1], label.shape[2], 4))
    # print(label.shape)
    for m in range(label.shape[0]):
        label_one = label[m:m + 1,:, :, :]
        label_b = None
        label_1 = np.zeros((1, label_one.shape[1], label_one.shape[2], 1))
        label_2 = np.zeros((1, label_one.shape[1], label_one.shape[2], 1))
        label_3 = np.zeros((1, label_one.shape[1], label_one.shape[2], 1))
        for i in range(label_one.shape[1]):
            for j in range(label_one.shape[2]):
                if label_one[0, i, j, 0] == 1:
                    label_1[0, i, j, 0] = 1
                elif label_one[0, i, j, 0] == 2:
                    label_2[0, i, j, 0] = 1
                elif label_one[0, i, j, 0] == 3:
                    label_3[0, i, j, 0] = 1
                else:
                    label_b = 1 - label_1 - label_2 -label_3
        label_c = np.concatenate((label_b, label_1, label_2, label_3), axis=3)
        label_all = np.concatenate((label_all, label_c), axis=0)
    return label_all


def input_hdata_to_h5py(h_path):
    count = 0
    train_count = 0
    for sub_name in os.listdir(h_path):
        sub_path = os.path.join(h_path, sub_name)
        # print(sub_path)
        count += 1
        print('count')
        # print(count)

        if count in range(1,18) or count in range(39,42):
        # if count in range(18, 47):
            train_count += 1
            print('train_count')
            print(train_count)
            label_right_all = get_target_hlabel(sub_path)
            image_right_all = get_target_himage(sub_path)

            transfored_image_r, label_array_r = get_train_hdata(image_right_all, label_right_all, no_label_number=5)
            # transfored_image_l_to_r, label_array_l_to_r = get_train_hdata(image_left_to_right_all, label_left_to_right_all, no_label_number=5)

            onehot_label_r = hlabel_to_onehot(label_array_r)
            # onehot_label_l_to_r = hlabel_to_onehot(label_array_l_to_r)

            print('train_set')
            print('r')
            print(transfored_image_r.shape)
            print(onehot_label_r.shape)
            # print('l_to_r')
            # print(label_array_l_to_r.shape)
            # print(onehot_label_l_to_r.shape)

            # file_train = h5py.File(dir1 + 'train_original2_h_3D_' + str(train_count) + '.h5', 'w')
            # file_train.create_dataset('train_data', data=transfored_image_r)
            # file_train.create_dataset('train_label', data=onehot_label_r)
            # file_train.close()


        elif count in range(42,47):
            label_right_all = get_target_hlabel(sub_path)
            image_right_all = get_target_himage(sub_path)

            transfored_image_r, label_array_r = get_train_hdata(image_right_all, label_right_all, no_label_number=20)
            # transfored_image_l_to_r, label_array_l_to_r = get_train_hdata(image_left_to_right_all,
            #                                                               label_left_to_right_all, no_label_number=20)

            onehot_label_r = hlabel_to_onehot(label_array_r)
            # onehot_label_l_to_r = hlabel_to_onehot(label_array_l_to_r)

            print('validation_set')
            print('r')
            print(transfored_image_r.shape)
            print(onehot_label_r.shape)
            # print('l_to_r')
            # print(label_array_l_to_r.shape)
            # print(onehot_label_l_to_r.shape)

            # file_train = h5py.File(dir1 + 'validation_original2_ht_3D_' + str(count-41) + '.h5', 'w')
            # file_train.create_dataset('vali_data', data=transfored_image_r)
            # file_train.create_dataset('vali_label', data=onehot_label_r)
            # file_train.close()

            # file_train = h5py.File(dir1 + 'validation_original_ht_3D_' + str(count + 2) + '.h5', 'w')
            # file_train.create_dataset('vali_data', data=transfored_image_l_to_r)
            # file_train.create_dataset('vali_label', data=onehot_label_l_to_r)
            # file_train.close()

        elif count in range(18,39):
            print(sub_path)
            print(count)
            label_right_all = get_target_hlabel(sub_path)
            image_right_all = get_target_himage(sub_path)

            transfored_image_r, label_array_r = get_3d_data(image_right_all, label_right_all)
            # transfored_image_l_to_r, label_array_l_to_r = get_3d_data(image_left_to_right_all, label_left_to_right_all)

            onehot_label_r = hlabel_to_onehot(label_array_r)
            # onehot_label_l_to_r = hlabel_to_onehot(label_array_r)
            print('test_set')
            print('r')
            print(transfored_image_r.shape)
            print(onehot_label_r.shape)
            # print('l_to_r')
            # print(label_array_l_to_r.shape)
            # print(onehot_label_l_to_r.shape)
            save_dir =dir1 + 'test2_original2_h_3D_' + str(count-17) + '.h5'
            print('save_dir')
            print(save_dir)
            file_train = h5py.File(dir1 + 'test2_original2_h_3D_' + str(count-17) + '.h5', 'w')
            file_train.create_dataset('test_data', data=transfored_image_r)
            file_train.create_dataset('test_label', data=onehot_label_r)
            file_train.close()

            # file_train = h5py.File(dir1 + 'test_original_h_3D_' + str(count + 21) + '.h5', 'w')
            # file_train.create_dataset('test_data', data=transfored_image_l_to_r)
            # file_train.create_dataset('test_label', data=onehot_label_l_to_r)
            # file_train.close()

        else:
            pass


def test_data_inputs(inputs_path):
    count = 0
    for sub_name in os.listdir(inputs_path):
        sub_path = os.path.join(inputs_path, sub_name)
        count += 1
        print('count')
        if count in range(11,12):
            print(count)
            print(sub_path)
            label_right_all = get_target_hlabel(sub_path)
            image_right_all = get_target_himage(sub_path)
            transfored_image_r, label_array_r = get_3d_data(image_right_all, label_right_all)
            onehot_label_r = hlabel_to_onehot(label_array_r)
            print('test_set')
            print('r')
            print(transfored_image_r.shape)
            print(onehot_label_r.shape)
            save_dir = dir1 + 'test2_original2_h_3D_' + str(count+21) + '.h5'
            print('save_dir')
            print(save_dir)
            file_train = h5py.File(dir1 + 'test2_original2_h_3D_' + str(count+21) + '.h5', 'w')
            file_train.create_dataset('test_data', data=transfored_image_r)
            file_train.create_dataset('test_label', data=onehot_label_r)
            file_train.close()

test_data_inputs(inputs_path)
# mean_save, std_save = meanstd(d_path)
# data_list = [mean_save, std_save]
# data_array = np.array(data_list)
# np.savetxt('datanorm.txt', data_array)

def main():
    vali_data = np.zeros((0, target_patch,target_patch, 3))
    vali_label = np.zeros((0, target_patch,target_patch, 4))
    train_data = np.zeros((0, target_patch,target_patch, 3))
    train_label = np.zeros((0, target_patch,target_patch, 4))
    # input_hdata_to_h5py(h_path)
    for j in range(1,21):
        # print(i)
        file1 = h5py.File(dir1 + 'train_original2_h_3D_' + str(j) + '.h5', 'r')
        train_image_0 = file1['train_data'][:]
        train_label_0 = file1['train_label'][:]
        file1.close()
        print(train_image_0.shape)
        print(train_label_0.shape)
        # for i in range(train_label_0.shape[0]):
        #     if np.sum(train_label_0[i, :, :, 3]) != 0:
        #         plt.imshow(train_image_0[i, :, :, 0], cmap='gray')
        #         # plt.contour(train_label_0[i, :, :, 1], linewidths=0.1, cmap=ListedColormap('red'))
        #         # plt.contour(train_label_0[i, :, :, 2], linewidths=0.1, cmap=ListedColormap('red'))
        #         # plt.contour(train_label_0[i, :, :, 3], linewidths=0.1, cmap=ListedColormap('red'))
        #         plt.show()
        #         plt.imshow(train_label_0[i, :, :, 0], cmap='gray')
        #         plt.show()
        #         plt.imshow(train_label_0[i, :, :, 1], cmap='gray')
        #         plt.show()
        #         plt.imshow(train_label_0[i, :, :, 2], cmap='gray')
        #         plt.show()
        #         plt.imshow(train_label_0[i, :, :, 3], cmap='gray')
        #         plt.show()

        train_data = np.concatenate((train_data, train_image_0),axis=0)
        train_label = np.concatenate((train_label, train_label_0), axis=0)
    print(train_data.shape)
    print(train_label.shape)
    file_train = h5py.File(dir1 + 'train_all2_h_3D_all_' + str(0) + '.h5', 'w')
    file_train.create_dataset('train_data', data=train_data)
    file_train.create_dataset('train_label', data=train_label)
    file_train.close()
    for k in range(1,6):
        # print(i)
        file1 = h5py.File(dir1 + 'validation_original2_ht_3D_' + str(k) + '.h5', 'r')
        train_image = file1['vali_data'][:]
        train_label = file1['vali_label'][:]
        file1.close()

        vali_data = np.concatenate((vali_data, train_image),axis=0)
        vali_label = np.concatenate((vali_label, train_label), axis=0)
    print(vali_data.shape)
    print(vali_label.shape)
    file_train = h5py.File(dir1 + 'validation_all2_ht_3D_' + str(0) + '.h5', 'w')
    file_train.create_dataset('vali_data', data=vali_data)
    file_train.create_dataset('vali_label', data=vali_label)
    file_train.close()


# main()
