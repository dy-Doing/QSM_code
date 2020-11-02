from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import ndimage
import h5py
import SimpleITK as sitk
import os
import re
import cv2 as cv
from skimage import draw, transform, io, feature
from matplotlib.colors import ListedColormap
from MeDIT.Visualization import Imshow3D, Imshow3DArray

# tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import load_model
from losses import tversky_loss
# from inputs import get_target_hlabel, get_target_himage, get_3d_data, hlabel_to_onehot

def crop_data(data_0, label_0):
    data_0 = data_0[:, (patch_row1 // 2 - patch_row // 2):(patch_row1 // 2 + patch_row // 2),
             (patch_col1 // 2 - patch_col // 2):(patch_col1 // 2 + patch_col // 2), :]
    label_0 = label_0[:, (patch_row1 // 2 - patch_row // 2):(patch_row1 // 2 + patch_row // 2),
              (patch_col1 // 2 - patch_col // 2):(patch_col1 // 2 + patch_col // 2), :]

    for j in range(data_0.shape[0]):
       data_0[j, :, :, :] = (data_0[j, :, :, :] - np.mean(data_0[j, :, :, :])) / np.std(data_0[j, :, :, :])
    return data_0, label_0

def Dice(y_, y):
    y1 = np.reshape(y, [-1, 1])
    y2 = np.reshape(y_, [-1, 1])
    return 2 * np.sum(y1 * y2) / (np.sum(y1) + np.sum(y2) + 0.00001)
def Dice_train(y, y_):
    y1 = tf.reshape(y, [-1, 1])
    y2 = tf.reshape(y_, [-1, 1])
    return 2 * tf.reduce_sum(y1 * y2) / (tf.reduce_sum(y1) + tf.reduce_sum(y2) + 0.00001)

def VOE(y_, y):
    y1 = np.reshape(y, [-1, 1])
    y2 = np.reshape(y_, [-1, 1])
    return 1-np.sum(y1 * y2)/(np.sum(y1) + np.sum(y2)-np.sum(y1 * y2) + 0.00001)

def IOU(y_, y):
    y1 = np.reshape(y, [-1, 1])
    y2 = np.reshape(y_, [-1, 1])
    return np.sum(y1 * y2) / (np.sum(y1) + np.sum(y2) - np.sum(y1 * y2) + 0.00001)

def RVD(y_, y):
    y1 = np.reshape(y, [-1, 1])
    y2 = np.reshape(y_, [-1, 1])
    return np.abs(np.sum(y1)/(np.sum(y2) + 0.00001)-1)

def Precision(y_, y):
    y1 = np.reshape(y, [-1, 1])
    y2 = np.reshape(y_, [-1, 1])
    return np.sum(y1*y2)/(np.sum(y1)+0.00001)
def Recall(y_, y):
    y1 = np.reshape(y, [-1, 1])
    y2 = np.reshape(y_, [-1, 1])
    return np.sum(y1*y2)/(np.sum(y2)+0.00001)

def RemoveSmallRegion(mask):
    label_im, nb_labels = ndimage.label(mask)
    p = []
    for i in range(1, nb_labels+1):
        c = (label_im == i).sum()
        p.append(c)
    size_thres_all = sorted(p)
    # print(size_thres_all)
    size_thres = 100000
    if len(size_thres_all) == 0:
        pass
    elif len(size_thres_all) == 1:
        size_thres = max(size_thres_all)
    elif len(size_thres_all) > 1:
        del(size_thres_all[-1])
        size_thres = max(size_thres_all)
    for i in range(1, nb_labels + 1):
        if (label_im == i).sum() < size_thres:
            # remove the small ROI in mask
            mask[label_im == i] = 0
    return mask

def write_output_as_nii(test_path, y1_out, y2_out, y3_out, pad_number_1 = 128//2, pad_number_2 = 88):
    y_out = y1_out*4 + y2_out*5 + y3_out*6
    y1_out_all_1 = np.lib.pad(y_out,((pad_number_1+33,pad_number_1-33),(pad_number_2, pad_number_2), (1,2)),'constant',constant_values=0)
    y1_out_all_2 = np.zeros((y1_out_all_1.shape[0], y1_out_all_1.shape[1], y1_out_all_1.shape[2]))
    for o in range(y1_out_all_1.shape[2]):
         y1_out_all_1_f = cv.flip(y1_out_all_1[:, :, o], 0, dst=None)
         y1_out_all_2[:, :, o] = y1_out_all_1_f
    y1_out_all = np.transpose(y1_out_all_2,(0,2,1))
    y1_out_nii = sitk.GetImageFromArray(y1_out_all)
    count_0 = 0
    nums = []
    for sub_name in os.listdir(test_path):
        num = find_numfromstr(sub_name)
        nums.append(num)
    print(nums)
    nums = sorted(nums)
    nums_s = nums[16:]
    print(nums_s[j])
    for num in nums_s:
        count_0 += 1
        if num <=22:
           count = num - 17
        else:
            count = num - 19
        # print(nums_[j-1])
        if count == j:
            print('count')
            print(count)

            nums_s = str(nums_s[j])
            save_dir = nums_s.zfill(2) + '_mag_pha_results_PDF_MEDI_lambda1000'

            sub_path = save_cnn + '/' + save_dir
            os.mkdir(sub_path)
            # for filename_1 in os.listdir(sub_path):
            #     if 'M.nii' in filename_1:
            #         sub_sub_path_1 = os.path.join(sub_path, filename_1)
            #         print(sub_sub_path_1)
            #         image = sitk.ReadImage(sub_sub_path_1)
            #         image_array = sitk.GetArrayFromImage(image)
            #
            #         filename_label = 'Untitled.nii.gz'
            #         sub_sub_path_2 = os.path.join(sub_path, filename_label)
            #         label = sitk.ReadImage(sub_sub_path_2)
            #         label_array = sitk.GetArrayFromImage(label)
            #         print('image_array')
            #         print(image_array.shape)
            #         print(label_array.shape)
            #         print(y1_out_all.shape)
            #         for o in range(y1_out_all.shape[1]):
            #             if np.sum(y1_out_all[:, o, :]) != 0:
            #                 plt.imshow(image_array[:, o, :], cmap='gray')
            #                 plt.contour(label_array[:, o, :], linewidths=0.19, cmap=ListedColormap('red'))
            #                 plt.contour(y1_out_all[:, o, :], linewidths=0.19, cmap=ListedColormap('green'))
            #                 plt.show()
            path = sub_path + '/CNN_h_out.nii.gz'
            print(path)
            sitk.WriteImage(y1_out_nii, path)
def write_output_as_nii_1(test_path, y1_out, y2_out, y3_out, pad_number_1 = 128//2, pad_number_2 = 88):
    y_out = y1_out*4 + y2_out*5 + y3_out*6
    y1_out_all_1 = np.lib.pad(y_out,((pad_number_1+33,pad_number_1-33),(pad_number_2, pad_number_2), (1,2)),'constant',constant_values=0)
    y1_out_all_2 = np.zeros((y1_out_all_1.shape[0], y1_out_all_1.shape[1], y1_out_all_1.shape[2]))
    for o in range(y1_out_all_1.shape[2]):
         y1_out_all_1_f = cv.flip(y1_out_all_1[:, :, o], 0, dst=None)
         y1_out_all_2[:, :, o] = y1_out_all_1_f
    y1_out_all = np.transpose(y1_out_all_2,(0,2,1))
    y1_out_nii = sitk.GetImageFromArray(y1_out_all)
    nums = []
    for sub_name in os.listdir(test_path):
        num = find_numfromstr(sub_name)
        nums.append(num)
    # print(nums)
    nums = sorted(nums)
    print(nums)
    for num1 in nums:
        count = num1 - 27
        # print(count)
        if count == j:
            print('count')
            print(count)
            print(num1)
            nums_s = str(num1)
            save_dir = nums_s.zfill(2) + '_mag_pha_results_PDF_MEDI_lambda1000'

            sub_path = save_cnn + '/' + save_dir
            os.mkdir(sub_path)
            # for filename_1 in os.listdir(test_path):
            #     sub_sub_path1 = os.path.join(test_path, filename_1)
            #     for filename_2 in os.listdir(sub_sub_path1):
            #         sub_sub_path = os.path.join(sub_sub_path1, filename_2)
            #         if 'M.nii' in sub_sub_path:
            #             sub_sub_path_1 = sub_sub_path
            #             print(sub_sub_path_1)
            #             image = sitk.ReadImage(sub_sub_path_1)
            #             image_array = sitk.GetArrayFromImage(image)
            #
            #             filename_label = 'Untitled.nii.gz'
            #             sub_sub_path_2 = os.path.join(sub_sub_path1, filename_label)
            #             print(sub_sub_path_2)
            #             label = sitk.ReadImage(sub_sub_path_2)
            #             label_array = sitk.GetArrayFromImage(label)
            #             print('image_array')
            #             print(image_array.shape)
            #             print(label_array.shape)
            #             print(y1_out_all.shape)
            #             for o in range(y1_out_all.shape[1]):
            #                 if np.sum(y1_out_all[:, o, :]) != 0:
            #                     plt.imshow(image_array[:, o, :], cmap='gray')
            #                     plt.contour(label_array[:, o, :], linewidths=0.19, cmap=ListedColormap('red'))
            #                     plt.contour(y1_out_all[:, o, :], linewidths=0.19, cmap=ListedColormap('green'))
            #                     plt.show()
            path = sub_path + '/CNN_h_out.nii.gz'
            print(path)
            sitk.WriteImage(y1_out_nii, path)

def plot_hist_of_dice(dice_all1):
     n, bins, patchesm = plt.hist(dice_all1, bins=50, facecolor='blue', alpha=0.5, edgecolor='black')
     # # y = mlab.normpdf(bins, np.sum(dice_all1)/21,np.std(dice_all1))
     # # plt.plot(bins, y, 'r--')
     font2 = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': 20}
     font1 = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': 14}
     plt.xlabel('Dice', font1)
     plt.ylabel('number', font1)
     plt.title(r'Dice of test set', font2)
     plt.subplots_adjust(left=0.15)
     plt.show()
def find_numfromstr(str):
    f = re.findall('(\d+)', str)
    num = int(f[0])
    return num
def show_output(y1_in,y1_out,y2_in,y2_out, y3_in,y3_out,y_image):
    n = y_image.shape[2]
    for h in range(n):
        if np.sum(label_test[h, :, :, 0]) != label_test.shape[1]*label_test.shape[2] \
                or np.sum(y1_out[:, :, h]) != 0 or np.sum(y2_out[:, :, h]) != 0:
            # plt.imshow(label_test[h, :, :, 0], cmap='gray')
            # plt.show()
            # plt.imshow(y1_out[:, :, h], cmap='gray')
            # plt.show()
            print(h)
            dice_slice1 = Dice(y1_in[:, :, h], y1_out[:, :, h])
            dice_slice2 = Dice(y2_in[:, :, h], y2_out[:, :, h])
            dice_slice3 = Dice(y3_in[:, :, h], y3_out[:, :, h])
            print(dice_slice1)
            print(dice_slice2)
            print(dice_slice3)
            plt.figure('p', figsize=(12, 12))
            plt.subplot(121)
            plt.imshow(y_image[:, :, h], cmap='gray', vmin=-1, vmax=5)
            plt.contour(y1_in[:, :, h], linewidths=0.19, cmap=ListedColormap('red'))
            plt.contour(y2_in[:, :, h], linewidths=0.19, cmap=ListedColormap('green'))
            plt.contour(y3_in[:, :, h], linewidths=0.19, cmap=ListedColormap('yellow'))
            plt.subplot(122)
            plt.imshow(y_image[:, :, h], cmap='gray', vmin=-1, vmax=5)
            plt.contour(y1_out[:, :, h], linewidths=0.19, cmap=ListedColormap('red'))
            plt.contour(y2_out[:, :, h], linewidths=0.19, cmap=ListedColormap('green'))
            plt.contour(y3_out[:, :, h], linewidths=0.19, cmap=ListedColormap('yellow'))
            plt.show()
    # y_image = (y_image - np.min(y_image))/(np.max(y_image)-np.min(y_image))
    # show2 = Imshow3DArray(y_image, ROI=[y1_in,y1_out])



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

test_path = r'//ALG-cloud2/Incoming/ZFF/haacke/test_set'
patch_row1 = 96
patch_col1 = 96
patch_row = 64
patch_col = 64
m = 64
dice_all_1 = []
dice_all_2 = []
dice_all_3 = []
VOE_all_1 = []
IOU_all_1 = []
RVD_all_1 = []
precision_all_1 = []
recall_all_1 = []
VOE_all_2 = []
IOU_all_2 = []
RVD_all_2 = []
precision_all_2 = []
recall_all_2 = []
VOE_all_3 = []
IOU_all_3 = []
RVD_all_3 = []
precision_all_3 = []
recall_all_3 = []
y1_all = np.zeros((0, m, m, 1))
y2_all = np.zeros((0, m, m, 1))
y3_all = np.zeros((0, m, m, 1))
output_all =[]

h_path_1 = r'D:/htfg_all/QSMseg_0305'
h_path = r'D:/htfg_all/RESULT1_2'
save_cnn =r'D:/htfg_all/CNN_out_all/CNN_out_6_1'
log_dir = r'C:/Users/ZHOUFF/PycharmProjects/htfg_tf2/log_all/log_attUnet/att_Unet32_4_2/'
model_dir = r'C:/Users/ZHOUFF/PycharmProjects/htfg_tf2/model_all/model_attUnet/att_Unet16_4_1.h5'
log_dir_transfer = r'C:/Users/ZHOUFF/PycharmProjects/htfg_tf2/log_all/log_attUnet_transfer/att_Unet16_4_1/'
model_dir_transfer = r'C:/Users/ZHOUFF/PycharmProjects/htfg_tf2/model_all/model_attUnet_transfer/att_Unet16_4_6.h5'
# model = load_model(log_dir + 'ep067-loss0.009-val_loss0.005.h5', custom_objects={'tversky_loss': tversky_loss,'Dice': Dice_train})
model = load_model(model_dir_transfer, custom_objects={'tversky_loss': tversky_loss,'Dice': Dice_train})
model.summary()

for j in np.arange(1, 22):
        # x=j
    if j == 6:
        pass
        #   x = j+1
        # else:
        #     pass
    else:
        print(j)
    #     print(x)
        file = h5py.File(r'C:/Users/ZHOUFF/PycharmProjects/htfg/data/h_h5/test2_original2_h_3D_' + str(j) + '.h5', 'r')
        # file1 = h5py.File(r'C:/Users/ZHOUFF/PycharmProjects/htfg/data/h_h5/test_original2_h_3D_dice_' + str(j+17) + '.h5', 'r')
        # label_test = file1['test_label_L'][:]
        data_test_0 = file['test_data'][:]
        label_test_0 = file['test_label'][:]
        file.close()
        # file1.close()
        data_test, label_test = crop_data(data_test_0, label_test_0)
        # for h in range(label_test.shape[0]):
        #     if np.sum(label_test[h, :, :, 0]) != label_test.shape[1]*label_test.shape[2]:
        #         plt.imshow(data_test[h,:, :, 1], cmap='gray',vmin=-4,vmax=8)
        #         plt.contour(label_test[h, :, :, 1], linewidths=0.19, cmap=ListedColormap('red'))
        #         plt.contour(label_test[h, :, :, 2], linewidths=0.19, cmap=ListedColormap('green'))
        #         plt.contour(label_test[h, :, :, 3], linewidths=0.19, cmap=ListedColormap('yellow'))
        #         plt.show()
        n = np.size(data_test, 0)
        y1_in_data = []
        y1_out_data = []
        y2_in_data = []
        y2_out_data = []
        y3_in_data = []
        y3_out_data = []
        output = np.zeros([n, m, m, 4])
        for i in np.arange(n):
            inputs = data_test[i, :, :, :]
            label = label_test[i, :, :, :]
            inputs_1 = np.reshape(inputs, [1, m, m, 3])
            recon, out8 = model.predict(inputs_1)
            # print(recon.shape)
            recon = np.reshape(recon, [m, m, 4])
            output[i, :, :, :] = recon
        y1_out = np.asarray(output[:, :, :, 1] > 0.5, dtype=np.uint8)
        y2_out = np.asarray(output[:, :, :, 2] > 0.5, dtype=np.uint8)
        y3_out = np.asarray(output[:, :, :, 3] > 0.5, dtype=np.uint8)
        y1_out = np.transpose(y1_out, (1, 2, 0))
        y2_out = np.transpose(y2_out, (1, 2, 0))
        y3_out = np.transpose(y3_out, (1, 2, 0))
        y1_out = RemoveSmallRegion(y1_out)
        y2_out = RemoveSmallRegion(y2_out)
        y3_out = RemoveSmallRegion(y3_out)

        # write_output_as_nii_1(h_path_1, y1_out, y2_out, y3_out)
        write_output_as_nii(h_path, y1_out, y2_out, y3_out)

#         y1_in = label_test[:, :, :, 1]
#         y2_in = label_test[:, :, :, 2]
#         y3_in = label_test[:, :, :, 3]
#         y1_in = np.transpose(y1_in, (1, 2, 0))
#         y2_in = np.transpose(y2_in, (1, 2, 0))
#         y3_in = np.transpose(y3_in, (1, 2, 0))
#
#         y_image = data_test[:, :, :, 1]
#         y_image = np.transpose(y_image, (1, 2, 0))
#         # show_output(y1_in,y1_out,y2_in,y2_out, y3_in,y3_out,y_image)
#         for k in np.arange(n):
#             y1_in_data.append(y1_in[:, :, k])
#             y1_out_data.append(y1_out[:, :, k])
#
#             y2_in_data.append(y2_in[:, :, k])
#             y2_out_data.append(y2_out[:, :, k])
#
#             y3_in_data.append(y3_in[:, :, k])
#             y3_out_data.append(y3_out[:, :, k])
#
#         dice1 = Dice(y1_in_data, y1_out_data)
#         dice2 = Dice(y2_in_data, y2_out_data)
#         dice3 = Dice(y3_in_data, y3_out_data)
# #
# #         Voe1 = VOE(y1_in_data, y1_out_data)
# #         Rvd1 = RVD(y1_in_data, y1_out_data)
# #         precision1 = Precision(y1_in_data, y1_out_data)
# #         recall1 = Recall(y1_in_data, y1_out_data)
# #         Iou1 = IOU(y1_in_data, y1_out_data)
# #
# #         Voe2 = VOE(y2_in_data, y2_out_data)
# #         Rvd2 = RVD(y2_in_data, y2_out_data)
# #         precision2 = Precision(y2_in_data, y2_out_data)
# #         recall2 = Recall(y2_in_data, y2_out_data)
# #         Iou2 = IOU(y2_in_data, y2_out_data)
# #
# #         Voe3 = VOE(y3_in_data, y3_out_data)
# #         Rvd3 = RVD(y3_in_data, y3_out_data)
# #         precision3 = Precision(y3_in_data, y3_out_data)
# #         recall3 = Recall(y3_in_data, y3_out_data)
# #         Iou3 = IOU(y3_in_data, y3_out_data)
#         print('Dice1=%f' % (dice1))
#         print('Dice2=%f' % (dice2))
#         print('Dice3=%f' % (dice3))
# #         # print('VOE=%f'%(Voe))
# #         # print('RVD=%f'%(Rvd))
# #         # print('Precision=%f' % (precision))
# #         # print('Recall=%f' % (recall))
# #         # print('Iou=%f' % (Iou))
#         dice_all_1.append(dice1)
#         dice_all_2.append(dice2)
#         dice_all_3.append(dice3)
# #
# #         VOE_all_1.append(Voe1)
# #         RVD_all_1.append(Rvd1)
# #         precision_all_1.append(precision1)
# #         recall_all_1.append(recall1)
# #         IOU_all_1.append(Iou1)
# #
# #         VOE_all_2.append(Voe2)
# #         RVD_all_2.append(Rvd2)
# #         precision_all_2.append(precision2)
# #         recall_all_2.append(recall2)
# #         IOU_all_2.append(Iou2)
# #
# #         VOE_all_3.append(Voe3)
# #         RVD_all_3.append(Rvd3)
# #         precision_all_3.append(precision3)
# #         recall_all_3.append(recall3)
# #         IOU_all_3.append(Iou3)
# #
# dice_all1 = np.reshape(dice_all_1, [-1, 1])
# std_dice_1 = np.std(dice_all1)
# print('Dice_std1=%f' % (std_dice_1))
# print((np.sum(dice_all1) / 36))
# # plot_hist_of_dice(dice_all1)
# dice_all2 = np.reshape(dice_all_2, [-1, 1])
# std_dice_2 = np.std(dice_all2)
# print('Dice_std2=%f' % (std_dice_2))
# print((np.sum(dice_all2) / 36))
# dice_all3 = np.reshape(dice_all_3, [-1, 1])
# std_dice_3 = np.std(dice_all3)
# print('Dice_std3=%f' % (std_dice_3))
# print((np.sum(dice_all3) / 36))
# # plot_hist_of_dice(dice_all2)
# # plot_hist_of_dice(dice_all3)
# VOE_all1 = np.reshape(VOE_all_1, [-1, 1])
# std_voe_1 = np.std(VOE_all1)
# print('VOE_std1=%f' % (std_voe_1))
# print(np.sum(VOE_all1) / 36)
# RVD_all1 = np.reshape(RVD_all_1, [-1, 1])
# std_rvd_1 = np.std(RVD_all1)
# print('RVD_std1=%f' % (std_rvd_1))
# print(np.sum(RVD_all1) / 36)
# precision_all1 = np.reshape(precision_all_1, [-1, 1])
# std_pre_1 = np.std(precision_all1)
# print('precision_std1=%f' % (std_pre_1))
# print(np.sum(precision_all1) / 36)
# recall_all1 = np.reshape(recall_all_1, [-1, 1])
# std_re_1 = np.std(recall_all1)
# print('recall_std1=%f' % (std_re_1))
# print(np.sum(recall_all1) / 36)
# Iou_all1 = np.reshape(IOU_all_1, [-1, 1])
# std_iou_1 = np.std(Iou_all1)
# print('Iou_std1=%f' % (std_iou_1))
# print(np.sum(Iou_all1) / 36)
#
# VOE_all2 = np.reshape(VOE_all_2, [-1, 1])
# std_voe_2 = np.std(VOE_all2)
# print('VOE_std2=%f' % (std_voe_2))
# print(np.sum(VOE_all2) / 36)
# RVD_all2 = np.reshape(RVD_all_2, [-1, 1])
# std_rvd_2 = np.std(RVD_all2)
# print('RVD_std2=%f' % (std_rvd_2))
# print(np.sum(RVD_all2) / 36)
# precision_all2 = np.reshape(precision_all_2, [-1, 1])
# std_pre_2 = np.std(precision_all2)
# print('precision_std2=%f' % (std_pre_2))
# print(np.sum(precision_all2) / 36)
# recall_all2 = np.reshape(recall_all_2, [-1, 1])
# std_re_2 = np.std(recall_all2)
# print('recall_std2=%f' % (std_re_2))
# print(np.sum(recall_all2) / 36)
# Iou_all2 = np.reshape(IOU_all_2, [-1, 1])
# std_iou_2 = np.std(Iou_all2)
# print('Iou_std2=%f' % (std_iou_2))
# print(np.sum(Iou_all2) / 36)
#
# VOE_all3 = np.reshape(VOE_all_3, [-1, 1])
# std_voe_3 = np.std(VOE_all3)
# print('VOE_std3=%f' % (std_voe_3))
# print(np.sum(VOE_all3) / 36)
# RVD_all3 = np.reshape(RVD_all_3, [-1, 1])
# std_rvd_3 = np.std(RVD_all3)
# print('RVD_std3=%f' % (std_rvd_3))
# print(np.sum(RVD_all3) / 36)
# precision_all3 = np.reshape(precision_all_3, [-1, 1])
# std_pre_3 = np.std(precision_all3)
# print('precision_std3=%f' % (std_pre_3))
# print(np.sum(precision_all3) / 36)
# recall_all3 = np.reshape(recall_all_3, [-1, 1])
# std_re_3 = np.std(recall_all3)
# print('recall_std3=%f' % (std_re_3))
# print(np.sum(recall_all3) / 36)
# Iou_all3 = np.reshape(IOU_all_3, [-1, 1])
# std_iou_3 = np.std(Iou_all3)
# print('Iou_std3=%f' % (std_iou_3))
# print(np.sum(Iou_all3) / 36)
# import pandas as pd
# z = list(zip(dice_all_1, dice_all_2, dice_all_3,
#              VOE_all_1, VOE_all_2, VOE_all_3,
#              RVD_all_1, RVD_all_2, RVD_all_3,
#              precision_all_1, precision_all_2, precision_all_3,
#              recall_all_1, recall_all_2, recall_all_3,
#              IOU_all_1, IOU_all_2, IOU_all_3))
# name = ['RN_dice', 'SN_dice','STN_dice',
#         'RN_VOE', 'SN_VOE', 'STN_VOE',
#         'RN_RVD', 'SN_RVD','STN_RVD',
#         'RN_precision', 'SN_precision','STN_precision',
#         'RN_recall', 'SN_recall', 'STN_recall',
#         'RN_IOU', 'SN_IOU', 'STN_IOU']
# test = pd.DataFrame(columns=name, data=z)
# test.to_csv(r'D:\htfg_all\dice_list_h.csv')