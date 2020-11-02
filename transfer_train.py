import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import h5py
import os
import cv2
import keras
import pandas as pd
from keras.models import load_model
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from losses import tversky_loss
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from MeDIT.DataAugmentor import DataAugmentor3D, AugmentParametersGenerator
# from invertedNet import unet
from attention_model_d import attn_reg_d
from transfer_Net import attn_reg_t

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def Dice(y, y_):
    y1 = tf.reshape(y, [-1, 1])
    y2 = tf.reshape(y_, [-1, 1])
    return 2 * tf.reduce_sum(y1 * y2) / (tf.reduce_sum(y1) + tf.reduce_sum(y2) + 0.00001)
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
def MeanReduce(image, number):
    row, col = image.shape[0], image.shape[1]
    image_New = np.zeros((row//number, col//number))
    for i in range(row//number):
        for j in range(col//number):
            x = np.max(image[number*i:number*(i+1),number*j:(j+1)*number])
            # print(x)
            image_New[i, j] = x
    # print(image_New)
    return image_New
def inputcase2(dir1, num_str, num_end, train_batch_1, filename, data_name, label_name):
    while True:
        index_0 = [i for i in np.arange(num_str, num_end)]
        np.random.shuffle(index_0)
        index1 = index_0[0:1]

        file = h5py.File(dir1 + filename + str(index1[0]) + '.h5', 'r')
        train_image = file[data_name][:]
        train_label = file[label_name][:]
        file.close()

        index_all = [w for w in np.arange(train_label.shape[0])]
        np.random.shuffle(index_all)
        index_batch = index_all[:train_batch_1]

        image_for_train = train_image[index_batch, :, :, :]
        label_for_train = train_label[index_batch, :, :, :]

        for j in range(label_for_train.shape[0]):
            image_for_train[j, :, :, 0:3],label_for_train[j, :, :, 0:4] = \
                dataAug(image_for_train[j, :, :, :3],label_for_train[j, :, :, 0:4])
            # plt.imshow(image_for_train[j, :, :, 1],cmap='gray')
            # plt.show()
        image_for_train, label_for_train = crop_data(image_for_train, label_for_train)
        label_down_3 = np.zeros((label_for_train.shape[0], label_for_train.shape[1] // 2, label_for_train.shape[2] // 2,
                                 label_for_train.shape[3]))
        # label_down_2 = np.zeros((label_for_train.shape[0], label_for_train.shape[1] // 4, label_for_train.shape[2] // 4,
        #                          label_for_train.shape[3]))
        # label_down_1 = np.zeros((label_for_train.shape[0], label_for_train.shape[1] // 8, label_for_train.shape[2] // 8,
        #                          label_for_train.shape[3]))
        for j in range(label_for_train.shape[0]):
            label_down_3[j, :, :, 0] = cv2.resize(label_for_train[j, :, :, 0],
                                                  (label_for_train.shape[1] // 2, label_for_train.shape[2] // 2),
                                                  interpolation=cv2.INTER_CUBIC)
            label_down_3[j, :, :, 1] = cv2.resize(label_for_train[j, :, :, 1],
                                                  (label_for_train.shape[1] // 2, label_for_train.shape[2] // 2),
                                                  interpolation=cv2.INTER_CUBIC)
            label_down_3[j, :, :, 2] = cv2.resize(label_for_train[j, :, :, 2],
                                                  (label_for_train.shape[1] // 2, label_for_train.shape[2] // 2),
                                                  interpolation=cv2.INTER_CUBIC)
        #     label_down_3[j, :, :, 3] = cv2.resize(label_for_train[j, :, :, 3],
        #                                           (label_for_train.shape[1] // 2, label_for_train.shape[2] // 2),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_2[j, :, :, 0] = cv2.resize(label_for_train[j, :, :, 0],
        #                                           (label_for_train.shape[1] // 4, label_for_train.shape[2] // 4),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_2[j, :, :, 1] = cv2.resize(label_for_train[j, :, :, 1],
        #                                           (label_for_train.shape[1] // 4, label_for_train.shape[2] // 4),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_2[j, :, :, 2] = cv2.resize(label_for_train[j, :, :, 2],
        #                                           (label_for_train.shape[1] // 4, label_for_train.shape[2] // 4),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_2[j, :, :, 3] = cv2.resize(label_for_train[j, :, :, 3],
        #                                           (label_for_train.shape[1] // 4, label_for_train.shape[2] // 4),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_1[j, :, :, 0] = cv2.resize(label_for_train[j, :, :, 0],
        #                                           (label_for_train.shape[1] // 8, label_for_train.shape[2] // 8),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_1[j, :, :, 1] = cv2.resize(label_for_train[j, :, :, 1],
        #                                           (label_for_train.shape[1] // 8, label_for_train.shape[2] // 8),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_1[j, :, :, 2] = cv2.resize(label_for_train[j, :, :, 2],
        #                                           (label_for_train.shape[1] // 8, label_for_train.shape[2] // 8),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_1[j, :, :, 2] = cv2.resize(label_for_train[j, :, :, 2],
        #                                           (label_for_train.shape[1] // 8, label_for_train.shape[2] // 8),
        #                                           interpolation=cv2.INTER_CUBIC)

        # for j in range(label_for_train.shape[0]):
        #
        #     label_down_3[j, :, :, 0] = MeanReduce(label_for_train[j, :, :, 0],2)
        #     label_down_3[j, :, :, 1] = MeanReduce(label_for_train[j, :, :, 1],2)
        #     label_down_3[j, :, :, 2] = MeanReduce(label_for_train[j, :, :, 2],2)
        #     label_down_3[j, :, :, 3] = MeanReduce(label_for_train[j, :, :, 3],2)
            #
            # label_down_2[j, :, :, 0] = MeanReduce(label_down_3[j, :, :, 0], 2)
            # label_down_2[j, :, :, 1] = MeanReduce(label_down_3[j, :, :, 1], 2)
            # label_down_2[j, :, :, 2] = MeanReduce(label_down_3[j, :, :, 2], 2)
            # label_down_2[j, :, :, 3] = MeanReduce(label_down_3[j, :, :, 3], 2)
            # label_down_1[j, :, :, 0] = MeanReduce(label_for_train[j, :, :, 0], 8)
            # label_down_1[j, :, :, 1] = MeanReduce(label_for_train[j, :, :, 1], 8)
            # label_down_1[j, :, :, 2] = MeanReduce(label_for_train[j, :, :, 2], 8)
            # label_down_1[j, :, :, 3] = MeanReduce(label_for_train[j, :, :, 3], 8)
        # plt.imshow(image_for_train[3, :, :, 1], cmap='gray')
        # plt.show()
        yield (image_for_train, [label_for_train, label_down_3])
def crop_data(data_0, label_0):
    data_0 = data_0[:, (patch_row1 // 2 - patch_row // 2):(patch_row1 // 2 + patch_row // 2),
             (patch_col1 // 2 - patch_col // 2):(patch_col1 // 2 + patch_col // 2), :]
    label_0 = label_0[:, (patch_row1 // 2 - patch_row // 2):(patch_row1 // 2 + patch_row // 2),
              (patch_col1 // 2 - patch_col // 2):(patch_col1 // 2 + patch_col // 2), :]
    for j in range(data_0.shape[0]):
       data_0[j, :, :, :] = (data_0[j, :, :, :] - np.mean(data_0[j, :, :, :])) / np.std(data_0[j, :, :, :])
    return data_0, label_0


def build_unet():
    model = attn_reg_t([patch_row, patch_col, 3])
    print('Model Compiled')
    return model

def D_unet():
    model = attn_reg_d([patch_row, patch_col, 3])
    print('Model D')
    return model

def plot_training(history, log_dir, name):
    acc = history.history['acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(epochs, loss, 'r-', label='Train loss')
    lns2 = ax.plot(epochs, val_loss, 'g-', label='Validation loss')
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.grid()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    plt.savefig(log_dir + name+'.png')


if __name__ == '__main__':
    # output_width = 256
    # output_height = 256
    # batch_size = 8
    # EPOCH = 300
    # log_dir = 'result/cls04/'
    # root_path = r'/home/yiyingqiao/PycharmProjects/data_solve/DRsegment/xmzj/256'
    # train_annotation_path = r'note/multi/train_label.txt'
    # val_annotation_path = r'note/multi/val_label.txt'
    # test_annotation_path = r'note/multi/test_label.txt'
    # with open(train_annotation_path) as f:
    #     train_lines = f.readlines()
    # num_train = len(train_lines)
    # with open(val_annotation_path) as f:
    #     val_lines = f.readlines()
    # num_val = len(val_lines)
    # with open(test_annotation_path) as f:
    #     test_lines = f.readlines()
    # num_test = len(test_lines)
    # train_generator = generator(root_path, train_lines, batch_size, output_width, output_height)
    # val_generator = generator(root_path, val_lines, batch_size, output_width, output_height)
    # test_generator = generator(root_path, test_lines, num_test, output_width, output_height)
    dir1 = r'C:/Users/ZHOUFF/PycharmProjects/htfg/data/h_h5/'
    log_dir = r'C:\Users\ZHOUFF\PycharmProjects\htfg_tf2\log_all\log_attUnet_transfer\att_Unet16_4_4'
    model_dir = r'C:/Users/ZHOUFF/PycharmProjects/htfg_tf2/model_all/model_attUnet_transfer/att_Unet16_4_4.h5'
    train_number = 580
    vali_number = 298
    train_batch = 24
    vali_batch = 50

    patch_row1 = 96
    patch_col1 = 96

    patch_row = 64
    patch_col = 64
    loss = {
        # 'pred2':tversky_loss,
        'pred3_h': tversky_loss,
        'final_h': tversky_loss}
    loss_weights = {
        # 'pred2':0.2,
        'pred3_h': 0.3,
        'final_h': 0.7}
    dice = {
        # 'pred2': Dice,
        'pred3_h': Dice,
        'final_h': Dice}


    trainGenerator = inputcase2(dir1, 0, 1, train_batch, 'train_all2_h_3D_all_', 'train_data', 'train_label')
    validationGenerator = inputcase2(dir1, 0, 1, vali_batch,  'validation_all2_ht_3D_', 'vali_data', 'vali_label')
    model = build_unet()
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + '\ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_best_only=True, period=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1)

    pretrained_weights = r'C:/Users/ZHOUFF/PycharmProjects/htfg_tf2/log_all/log_attUnet_d/att_Unet32_4_d_2/ep345-loss0.005-val_loss0.003.h5'
    model.load_weights(pretrained_weights, by_name=True)
    print('Load weights {}.'.format(pretrained_weights))

    # count = 0
    for i in range(len(model.layers)-19):
        # if model.layers[i].name in dict_d:
        #     count += 1
            model.layers[i].trainable = False
    # print(count)
    print('Freeze the first {} layers of total {} layers.'.format((len(model.layers)-38), len(model.layers)))

    model.compile(optimizer=Adam(lr=1e-3), loss=loss, loss_weights=loss_weights, metrics=dice)
    history = model.fit_generator(trainGenerator, validation_data=validationGenerator, epochs=500,
                                  steps_per_epoch=max(1, train_number//train_batch),
                                  validation_steps=max(1, vali_number//vali_batch),

                                  callbacks=[logging, checkpoint, early_stopping])
    model.save(model_dir)
    # # model.save(log_dir)
    # # plot_training(history, log_dir, 'stage_1')

    # for i in range(len(model.layers)):
    #     model.layers[i].trainable = True
    # print('Unfreeze all of the layers.')
    # model.compile(optimizer=Adam(lr=1e-3), loss=loss, loss_weights=loss_weights, metrics=dice)
    # history = model.fit_generator(trainGenerator, validation_data=validationGenerator, epochs=500,
    #                               steps_per_epoch=max(1, train_number//train_batch),
    #                               validation_steps=max(1, vali_number//vali_batch),
    #                               # initial_epoch=20,
    #                               callbacks=[logging, checkpoint, early_stopping])
    # model.save(model_dir)
    # # plot_training(history, log_dir, 'final')






