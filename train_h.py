
from __future__ import absolute_import, division, print_function
import tensorflow as tf
# tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from attention_model import attn_reg_h
import numpy as np
import h5py
from losses import tversky_loss
from tensorflow.keras.optimizers import Adam
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from MeDIT.DataAugmentor import DataAugmentor2D, DataAugmentor3D
from MeDIT.DataAugmentor import AugmentParametersGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def maxpooling(feature_map, size=2, stride=2):
    channel = feature_map.shape[0]
    height = feature_map.shape[1]
    width = feature_map.shape[2]
    padding_height = np.uint16(round((height - size + 1) / stride))
    padding_width = np.uint16(round((width - size + 1) / stride))

    pool_out = np.zeros((channel, padding_height, padding_width), dtype=np.uint8)

    for map_num in range(channel):
        out_height = 0
        for r in np.arange(0, height, stride):
            out_width = 0
            for c in np.arange(0, width, stride):
                pool_out[map_num, out_height, out_width] = np.max(feature_map[map_num, r:r + size, c:c + size])
                out_width = out_width + 1
            out_height = out_height + 1
    # print(pool_out.shape)
    return pool_out[..., np.newaxis]

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
        # for j in range(label_for_train.shape[0]):
        #     label_down_3[j, :, :, 0] = cv2.resize(label_for_train[j, :, :, 0],
        #                                           (label_for_train.shape[1] // 2, label_for_train.shape[2] // 2),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_3[j, :, :, 1] = cv2.resize(label_for_train[j, :, :, 1],
        #                                           (label_for_train.shape[1] // 2, label_for_train.shape[2] // 2),
        #                                           interpolation=cv2.INTER_CUBIC)
        #     label_down_3[j, :, :, 2] = cv2.resize(label_for_train[j, :, :, 2],
        #                                           (label_for_train.shape[1] // 2, label_for_train.shape[2] // 2),
        #                                           interpolation=cv2.INTER_CUBIC)
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

        for j in range(label_for_train.shape[0]):

            label_down_3[j, :, :, 0] = MeanReduce(label_for_train[j, :, :, 0],2)
            label_down_3[j, :, :, 1] = MeanReduce(label_for_train[j, :, :, 1],2)
            label_down_3[j, :, :, 2] = MeanReduce(label_for_train[j, :, :, 2],2)
            label_down_3[j, :, :, 3] = MeanReduce(label_for_train[j, :, :, 3],2)
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

def plot_training(history, log_dir):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(epochs, loss, 'r-', label='Train loss')
    lns2 = ax.plot(epochs, val_loss, 'g-', label='Validation loss')
    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.savefig(log_dir + '/training.png')

dir1 = r'C:/Users/ZHOUFF/PycharmProjects/htfg/data/h_h5/'
log_dir = r'C:\Users\ZHOUFF\PycharmProjects\htfg_tf2\log_all\log_attUnet\att_Unet16_4_eeeee'
model_dir = r'C:/Users/ZHOUFF/PycharmProjects/htfg_tf2/model_all/model_attUnet/att_Unet16_4_eeee.h5'
# train_number = 1162
# vali_number = 269
train_number = 580
vali_number = 298
train_batch = 24
vali_batch = 50

patch_row1 = 96
patch_col1 = 96
patch_row = 64
patch_col = 64

model = attn_reg_h([patch_row, patch_col, 3])
loss = {
        # 'pred2':tversky_loss,
        'pred3':tversky_loss,
        'final': tversky_loss}
loss_weights = {
                # 'pred2':0.2,
                'pred3':0.3,
                'final': 0.7}
dice = {
        # 'pred2': Dice,
        'pred3': Dice,
        'final': Dice}
model.compile(optimizer=Adam(lr=1e-3), loss=loss, loss_weights=loss_weights, metrics=dice)

# 载入数据

trainGenerator = inputcase2(dir1, 0,1, train_batch, 'train_all2_h_3D_all_', 'train_data', 'train_label')
validationGenerator = inputcase2(dir1,0, 1, vali_batch, 'validation_all2_ht_3D_', 'vali_data', 'vali_label')

# 训练模型
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + '\ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss', save_best_only=True, period=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1)
history = model.fit_generator(trainGenerator, steps_per_epoch=train_number//train_batch, epochs=500,
                                  validation_data=validationGenerator,
                                  validation_steps=vali_number//vali_batch, callbacks=[logging, checkpoint, early_stopping])

print('history:')
print(history.history)
model.save(model_dir)
# plot_training(history, log_dir)