import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from tensorflow.keras.initializers import VarianceScaling, Constant

from losses import tversky_loss

# initializers
kinit = VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
bias_init = Constant(value=0.1)


def expend_as(tensor, rep, name):
    my_repeat = tensor
    for i in range(rep-1):
        my_repeat = concatenate([my_repeat, tensor])
        # my_repeat = Lambda(lambda x, repnum: repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},
        #                    name='psi_up' + name)(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''

    shape_x = x.shape  # 32
    shape_g = g.shape  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl' + name)(x)  # 16
    shape_theta_x = theta_x.shape

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same', name='g_up' + name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name='psi' + name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = sigmoid_xg.shape
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3], name)
    y = multiply([upsample_psi, x], name='q_attn' + name)

    result = Conv2D(shape_x[3], (1, 1), padding='same', name='q_attn_conv' + name)(y)
    result_bn = BatchNormalization(name='q_attn_bn' + name)(result)
    return result_bn


def UnetGatingSignal(input, is_batchnorm, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = input.shape
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",
               kernel_initializer=kinit, bias_initializer=bias_init, name=name + '_conv')(
        input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_act')(x)
    return x


def UnetConv2D(input, outdim, is_batchnorm, name):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit,
               bias_initializer=bias_init, padding="same", name=name + '_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit,
               bias_initializer=bias_init, padding="same", name=name + '_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x


def attn_reg_d(input_size):
    img_input = Input(shape=input_size, name='input_scale1')

    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv2D(img_input, 16, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    input2 = Conv2D(16, (3, 3), padding='same', activation='relu', name='conv_scale2',
                    kernel_initializer=kinit, bias_initializer=bias_init)(scale_img_2)
    input2 = concatenate([input2, pool1], axis=3)
    # input2 = pool1
    conv2 = UnetConv2D(input2, 32, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    input3 = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_scale3',
                    kernel_initializer=kinit, bias_initializer=bias_init)(scale_img_3)
    input3 = concatenate([input3, pool2], axis=3)
    # input3 = pool2
    conv3 = UnetConv2D(input3, 64, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    input4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_scale4',
                    kernel_initializer=kinit, bias_initializer=bias_init)(scale_img_4)
    input4 = concatenate([input4, pool3], axis=3)
    # input4 = pool3
    conv4 = UnetConv2D(input4, 128, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = UnetConv2D(pool4, 256, is_batchnorm=True, name='center')

    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                       kernel_initializer=kinit, bias_initializer=bias_init)(center), attn1],
                      name='up1')
    conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate(
        [Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu',
                         kernel_initializer=kinit, bias_initializer=bias_init)(conv6),
         attn2], name='up2')
    conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                         kernel_initializer=kinit, bias_initializer=bias_init)(conv7),
         attn3], name='up3')
    conv8 = UnetConv2D(up3, 32, is_batchnorm=True, name='conv8')

    up4 = concatenate(
        [Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu',
                         kernel_initializer=kinit, bias_initializer=bias_init)(conv8),
         conv1], name='up4')

    conv9 = UnetConv2D(up4, 16, is_batchnorm=True, name='conv9')

    # out6 = Conv2D(3, (1, 1), activation='softmax', name='pred1')(conv6)
    # out7 = Conv2D(4, (1, 1), activation='softmax', name='pred2')(conv7)
    out8 = Conv2D(3, (1, 1), activation='softmax', name='pred3')(conv8)
    out9 = Conv2D(3, (1, 1), activation='softmax', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out9, out8])

    return model


if __name__ == '__main__':
    attn_reg_d([80, 80, 3])
