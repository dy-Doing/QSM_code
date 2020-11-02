import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from tensorflow.keras.initializers import VarianceScaling, Constant
# from tensorflow_addons.activations import gelu


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

    shape_x = x.shape
    shape_g = g.shape

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
               bias_initializer=bias_init, padding="same", name=name + '_1', activation='relu')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit,
               bias_initializer=bias_init, padding="same", name=name + '_2', activation='relu')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    return x


def Transpose2D(input, filters):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu',
                    kernel_initializer=kinit, bias_initializer=bias_init)(input)
    return x


def OutputConv2D(input, outdim, name):
    x = Conv2D(16, (3, 3), strides=(1, 1), kernel_initializer=kinit,
               bias_initializer=bias_init, padding="same", name='conv_' + name, activation='relu')(input)
    x = BatchNormalization( name='out_bn_' + name)(x)
    out = Conv2D(outdim, (1, 1), activation='softmax', name= name)(x)
    return out


def Xnet(input_size, encoder_filters=(16, 32, 64, 128, 256), attention=True):
    img_input = Input(shape=input_size, name='input1')

    conv1 = UnetConv2D(img_input, encoder_filters[0], is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = UnetConv2D(pool1, encoder_filters[1], is_batchnorm=True, name='conv2')
    up1_1 = concatenate([Transpose2D(conv2, encoder_filters[0]), conv1], axis=-1, name='up1_1')
    conv1_1 = UnetConv2D(up1_1, encoder_filters[0], is_batchnorm=True, name='conv1_1')

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = UnetConv2D(pool2, encoder_filters[2], is_batchnorm=True, name='conv3')
    up2_1 = concatenate([Transpose2D(conv3, encoder_filters[1]), conv2], axis=-1, name='up2_1')
    conv2_1 = UnetConv2D(up2_1, encoder_filters[1], is_batchnorm=True, name='conv2_1')
    up1_2 = concatenate([Transpose2D(conv2_1, encoder_filters[0]), conv1, conv1_1], axis=-1, name='up1_2')
    conv1_2 = UnetConv2D(up1_2, encoder_filters[0], is_batchnorm=True, name='conv1_2')

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = UnetConv2D(pool3, encoder_filters[3], is_batchnorm=True, name='conv4')
    if attention:
        g1 = UnetGatingSignal(conv4, is_batchnorm=True, name='g1')
        attn1 = AttnGatingBlock(conv3, g1, 128, '_1')
        up3_1 = concatenate([Transpose2D(conv4, encoder_filters[2]), attn1], axis=-1, name='up3_1')
    else:
        up3_1 = concatenate([Transpose2D(conv4, encoder_filters[2]), conv3], axis=-1, name='up3_1')

    conv3_1 = UnetConv2D(up3_1, encoder_filters[2], is_batchnorm=True, name='conv3_1')
    if attention:
        g2 = UnetGatingSignal(conv3_1, is_batchnorm=True, name='g2')
        attn2 = AttnGatingBlock(conv2, g2, 64, '_2')
        up2_2 = concatenate([Transpose2D(conv3_1, encoder_filters[1]), conv2_1, attn2], axis=-1, name='up2_2')
    else:
        up2_2 = concatenate([Transpose2D(conv3_1, encoder_filters[1]), conv2, conv2_1], axis=-1, name='up2_2')

    conv2_2 = UnetConv2D(up2_2, encoder_filters[1], is_batchnorm=True, name='conv2_2')
    up1_3 = concatenate([Transpose2D(conv2_2, encoder_filters[0]), conv1, conv1_1, conv1_2], axis=-1, name='up1_3')
    conv1_3 = UnetConv2D(up1_3, encoder_filters[0], is_batchnorm=True, name='conv1_3')

    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # conv5 = UnetConv2D(pool4, encoder_filters[4], is_batchnorm=True, name='conv5')
    # up4_1 = concatenate([Transpose2D(conv5, encoder_filters[3]), conv4], axis=-1, name='up4_1')
    # conv4_1 = UnetConv2D(up4_1, encoder_filters[3], is_batchnorm=True, name='conv4_1')
    # up3_2 = concatenate([Transpose2D(conv4_1, encoder_filters[2]), conv3, conv3_1], axis=-1, name='up3_2')
    # conv3_2 = UnetConv2D(up3_2, encoder_filters[2], is_batchnorm=True, name='conv3_2')
    # up2_3 = concatenate([Transpose2D(conv3_2, encoder_filters[1]), conv2, conv2_1, conv2_2], axis=-1, name='up2_3')
    # conv2_3 = UnetConv2D(up2_3, encoder_filters[1], is_batchnorm=True, name='conv2_3')
    # up1_4 = concatenate([Transpose2D(conv2_3, encoder_filters[0]), conv1, conv1_1, conv1_2, conv1_3], axis=-1, name='up1_4')
    # conv1_4 = UnetConv2D(up1_4, encoder_filters[0], is_batchnorm=True, name='conv1_4')

    output1 = OutputConv2D(conv1_1, 3, 'output1')
    output2 = OutputConv2D(conv1_2, 3, 'output2')
    output3 = OutputConv2D(conv1_3, 3, 'final')
    # output4 = OutputConv2D(conv1_4, 2, 'output4')
    model = Model(inputs=[img_input], outputs=[output3, output2])
    return model


if __name__ == '__main__':
    Xnet([80, 80, 3])
