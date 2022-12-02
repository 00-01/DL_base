from keras import Input, Model
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Conv3D, DepthwiseConv2D, Dropout, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D
from keras.optimizers import Adam


def make_inc():
    val = [0]
    def inc():
        val[0] += 1
        return val[0]
    return inc


def res1(input, filter):
    x = Conv2D(filter, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.25)(x)

    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([input, x])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.25)(x)

    return x


def res2(input, filter):
    y = Conv2D(filter, (3, 3), strides=2, padding='same')(input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(.25)(y)

    x = Conv2D(filter, (3, 3), strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.25)(x)

    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([y, x])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.25)(x)

    return x


def res_block(filter, size, stride, padding, activation, dropout, name, input):
    k = Conv2D(filter, size, strides=stride, padding=padding, name=f"{name}_C2_skip")(input)
    k = BatchNormalization(name=f"{name}_BN_skip")(k)

    x = Conv2D(filter, size, strides=stride, padding=padding, name=f"{name}_C2")(input)
    x = BatchNormalization(name=f"{name}_0_BN")(x)
    x = Activation(activation, name=f"{name}_0_ACT")(x)
    x = Dropout(dropout, name=f"{name}_0_DO")(x)

    x = Conv2D(filter, size, padding='same', name=f"{name}_1_C2")(x)
    x = BatchNormalization(name=f"{name}_1_BN")(x)

    x = Add(name=f"{name}_ADD")([k, x])

    x = Activation(activation, name=f"{name}_2_ACT")(x)
    x = Dropout(dropout, name=f"{name}_2_DO")(x)

    return x


def con(name, input, filter=64, size=3, stride=1, padding='valid', activation='elu', dropout=0.25):
    x = Conv2D(filter, size, strides=stride, padding=padding, name=f"{name}_C2")(input)
    x = BatchNormalization(name=f"{name}_BN")(x)
    x = Activation(activation, name=f"{name}_ACT")(x)
    x = Dropout(dropout, name=f"{name}_DO")(x)

    return x


def bottleneck(name, input, filter=32, size0=3, size1=3, stride=1, padding='same', activation='elu', dropout=0.25):
    skip = Conv2D(filter, size0, strides=stride, padding=padding, activation=activation, name=f"{name}_SKIP")(input)
    if padding == 'valid':
        input = Conv2D(filter, size0, strides=stride, padding=padding, activation=activation, name=f"{name}_C0")(input)
        stride = 1
    x = Conv2D(filter*2, 1, strides=stride, padding='same', activation=activation, name=f"{name}_C")(input)
    x = DepthwiseConv2D(size1, padding='same', activation=activation, name=f"{name}_DW_C")(x)
    x = Conv2D(filter, 1, 1, padding='same', name=f"{name}_PW_C")(x)
    x = Dropout(dropout, name=f"{name}_DO0")(x)
    a = Add(name=f"{name}_ADD")([skip, x])
    return a


def unet(IN_SHAPE, pretrained_weights=None):
    input = Input(shape=(IN_SHAPE))

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input, conv10)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def res_unet(filter_root, depth, n_class=2, input_size=(80, 80, 1), activation='elu', batch_norm=True, final_activation='softmax'):
    """
    Build UNet model with ResBlock.
    Args:
        filter_root (int): Number of filters to start with in first convolution.
        depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model. Filter root and image size should be multiple of 2^depth.
        n_class (int, optional): How many classes in the output layer. Defaults to 2.
        input_size (tuple, optional): Input image size. Defaults to (80, 80, 1).
        activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
        batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
        final_activation (str, optional): activation for output layer. Defaults to 'softmax'.
    Returns:
        obj: keras model object
    """
    if len(input_size) == 3:
        Conv, MaxPooling, UpSampling = Conv2D, MaxPooling2D, UpSampling2D
    elif len(input_size) == 4:
        Conv, MaxPooling, UpSampling = Conv3D, MaxPooling3D, UpSampling3D

    long_connection_store = {}  # Dictionary for long connections


    inputs = Input(input_size)
    x = inputs

    for i in range(depth):
        out_channel = 2**i*filter_root

        # Residual/Skip connection
        res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
        if batch_norm:  conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
        act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
        if batch_norm:  conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)
        resconnection = Add(name="Add{}_1".format(i))([res, conv2])
        act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

        if i < depth-1:
            long_connection_store[str(i)] = act2
            x = MaxPooling(padding='same', name="MaxPooling{}_1".format(i))(act2)
        else:
            x = act2

    for i in range(depth-2, -1, -1):
        out_channel = 2**(i)*filter_root

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]

        up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv(out_channel, 2, activation='relu', padding='same', name="upConv{}_1".format(i))(up1)

        up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])

        up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conc)
        if batch_norm:  up_conv2 = BatchNormalization(name="upBN{}_1".format(i))(up_conv2)
        up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

        up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)
        if batch_norm:  up_conv2 = BatchNormalization(name="upBN{}_2".format(i))(up_conv2)

        # Residual/Skip connection
        res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)
        resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])
        x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

    output = Conv(n_class, 1, padding='same', activation=final_activation, name='output')(x)

    return Model(inputs, output, name='Res-UNet')


