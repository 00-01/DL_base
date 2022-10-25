from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dropout


def res1(tensor, filter):
    x = Conv2D(filter, (3, 3), padding='same')(tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.25)(x)

    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([tensor, x])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.25)(x)

    return x


def res2(tensor, filter):
    a = Conv2D(filter, (3, 3), strides=2, padding='same')(tensor)
    a = BatchNormalization()(a)
    a = Activation('relu')(a)
    a = Dropout(.25)(a)

    x = Conv2D(filter, (3, 3), strides=2, padding='same')(tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.25)(x)

    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([a, x])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.25)(x)

    return x


def res(filter, size, stride, padding, activation, dropout, name, tensor):
    a = Conv2D(filter, size, strides=stride, padding=padding, name=f"{name}_C2_skip")(tensor)
    a = BatchNormalization(name=f"{name}_BN_skip")(a)
    a = Activation(activation, name=f"{name}_ACT_skip")(a)
    a = Dropout(dropout, name=f"{name}_DO_skip")(a)

    x = Conv2D(filter, size, strides=stride, padding=padding, name=f"{name}_C2")(tensor)
    x = BatchNormalization(name=f"{name}_0_BN")(x)
    x = Activation(activation, name=f"{name}_0_ACT")(x)
    x = Dropout(dropout, name=f"{name}_0_DO")(x)

    x = Conv2D(filter, size, padding='same', name=f"{name}_1_C2")(x)
    x = BatchNormalization(name=f"{name}_1_BN")(x)

    x = Add(name=f"{name}_ADD")([a, x])

    x = BatchNormalization(name=f"{name}_2_BN")(x)
    x = Activation(activation, name=f"{name}_2_ACT")(x)
    x = Dropout(dropout, name=f"{name}_2_DO")(x)

    return x


def con(filter, size, stride, padding, activation, dropout, name, tensor):
    x = Conv2D(filter, size, strides=stride, padding=padding, name=f"{name}_C2")(tensor)
    x = BatchNormalization(name=f"{name}_BN")(x)
    x = Activation(activation, name=f"{name}_ACT")(x)
    x = Dropout(dropout, name=f"{name}_DO")(x)

    return x

def mobilenet_ssd():


    return x