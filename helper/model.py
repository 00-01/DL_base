from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dropout


def make_inc():
    val = [0]
    def inc():
        val[0] += 1
        return val[0]
    return inc


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
    y = Conv2D(filter, (3, 3), strides=2, padding='same')(tensor)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(.25)(y)

    x = Conv2D(filter, (3, 3), strides=2, padding='same')(tensor)
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


def res(filter, size, stride, padding, activation, dropout, name, tensor):
    k = Conv2D(filter, size, strides=stride, padding=padding, name=f"{name}_C2_skip")(tensor)
    k = BatchNormalization(name=f"{name}_BN_skip")(k)

    x = Conv2D(filter, size, strides=stride, padding=padding, name=f"{name}_C2")(tensor)
    x = BatchNormalization(name=f"{name}_0_BN")(x)
    x = Activation(activation, name=f"{name}_0_ACT")(x)
    x = Dropout(dropout, name=f"{name}_0_DO")(x)

    x = Conv2D(filter, size, padding='same', name=f"{name}_1_C2")(x)
    x = BatchNormalization(name=f"{name}_1_BN")(x)

    a = Add(name=f"{name}_ADD")([k, x])

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


