import os

import tensorflow as tf


def set_gpu_memory(GPU_SET):
    if GPU_SET == 0:  ## CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                if GPU_SET == 1:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                elif GPU_SET == 2:
                    tf.config.experimental.set_virtual_device_configuration(
                            gpus[0],
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except Exception as E:
                print(E)


# export CUDA_VISIBLE_DEVICES = (1,2)