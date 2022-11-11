import tensorflow as tf


def save_model(model, model_save_name):
    return model.save(f"{model_save_name}.h5")


def save_to_tflite(model, model_save_name):
    # converter = tf.lite.TFLiteConverter.from_saved_model(model_path_last)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()
    with open(f'{model_save_name}.tflite', 'wb') as f:
        f.write(tflite_model)