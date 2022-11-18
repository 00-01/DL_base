from glob import glob

import numpy as np
import tensorflow as tf


# def save_model(model, model_save_name):
#     return model.save(f"{model_save_name}.h5")
# Helper function to run inference on a TFLite model


def run_tflite(tflite_file, test_image_indices):
    global test_images

    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices),), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = test_images[test_image_index]
        test_label = test_labels[test_image_index]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image/input_scale+input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output.argmax()

    return predictions


# def representative_dataset(dataset):
#     for data in dataset:
#         yield {"image": data.image,
#                "bias": data.bias,}

# def representative_dataset():
#     for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
#         yield [tf.dtypes.cast(data, tf.float32)]

# def representative_dataset():
#     for _ in range(100):
#         data = np.random.rand(1, 244, 244, 3)
#         yield [data.astype(np.float32)]

def representative_dataset():
    for i in range(100):
        a = np.reshape(X2[i], (1, X2.shape[1], X2.shape[2], 1))
        yield [a.astype(np.float32)]

def save_to_tflite(model, model_save_name, quantize=0, data=None):
    # converter = tf.lite.TFLiteConverter.from_saved_model(model_path_last)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize == 8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # for i in range(100):
        #     a = np.reshape(X2[i], (1, X2.shape[1], X2.shape[2], 1))
        # representative_dataset = [np.reshape(data[i], (1, data.shape[1], data.shape[2], 1)).astype(np.float32) for i in range(1000)]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
    elif quantize == 16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(f'{model_save_name}.tflite', 'wb') as f:
        f.write(tflite_model)


def inference_tflite(MODEL):
    interpreter = tf.lite.Interpreter(model_path=MODEL)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def inference_v2_1(MODEL, input, dtype='float32'):
    interpreter, input_details, output_details = inference_tflite(MODEL)

    if dtype == 'float32':
        input = input.astype(np.float32) #/255
    elif dtype == 'uint8':
        input = (input*255).astype(np.uint8)
    input1 = input.reshape(1, input.shape[0], input.shape[1], 1)

    interpreter.set_tensor(input_details[0]['index'], input1)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output = str(round(output_data[0, 0, 0, 0]))

    return output


if __name__ == "__main__":
    SAVE_PATH = f"../OUT/v2.1/model"
    model_num = -1
    model_path_last = sorted(glob(f"{SAVE_PATH}/model/*.h5"))[model_num]

    model = tf.keras.models.load_model(model_path_last)
    model_save_name = f"{SAVE_PATH}/model/new_model.tflite"
    # model_save_name = f"../OUT/v2.1/model/20221115-151918_mobileNet_ckpt.h5"

    save_to_tflite(model, model_save_name)

