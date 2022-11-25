import tensorflow as tf


def lr_flip():
    result = tf.image.flip_left_right(image)

    return result


def lr_flip():
    result = tf.image.flip_left_right(image)

    return result


def augment(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE+6, IMG_SIZE+6)
    # Make a new seed.
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # Random crop back to the original size.
    image = tf.image.stateless_random_crop(
            image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    # Random brightness.
    image = tf.image.stateless_random_brightness(
            image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label

flip_lr = flip_left_right(image)
flip_ud = flip_up_down(image)
rot_90 = rot90(image)
crop_center = central_crop(image, central_fraction=0.5)
rgb2gray = rgb_to_grayscale(image)
saturate = adjust_saturation(image, 3)
bright = adjust_brightness(image, 0.4)

## ---------------------------------------------------------------- random_brightness
for i in range(3):
    seed = (i, 0)  # tuple of size (2,)
    random_brightness = stateless_random_brightness(image, max_delta=0.95, seed=seed)
    visualize(image, random_brightness)

## ---------------------------------------------------------------- random_contrast
for i in range(3):
    seed = (i, 0)  # tuple of size (2,)
    random_contrast = stateless_random_contrast(image, lower=0.1, upper=0.9, seed=seed)
    visualize(image, random_contrast)

## ---------------------------------------------------------------- random_crop
for i in range(3):
    seed = (i, 0)  # tuple of size (2,)
    random_crop = tf.image.stateless_random_crop(image, size=[210, 300, 3], seed=seed)
    visualize(image, random_crop)

