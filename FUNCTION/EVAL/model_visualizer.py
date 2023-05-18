from keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
import visualkeras


def visual_keras(model, TIME):
    plot_model_path = f"OUT/{MODEL_VERSION}/plot/{TIME}.png"
    # color_map = defaultdict(dict)
    # color_map[Conv2D]['fill'] = 'orange'
    # color_map[ZeroPadding2D]['fill'] = 'gray'
    # color_map[Dropout]['fill'] = 'pink'
    # color_map[MaxPooling2D]['fill'] = 'red'
    # color_map[Dense]['fill'] = 'green'
    # color_map[Flatten]['fill'] = 'teal'

    visualkeras.layered_view(model,
                             draw_volume=True,
                             spacing=10,
                             # scale_xy=1,
                             # scale_z=1,
                             # max_z=1000,
                             # color_map=color_map,
                             # type_ignore=['Dropout,],
                             # to_file=plot_model_path,
                             ).show()  # display using your system viewer


def weight_visualizer(model, n1, n2):
    for i, l in enumerate(model.layers[n1:n2]):
        if "C2" in l.name:
            filters, biases = l.get_weights()
            print(l.name, filters.shape)

            fig = plt.figure(figsize=(8, 12))
            rows = int(round(np.sqrt(filters.shape[3])))
            columns = int(round(np.sqrt(filters.shape[3])))
            for i in range(1, columns*rows+1):
                f = filters[:, :, :, i-1]
                fig = plt.subplot(rows, columns, i)
                fig.axis('off')
                # fig.set_xticks([])  # Turn off axis
                # fig.set_yticks([])
                plt.imshow(f[:, :, 0], cmap='gray')  # Show only the filters from 0th channel (R)
                # ix += 1
            plt.show()


def filter_visualizer(model, img):
    conv_layers = model.layers[1].output
    print(conv_layers)
    # conv_layers = [i.output for i in MODEL.layers if "C2" in i.name][:1]
    # print(conv_layers)
    visualize_model = Model(model.inputs, conv_layers)
    print(visualize_model.summary())
    # %%
    for i in range(20, 100, 10):
        re_img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        conv_img = visualize_model.predict(re_img)
        # columns = int(round(np.sqrt(MODEL.shape[1])))
        # rows = int(round(np.sqrt(MODEL.shape[2])))
        columns = 8
        rows = 8
        for c_img in conv_img:
            # pos = 1
            fig = plt.figure(figsize=(12, 12))
            for i in range(1, columns*rows+1):
                fig = plt.subplot(rows, columns, i)
                fig.axis('off')
                plt.imshow(c_img[:, :, i-1], cmap='gray')
                # pos += 1
            plt.show()


