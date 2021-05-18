import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def gradCAM(orig, model, model_string, DIM, HM_DIM, last_layer, intensity=0.5, res=250):

    if model_string == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

    elif model_string == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

    elif model_string == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

    else:  # use InceptionV3 and imagenet as default
        from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions


    img = image.load_img(orig, target_size=(DIM, DIM))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = imagenet_utils.decode_predictions(preds)
    (imagenetID, label, prob) = decoded[0][0]
    label = "{}: {:.2f}%".format(label, prob * 100)
    print("[INFO] {}".format(label))

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(last_layer)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output]) # run model and achive certain output 'last_conv_layer.output'
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer) # class out: take derivative; last_conv_layer: variable to derive from
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((HM_DIM, HM_DIM))

    img_original = cv2.imread(orig)
    heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
    # plt.matshow(heatmap)
    # plt.show()

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # plt.matshow(heatmap)
    # plt.show()
    img_heatmap = np.uint8(heatmap * intensity + img_original)

    cv2.imshow("window 2", img_heatmap)
    cv2.waitKey()




