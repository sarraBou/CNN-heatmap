import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import wget

# loading settings
model_string = "resnet50"  # inceptionv3, resnet50, vgg16
dataset_string = "imagenet"  # imagenet

# other settings
heatmap_intensity = 0.5

# model dependent imports, weight loading and model dependent parameter setting
if model_string == "inceptionv3":
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
    if dataset_string == "imagenet":
        model = InceptionV3(weights='imagenet')
    target_input_dimension = 299
    heatmap_dimension = 8
    last_layer_name = 'conv2d_93'
elif model_string == "resnet50":
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    if dataset_string == "imagenet":
        model = ResNet50(weights="imagenet")
    target_input_dimension = 224
    heatmap_dimension = 7
    last_layer_name = 'conv5_block3_3_conv'
elif model_string == "vgg16":
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
    if dataset_string == "imagenet":
        model = VGG16(weights="imagenet")
    target_input_dimension = 224
    heatmap_dimension = 14
    last_layer_name = 'block5_conv3'
else:  # use InceptionV3 and imagenet as default
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
    if dataset_string == "imagenet":
        model = InceptionV3(weights='imagenet')
    target_input_dimension = 299
    heatmap_dimension = 8
    last_layer_name = 'conv2d_93'

image_url = "https://indiasendangered.com/wp-content/uploads/2011/09/elephant.jpg"
# image_url = "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12234558/Chinook-On-White-03.jpg"
# image_url = "https://icatcare.org/app/uploads/2018/07/Thinking-of-getting-a-cat.png"

image_filename = wget.download(image_url)

img_downsized = image.load_img(image_filename, target_size=(target_input_dimension, target_input_dimension))

#.imshow("window 1", cv2.imread(image_filename))  # Visualize image
#cv2.waitKey()

x = image.img_to_array(img_downsized)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(decode_predictions(preds))

with tf.GradientTape() as tape:
  last_conv_layer = model.get_layer(last_layer_name)
  iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output]) # run model and achive certain output 'last_conv_layer.output'
  model_out, last_conv_layer = iterate(x) # x is the image
  class_out = model_out[:, np.argmax(model_out[0])] # ?
  grads = tape.gradient(class_out, last_conv_layer) # class out: take derivative; last_conv_layer: variable to derive from
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

heatmap = np.maximum(heatmap, 0) # ?
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((heatmap_dimension, heatmap_dimension))

#plt.matshow(heatmap)
#plt.show()

img_original = cv2.imread(image_filename)
heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
#plt.matshow(heatmap)
#plt.show()

heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
# plt.matshow(heatmap)
# plt.show()

img_heatmap = np.uint8(heatmap * heatmap_intensity + img_original)

cv2.imshow("window 2", img_heatmap)
cv2.waitKey()
