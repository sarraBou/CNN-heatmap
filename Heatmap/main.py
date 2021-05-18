import tensorflow.keras.backend as K
import wget
from Heatmap import GradCam

K.clear_session()
# loading settings
model_string = "inceptionv3"  # inceptionv3, resnet50, vgg16
dataset_string = "imagenet"  # imagenet

# other settings
heatmap_intensity = 0.5

# model dependent imports, weight loading and model dependent parameter setting
if model_string == "inceptionv3":
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    Model = InceptionV3
    if dataset_string == "imagenet":
        model = InceptionV3(weights='imagenet')
        print(model.summary())
    target_input_dimension = 299
    heatmap_dimension = 8
    last_layer_name = 'conv2d_93'
elif model_string == "resnet50":
    from tensorflow.keras.applications import ResNet50
    Model = ResNet50
    if dataset_string == "imagenet":
        model = ResNet50(weights="imagenet")
    target_input_dimension = 224
    heatmap_dimension = 7
    last_layer_name = 'conv5_block3_3_conv'
elif model_string == "vgg16":
    from tensorflow.keras.applications import VGG16
    Model = VGG16
    if dataset_string == "imagenet":
        model = VGG16(weights="imagenet")
    target_input_dimension = 224
    heatmap_dimension = 14
    last_layer_name = 'block5_conv3'
else:  # use InceptionV3 and imagenet as default
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    if dataset_string == "imagenet":
        model = InceptionV3(weights='imagenet')
    target_input_dimension = 299
    heatmap_dimension = 8
    last_layer_name = 'conv2d_93'

image_url = "https://indiasendangered.com/wp-content/uploads/2011/09/elephant.jpg"
# image_url = "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12234558/Chinook-On-White-03.jpg"
# image_url = "https://icatcare.org/app/uploads/2018/07/Thinking-of-getting-a-cat.png"

image_filename = wget.download(image_url)


GradCam.gradCAM(image_filename, model, model_string, target_input_dimension, heatmap_dimension, last_layer_name)