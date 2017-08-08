# Load the trained model
# and predict
from __future__ import print_function
import argparse

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn import DNN

import scipy

import numpy as np


class hotdogClassifier(object):
    """ hotdog classifier """

    def __init__(self):
        """ default constructor """
        # Image
        self.image_size = 32  # 32x32

        # tensorflow network variables
        self.tf_img_prep = None
        self.tf_img_aug = None
        self.tf_network = None
        self.tf_model = None

        # 1: setup image preprocessing
        self.setup_image_preprocessing()

        # 2: setup neural network
        self.setup_nn_network()

    def setup_image_preprocessing(self):
        """ Setup image preprocessing """
        # normalization of images
        self.tf_img_prep = ImagePreprocessing()
        self.tf_img_prep.add_featurewise_zero_center()
        self.tf_img_prep.add_featurewise_stdnorm()

        # Randomly create extra image data by rotating and flipping images
        self.tf_img_aug = ImageAugmentation()
        self.tf_img_aug.add_random_flip_leftright()
        self.tf_img_aug.add_random_rotation(max_angle=30.)

    def setup_nn_network(self):
        """ Setup neural network structure """

        # our input is an image of 32 pixels high and wide with 3 channels (RGB)
        # we will also preprocess and create synthetic images
        self.tf_network = input_data(shape=[None, self.image_size, self.image_size, 3],
                                     data_preprocessing=self.tf_img_prep,
                                     data_augmentation=self.tf_img_aug)

        # layer 1: convolution layer with 32 filters (each being 3x3x3)
        layer_conv_1 = conv_2d(self.tf_network, 32, 3, activation='relu', name='conv_1')

        # layer 2: max pooling layer
        self.tf_network = max_pool_2d(layer_conv_1, 2)

        # layer 3: convolution layer with 64 filters
        layer_conv_2 = conv_2d(self.tf_network, 64, 3, activation='relu', name='conv_2')

        # layer 4: Another convolution layer with 64 filters
        layer_conv_3 = conv_2d(layer_conv_2, 64, 3, activation='relu', name='conv_3')

        # layer 5: Max pooling layer
        self.tf_network = max_pool_2d(layer_conv_3, 2)

        # layer 6: Fully connected 512 node layer
        self.tf_network = fully_connected(self.tf_network, 512, activation='relu')

        # layer 7: Dropout layer (removes neurons randomly to combat overfitting)
        self.tf_network = dropout(self.tf_network, 0.5)

        # layer 8: Fully connected layer with two outputs (hotdog or non hotdog class)
        self.tf_network = fully_connected(self.tf_network, 2, activation='softmax')

        # define how we will be training our network
        accuracy = Accuracy(name="Accuracy")
        self.tf_network = regression(self.tf_network, optimizer='adam',
                                     loss='categorical_crossentropy',
                                     learning_rate=0.0005, metric=accuracy)

    def load_model(self, model_path):
        """ Load model """
        self.tf_model = DNN(self.tf_network, tensorboard_verbose=0)
        self.tf_model.load(model_path)

    def predict_image(self, image_path):
        """ Predict image """
        # Load the image file
        img = scipy.ndimage.imread(image_path, mode="RGB")

        # Scale it to 32x32
        img = scipy.misc.imresize(img, (32, 32),
                                  interp="bicubic").astype(np.float32, casting='unsafe')

        # Predict
        return self.tf_model.predict([img])


# setup argument parser
parser = argparse.ArgumentParser(description='Decide if an image is a picture of a hotdog or not')
parser.add_argument('image', type=str, help='The image image file to check')
args = parser.parse_args()

# hotdog classifier
CLASSIFIER = hotdogClassifier()
CLASSIFIER.load_model("model_hotdogs.tflearn")
print(args.image)
predicted = CLASSIFIER.predict_image(args.image)

# Check the result.
if np.argmax(predicted[0]) == 0:
    print("It's a hotdog")
    print(predicted[0])
else:
    print("It's not a hotdog")
    print(predicted[0])
