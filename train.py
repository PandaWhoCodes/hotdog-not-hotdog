import os
import glob

import numpy as np

from skimage import io
from sklearn.cross_validation import train_test_split

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import to_categorical

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn import DNN

class hotdogTrainer(object):
    """ HotDog trainer """
    def __init__(self, path_hotdog_images, path_non_hotdog_images):
        """ default constructor """
        # path information
        self.path_hotdog_images = path_hotdog_images
        self.path_non_hotdog_images = path_non_hotdog_images
        # print(len(self.path_hotdog_images))
        # images information
        self.image_size = 32 # 32x32
        self.list_hotdog_files = []
        self.list_nonhotdog_files = []
        self.total_images_count = 0

        # tensorflow dataset
        self.tf_data_counter = 0
        self.tf_image_data = None
        self.tf_image_labels = None
        self.tf_x = None
        self.tf_x_test = None
        self.tf_y = None
        self.tf_y_test = None

        # tensorflow network variables
        self.tf_img_prep = None
        self.tf_img_aug = None
        self.tf_network = None

    def init_np_variables(self):
        """ Initialize NP datastructures """
        self.tf_image_data = np.zeros((self.total_images_count, self.image_size,
                                       self.image_size, 3), dtype='float64')

        self.tf_image_labels = np.zeros(self.total_images_count)

    def train(self):
        """ Start training """
        # 1: build a list of image filenames
        self.build_image_filenames_list()

        # 2: use list information to init our numpy variables
        self.init_np_variables()

        # 3: Add images to our Tensorflow dataset
        self.add_tf_dataset(self.list_hotdog_files, 0)
        self.add_tf_dataset(self.list_nonhotdog_files, 1)

        # 4: Process TF dataset
        self.process_tf_dataset()

        # 5: Setup image preprocessing
        self.setup_image_preprocessing()

        # 6: Setup network structure
        self.setup_nn_network()

        # 7: Train our deep neural network
        tf_model = DNN(self.tf_network, tensorboard_verbose=3,
                       checkpoint_path='model_hotdogs.tfl.ckpt')

        tf_model.fit(self.tf_x, self.tf_y, n_epoch=100, shuffle=True,
                     validation_set=(self.tf_x_test, self.tf_y_test),
                     show_metric=True, batch_size=30,
                     snapshot_epoch=True,
                     run_id='model_hotdogs')

        # 8: Save model
        tf_model.save('model_hotdogs.tflearn')

    def build_image_filenames_list(self):
        """ Get list of filenames for hotdogs and non hotdogs """
        self.list_hotdog_files = glob.glob('train/processed_dogs/*.jpg')
        self.list_nonhotdog_files = glob.glob('train/processed_notDogs/*.jpg')
        self.total_images_count = len(self.list_hotdog_files) + len(self.list_nonhotdog_files)
        print(self.total_images_count)

    def add_tf_dataset(self, list_images, label):
        """ Add tensorflow data we will pass to our network """
        # process list of images
        for image_file in list_images:
            try:
                # read, store image and label
                img = io.imread(image_file)
                self.tf_image_data[self.tf_data_counter] = np.array(img)
                self.tf_image_labels[self.tf_data_counter] = label

                # increase counter
                self.tf_data_counter += 1
            except:
                # on error continue to the next image
                continue

    def process_tf_dataset(self):
        """ Process our TF dataset """
        # split our tf set in a test and training part
        self.tf_x, self.tf_x_test, self.tf_y, self.tf_y_test = train_test_split(
            self.tf_image_data, self.tf_image_labels, test_size=0.1, random_state=42)

        # encode our labels
        self.tf_y = to_categorical(self.tf_y, 2)
        self.tf_y_test = to_categorical(self.tf_y_test, 2)

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

# path where we can find our images
PATH_HOTDOG_FILES = os.path.join('train/processed_dogs', '*.jpg')
PATH_NONHOTDOG_FILES = os.path.join('train/processed_notDogs', '*.jpg')
# print(len(PATH_HOTDOG_FILES))
# print(len(PATH_NONHOTDOG_FILES))
# start training
HOTDOGTRAINER = hotdogTrainer(PATH_HOTDOG_FILES, PATH_NONHOTDOG_FILES)
HOTDOGTRAINER.train()