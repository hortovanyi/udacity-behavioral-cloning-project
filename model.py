import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import json
import random

from pathlib import PurePosixPath
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_log_path', './data',
                    "Directroy where training driving_log.csv can be found")
flags.DEFINE_string('validation_log_path', '',
                    "Directory where validation driving_log.csv can be found")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 1000, "The batch size.")
flags.DEFINE_integer('training_size', 30000,
                     "The number of training samples per epoch")
flags.DEFINE_integer('validation_size', 5000,
                     "The number of validation samples per epoch")
flags.DEFINE_float('dropout', .60,
                   "Keep dropout probabilities for nvidia model.")
flags.DEFINE_string('cnn_model', 'nvidia',
                    "cnn model either nvidia or commaai")

cameras = ['left', 'center', 'right']
camera_centre = ['center']
steering_adj = {'left': 0.25, 'center': 0., 'right': -.25}


# load image and convert to RGB
def load_image(log_path, filename):
    filename = filename.strip()
    if filename.startswith('IMG'):
        filename = log_path+'/'+filename
    else:
        # load it relative to where log file is now, not whats in it
        filename = log_path+'/IMG/'+PurePosixPath(filename).name
    img = cv2.imread(filename)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# randomily change the image brightness
def randomise_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # brightness - referenced Vivek Yadav post
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0

    bv = .25 + np.random.uniform()
    hsv[::2] = hsv[::2]*bv

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# crop camera image to fit nvidia model input shape
def crop_camera(img, crop_height=66, crop_width=200):
    height = img.shape[0]
    width = img.shape[1]

    # y_start = 60+random.randint(-10, 10)
    # x_start = int(width/2)-int(crop_width/2)+random.randint(-40, 40)
    y_start = 60
    x_start = int(width/2)-int(crop_width/2)

    return img[y_start:y_start+crop_height, x_start:x_start+crop_width]


# referenced Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
def jitter_image_rotation(image, steering):
    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering


# if driving in a straight line remove extra rows
def filter_driving_straight(data_df, hist_items=5):
    print('filtering straight line driving with %d frames consective' %
          hist_items)
    steering_history = deque([])
    drop_rows = []

    for idx, row in data_df.iterrows():
        # controls = [getattr(row, control) for control in vehicle_controls]
        steering = getattr(row, 'steering')

        # record the recent steering history
        steering_history.append(steering)
        if len(steering_history) > hist_items:
            steering_history.popleft()

        # if just driving in a straight
        if steering_history.count(0.0) == hist_items:
            drop_rows.append(idx)

    # return the dataframe minus straight lines that met criteria
    return data_df.drop(data_df.index[drop_rows])


# jitter random camera image, adjust steering and randomise brightness
def jitter_camera_image(row, log_path, cameras):
    steering = getattr(row, 'steering')

    # use one of the cameras randomily
    camera = cameras[random.randint(0, len(cameras)-1)]
    steering += steering_adj[camera]

    image = load_image(log_path, getattr(row, camera))
    image, steering = jitter_image_rotation(image, steering)
    image = randomise_image_brightness(image)

    return image, steering


# create a training data generator for keras fit_model
def gen_train_data(log_path='./data', log_file='driving_log.csv', skiprows=1,
                   cameras=cameras, filter_straights=False,
                   crop_image=True, batch_size=128):

    # load the csv log file
    print("Cameras: ", cameras)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right',
                    'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,
                          names=column_names, skiprows=skiprows)

    # filter out straight line stretches
    if filter_straights:
        data_df = filter_driving_straight(data_df)

    data_count = len(data_df)

    print("Log with %d rows." % (len(data_df)))

    while True:  # need to keep generating data

        # initialise data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:
            row = data_df.iloc[np.random.randint(data_count-1)]

            image, steering = jitter_camera_image(row, log_path, cameras)

            # flip 50% randomily that are not driving straight
            if random.random() >= .5 and abs(steering) > 0.1:
                image = cv2.flip(image, 1)
                steering = -steering

            if crop_image:
                image = crop_camera(image)

            features.append(image)
            labels.append(steering)

        # yield the batch
        yield (np.array(features), np.array(labels))


# create a valdiation data generator for keras fit_model
def gen_val_data(log_path='/u200/Udacity/behavioral-cloning-project/data',
                 log_file='driving_log.csv', camera=camera_centre[0],
                 crop_image=True, skiprows=1,
                 batch_size=128):

    # load the csv log file
    print("Camera: ", camera)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right',
                    'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,
                          names=column_names, skiprows=skiprows)
    data_count = len(data_df)
    print("Log with %d rows."
          % (data_count))

    while True:  # need to keep generating data

        # initialise data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:
            row = data_df.iloc[np.random.randint(data_count-1)]
            steering = getattr(row, 'steering')

            # adjust steering if not center
            steering += steering_adj[camera]

            image = load_image(log_path, getattr(row, camera))

            if crop_image:
                image = crop_camera(image)

            features.append(image)
            labels.append(steering)

        # yield the batch
        yield (np.array(features), np.array(labels))


def build_commaai_model():
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     #  input_shape=(ch, row, col),
                     #  output_shape=(ch, row, col)))
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def build_nvidia_model(img_height=66, img_width=200, img_channels=3,
                       dropout=.4):

    # build sequential model
    model = Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)
    model.add(Lambda(lambda x: x * 1./127.5 - 1,
                     input_shape=(img_shape),
                     output_shape=(img_shape), name='Normalization'))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l],
                                kernel_size[l][0], kernel_size[l][1],
                                border_mode=padding[l],
                                subsample=strides[l],
                                activation='elu'))
        model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())

    # fully connected layers with dropout
    neurons = [100, 50, 10]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation='elu'))
        model.add(Dropout(dropout))

    # logit output - steering angle
    model.add(Dense(1, activation='elu', name='Out'))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')
    return model


def get_callbacks():
    # checkpoint = ModelCheckpoint(
    #     "checkpoints/model-{val_loss:.4f}.h5",
    #     monitor='val_loss', verbose=1, save_weights_only=True,
    #     save_best_only=True)

    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)

    # return [checkpoint, tensorboard]

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                  patience=1, verbose=1, mode='auto')
    # return [earlystopping, checkpoint]
    return [earlystopping]


def main(_):

    cnn_model = FLAGS.cnn_model

    crop_image = False
    if cnn_model == 'nvidia':
        crop_image = True

    # build model and display layers
    if cnn_model == 'nvidia':
        model = build_nvidia_model(dropout=FLAGS.dropout)
    else:
        model = build_commaai_model()
    # for l in model.layers:
    #     print(l.name, l.input_shape, l.output_shape,
    #           l.activation if hasattr(l, 'activation') else 'none')
    print(model.summary())

    plot(model, to_file='model.png', show_shapes=True)

    model.fit_generator(
        gen_train_data(log_path=FLAGS.training_log_path,
                       cameras=cameras,
                       #    cameras=camera_centre,
                       crop_image=crop_image,
                       batch_size=FLAGS.batch_size
                       ),
        samples_per_epoch=FLAGS.training_size,
        nb_epoch=FLAGS.epochs,
        callbacks=get_callbacks(),
        validation_data=gen_val_data(log_path=FLAGS.validation_log_path,
                                     crop_image=crop_image,
                                     batch_size=FLAGS.batch_size),
        nb_val_samples=FLAGS.validation_size)

    # save weights and model
    model.save_weights('model.h5')
    with open('model.json', 'w') as modelfile:
        json.dump(model.to_json(), modelfile)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
