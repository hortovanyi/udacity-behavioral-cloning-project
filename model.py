import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import json
import random

from pathlib import PurePosixPath
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from collections import deque
from scipy.stats import norm
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_log_path', './data',
                    "Directroy where training driving_log.csv can be found")
flags.DEFINE_string('validation_log_path', '',
                    "Directory where validation driving_log.csv can be found")
flags.DEFINE_integer('epochs', 20, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")
flags.DEFINE_float('dropout', .50, "Keep dropout probabilities for nvidia model.")
flags.DEFINE_string('cnn_model', 'nvidia',
                    "cnn model either nvidia or commaai")

cameras = ['left', 'center', 'right']
camera_centre = ['center']
steering_adj = {'left': 0.25, 'center': 0., 'right': -.25}

# cameras = ['center']
# vehicle_controls = ['steering', 'throttle', 'brake']
vehicle_controls = ['steering']


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
    # return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def crop_camera(img, crop_height=66, crop_width=200):
    height = img.shape[0]
    width = img.shape[1]

    # y_start = 60+random.randint(-10, 10)
    # x_start = int(width/2)-int(crop_width/2)+random.randint(-40, 40)
    y_start = 60
    x_start = int(width/2)-int(crop_width/2)

    return img[y_start:y_start+crop_height, x_start:x_start+crop_width]


def load_data(log_path='./data', log_file='driving_log.csv', skiprows=1,
              cameras=cameras, sample_every=1, total_count=30000,
              filter_straights=False,
              crop_image=True):

    # initialise data extract
    features = []
    labels = []

    # used this routine from https://github.com/mvpcom/Udacity-CarND-Project-3
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

        return data_df.drop(data_df.index[drop_rows])

    def append_features_labels(image, controls):
        features.append(image)
        labels.append(controls)

    def gen_camera_image(row):
        # controls = [getattr(row, control) for control in vehicle_controls]
        steering = getattr(row, 'steering')

        # use one of the cameras randomily
        camera = cameras[random.randint(0, len(cameras)-1)]
        steering += steering_adj[camera]

        image = load_image(log_path, getattr(row, camera))

        image, steering = jitter_image_rotation(image, steering)

        if crop_image:
            image = crop_camera(image)

        # flip 50% randomily
        if random.random() >= .5:
            image = cv2.flip(image, 1)
            steering = -steering

        return image, steering

    # load and iterate over the csv log file
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

    print("Iterating with %d rows, sampling every %d, generating %d images."
          % (len(data_df), sample_every, total_count))

    # loop through a few times
    i = 0
    while i < total_count:
        for row in data_df.itertuples():
            # if this is not a row to sample then next
            if getattr(row, 'Index') % sample_every:
                continue

            image, steering = gen_camera_image(row)

            # append_features_labels(image, one_over_r(steering))
            append_features_labels(image, steering)

            # increment counter
            i += 1

    data = {'features': np.array(features),
            'labels': np.array(labels)}
    return data


def load_val_data(log_path='/u200/Udacity/behavioral-cloning-project/data',
                  log_file='driving_log.csv', camera=camera_centre,
                  crop_image=True, skiprows=1):

    def append_features_labels(image, controls):
        features.append(image)
        labels.append(controls)

    # initialise data extract
    features = []
    labels = []

    print("Camera: ", camera)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right',
                    'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,
                          names=column_names, skiprows=skiprows)
    print("Iterating with %d rows."
          % (len(data_df)))

    for row in data_df.itertuples():
        steering = getattr(row, 'steering')

        # adjust steering if not center
        steering += steering_adj[camera]

        image = load_image(log_path, getattr(row, camera))

        if crop_image:
            image = crop_camera(image)

        # append_features_labels(image, one_over_r(steering))
        append_features_labels(image, steering)

    data = {'features': np.array(features),
            'labels': np.array(labels)}
    return data


def load_train_val_data(training_log_path, validation_log_path,
                        crop_image=True):
    print("loading training data ...")
    train_data = load_data(training_log_path, sample_every=1,
                           filter_straights=True,
                           # cameras=camera_centre)
                           cameras=camera_centre, crop_image=crop_image)

    # X_train, X_val, y_train, y_val = train_test_split(train_data['features'],
    #                                                   train_data['labels'],
    #                                                   test_size=0.20,
    #                                                   random_state=737987)
    X_train = train_data['features']
    y_train = train_data['labels']

    print("loading valdiation data ...")
    validation_data = load_val_data(validation_log_path,
                                    camera=camera_centre[0],
                                    crop_image=crop_image)
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


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

    img_shape = (img_height, img_width, img_channels)

    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]
    pool_size = (2, 2)

    model = Sequential()
    model.add(Lambda(lambda x: x * 1./127.5 - 1,
                     input_shape=(img_shape),
                     output_shape=(img_shape), name='Normalization'))
    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l],
                                kernel_size[l][0], kernel_size[l][1],
                                border_mode=padding[l],
                                subsample=strides[l],
                                activation='elu'))
        # model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout))

    model.add(Flatten())

    neurons = [100, 50, 10]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation='elu'))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='elu', name='Out'))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')
    return model


def get_train_generator(X_data, y_data):
    datagen = ImageDataGenerator(
        # rescale=1./255,
        # rotation_range=15,
        # width_shift_range=0.4,
        # height_shift_range=0.4,
        # shear_range=0.2,
        # zoom_range=[.7, 1.],
        # horizontal_flip=True,
        fill_mode='nearest')
    datagen.fit(X_data)
    return datagen.flow(X_data, y_data, shuffle=True,
                        # save_to_dir='./augmented',
                        batch_size=FLAGS.batch_size)


def get_val_generator(X_data, y_data):
    datagen = ImageDataGenerator(
        # rescale=1./255
        )
    return datagen.flow(X_data, y_data, shuffle=True,
                        batch_size=FLAGS.batch_size)


def get_callbacks():
    checkpoint = ModelCheckpoint(
        "checkpoints/model-{val_loss:.4f}.h5",
        monitor='val_loss', verbose=1, save_weights_only=True,
        save_best_only=True)

    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)

    # return [checkpoint, tensorboard]

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                  patience=3, verbose=0, mode='auto')
    return [earlystopping, checkpoint]


def main(_):

    cnn_model = FLAGS.cnn_model

    crop_image = False
    if cnn_model == 'nvidia':
        crop_image = True

    # load bottleneck data
    X_train, y_train, X_val, y_val = load_train_val_data(
        FLAGS.training_log_path, FLAGS.validation_log_path,
        crop_image=crop_image)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

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
        get_train_generator(X_train, y_train),
        samples_per_epoch=len(X_train),
        nb_epoch=FLAGS.epochs,
        callbacks=get_callbacks(),
        validation_data=get_val_generator(X_val, y_val),
        nb_val_samples=len(X_val))

    # save weights and model
    model.save_weights('model.h5')
    with open('model.json', 'w') as modelfile:
        json.dump(model.to_json(), modelfile)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
