import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import json
import random

from pathlib import PurePosixPath
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
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
flags.DEFINE_integer('dropout', .50, "Keep dropout probabilities.")

cameras = ['left', 'center', 'right']
camera_centre = ['center']
steering_adj = {'left': 0.25, 'center': 0., 'right': -0.25}

# cameras = ['center']
# vehicle_controls = ['steering', 'throttle', 'brake']
vehicle_controls = ['steering']


def load_data(log_path='./data', log_file='driving_log.csv', skiprows=1,
              cameras=cameras, sample_every=1, filter_straights=False):
    """
    Utility function to load data and images.

    Arguments:
        log_path - String (defaults './data')
        log_file - String (defaults 'driving_log.csv')
        skiprows - Int

    Returns:

    """
    # initialise data extract
    features = []
    labels = []

    def crop_camera(img, crop_height=66, crop_width=200):
        height = img.shape[0]
        width = img.shape[1]

        # y_start = 60+random.randint(-10, 10)
        # x_start = int(width/2)-int(crop_width/2)+random.randint(-40, 40)
        y_start = 60
        x_start = int(width/2)-int(crop_width/2)

        return img[y_start:y_start+crop_height, x_start:x_start+crop_width]

    # load image and convert to RGB
    def load_image(filename):
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

    def filter_driving_straight(data_df, hist_items=4):
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

    def augment_data(data_df):
        mu = 0
        sigma = 0.2

        zero_total = len(data_df.loc[data_df['steering'] == 0.0])
        print("zero angle total: ", zero_total)

        bin_len = 75
        # bins with values from -1 to 1
        bin = np.linspace(-1, 1, bin_len)

        # middle bin is 0.0 steering angle
        assert bin[int(bin_len/2)] == 0.0

        for b in range(len(bin)):
            # select rows for this bin
            if bin[b] == 0.0:
                bd_count = zero_total
                bd_df = data_df[data_df['steering'] == 0.0]
            elif bin[b] < 0.0:
                bd_df = data_df[data_df['steering'].
                                between(bin[b], bin[b+1]-0.0001)]
                bd_count = len(bd_df)
            else:
                bd_df = data_df[data_df['steering'].
                                between(bin[b-1]+0.0001, bin[b])]
                bd_count = len(bd_df)

            # work how many require for a gausian distribution
            if b == int(bin_len/2):
                bd_desired = zero_total
            else:
                bd_desired = int((zero_total/2.)
                                 * norm.pdf(bin[b], mu, sigma))

            print("bin[%.2d] steering: %f count: %d desired: %d"
                  % (b, bin[b], bd_count, bd_desired))

            # append extra rows
            bd_needed = bd_desired - bd_count
            if bd_needed > 0 and bd_count > 0:
                bd_df = bd_df.sample(frac=(1.*bd_desired/bd_count),
                                     replace=True)

            # 1st time through save the dataframe for later bins to append
            if b == 0:
                final_df = bd_df

            else:
                final_df = final_df.append(bd_df, ignore_index=True)

        return final_df

    def append_features_labels(image, controls):
        features.append(image)
        labels.append(controls)

    # load and iterate over the csv log file
    print("Cameras: ", cameras)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right',
                    'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,
                          names=column_names, skiprows=skiprows)
    # filter out stright line stretches
    if filter_straights:
        data_df = filter_driving_straight(data_df)

    data_df = augment_data(data_df)

    print(data_df.describe())

    print("Iterating with %d rows, sampling every %d."
          % (len(data_df), sample_every))

    for row in data_df.itertuples():
        # if this is not a row to sample then next
        if getattr(row, 'Index') % sample_every:
            continue

        # controls = [getattr(row, control) for control in vehicle_controls]
        steering = getattr(row, 'steering')

        # use one of the cameras randomily
        camera = cameras[random.randint(0, len(cameras)-1)]

        image = load_image(getattr(row, camera))
        image = crop_camera(image)

        # append_features_labels(image, controls)
        steering = steering+steering_adj[camera]

        # flip 50% randomily
        if random.random() >= .5:
            image = cv2.flip(image, 1)
            steering = -steering

        append_features_labels(image, steering)

    data = {'features': np.array(features),
            'labels': np.array(labels)}
    return data


def load_train_val_data(training_log_path, validation_log_path):
    print("loading training data ...")
    train_data = load_data(training_log_path, sample_every=1,
                           filter_straights=True,
                           # cameras=camera_centre)
                           cameras=cameras)

    X_train, X_val, y_train, y_val = train_test_split(train_data['features'],
                                                      train_data['labels'],
                                                      test_size=0.20,
                                                      random_state=737987)
    # X_train = train_data['features']
    # y_train = train_data['labels']
    #
    # print("loading valdiation data ...")
    # validation_data = load_data(validation_log_path, cameras=camera_centre,
    #                             filter_straights=False)
    # # validation_data = load_data(validation_log_path)
    # X_val = validation_data['features']
    # y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def build_model(img_height=66, img_width=200, img_channels=3, dropout=.4):

    img_shape = (img_height, img_width, img_channels)

    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]
    pool_size = (1, 1)

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
    model.add(Dropout(dropout))

    neurons = [100, 50, 10, 1]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation='elu'))

    model.add(Dropout(dropout))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['accuracy'])
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
    return datagen.flow(X_data, y_data, batch_size=FLAGS.batch_size)


def get_callbacks():
    # checkpoint = ModelCheckpoint(
    #     "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
    #     monitor='val_loss', verbose=0, save_best_only=False,
    #     save_weights_only=True, mode='auto', period=1)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)

    # return [checkpoint, tensorboard]

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                  patience=0, verbose=0, mode='auto')
    return [earlystopping]


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_train_val_data(
        FLAGS.training_log_path, FLAGS.validation_log_path)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # build model and display layers
    model = build_model(dropout=FLAGS.dropout)
    # for l in model.layers:
    #     print(l.name, l.input_shape, l.output_shape,
    #           l.activation if hasattr(l, 'activation') else 'none')
    print(model.summary())

    plot(model, to_file='model.png', show_shapes=True)

    model.fit_generator(
        get_train_generator(X_train, y_train),
        samples_per_epoch=len(X_train)*3,
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
