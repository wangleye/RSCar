import dataLoader as loader
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split
from residual_blocks import building_residual_block


def basicModel():
    """
    basic model structure:
    conv 5*5*5 + averagepooling 2*2 + conv 5*5*15 + averagepooling 3*3 + flatten + dense 30 + softmax
    """
    our_model = Sequential()
    our_model.add(Convolution2D(5, 5, 5, activation='relu', input_shape=(24, 24, 1)))
    our_model.add(AveragePooling2D(pool_size=(2, 2)))
    our_model.add(Convolution2D(15, 5, 5, activation='relu'))
    our_model.add(AveragePooling2D(pool_size=(3, 3)))
    our_model.add(Flatten())
    our_model.add(Dense(30, activation='relu'))
    our_model.add(Dropout(0.25))
    our_model.add(Dense(2, activation='softmax'))
    our_model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    plot(our_model, to_file='basic_model.png')
    return our_model


def CnnModel():
    """
    deep model structure:
    conv 4*4*6 + maxpooling 2*2 + conv 4*4*12 + maxpooling 2*2 + flatten + dense 30 + softmax
    """
    our_model = Sequential()
    our_model.add(Convolution2D(6, 4, 4, activation='relu', input_shape=(24, 24, 1)))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Convolution2D(12, 4, 4, activation='relu'))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Flatten())
    our_model.add(Dropout(0.25))
    our_model.add(Dense(30, activation='relu'))
    our_model.add(Dropout(0.25))
    our_model.add(Dense(2, activation='softmax'))
    our_model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    plot(our_model, to_file='cnn_model.png')
    return our_model


def ResModel():
    """
    residual model structure:
    input +
    conv 3*3*6 + conv 3*3*6] * num_res_blocks +
    flatten +
    softmax
    """
    our_model = Sequential()
    num_res_clocks = 4
    kernel_size = (3, 3)
    n_channels = 6
    # build first layer
    our_model.add(building_residual_block((24, 24, 1), n_channels, kernel_size))
    print(our_model.output_shape)
    for i in range(num_res_clocks - 1):
        our_model.add(building_residual_block((24, 24, n_channels), n_channels, kernel_size))
    our_model.add(Flatten())
    our_model.add(Dropout(0.5))
    our_model.add(Dense(2, activation='softmax'))
    our_model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    plot(our_model, to_file='res_model.png')
    return our_model


def construct_X_Y(evaluation_data):
    X_data = []
    Y_data = []
    cnt = 0
    for each in evaluation_data.keys():
        dat = evaluation_data[each]
        for d in dat['feature']:  # one car may have several days' data
            X_data.append(np.array(d).flatten())
            Y_data.append(np.array(dat['label']).flatten())
            cnt += 1
    X_data = np.array(X_data).reshape(cnt, 24, 24, 1)
    Y_data = np.array(Y_data).reshape(cnt, 2)
    return X_data, Y_data


def train(X_train, Y_train, model, num_epoch):
    model.fit(X_train, Y_train, batch_size=4,
              nb_epoch=num_epoch, verbose=1)


def test(X_test, Y_test, model):
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)


if __name__ == "__main__":
    np.random.seed(123)  # expect reproduction, but Keras@Tensorflow still has problems (Keras Issue#2280)
    data = loader.mapFile()
    X_data, Y_data = construct_X_Y(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.4, random_state=123)
    for i in range(10):
        our_model = ResModel()
        train(X_train, Y_train, our_model, 2)
        test(X_test, Y_test, our_model)
