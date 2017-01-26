import dataLoader as loader
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from residual_blocks import building_residual_block


def basicModel():
    """
    basic model structure:
    conv 5*5*5 + averagepooling 2*2 + conv 5*5*15 + averagepooling 3*3 + flatten + dense 30 + softmax
    """
    our_model = Sequential()
    our_model.add(Convolution2D(5, 5, 5, activation='relu', input_shape=INPUT_SHAPE))
    our_model.add(AveragePooling2D(pool_size=(2, 2)))
    our_model.add(Convolution2D(15, 5, 5, activation='relu'))
    our_model.add(AveragePooling2D(pool_size=(3, 3)))
    our_model.add(Flatten())
    our_model.add(Dense(30, activation='relu'))
    our_model.add(Dropout(0.3))
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
    our_model.add(Convolution2D(6, 4, 4, activation='relu', input_shape=INPUT_SHAPE))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Convolution2D(12, 4, 4, activation='relu'))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Flatten())
    # our_model.add(Dropout(0.25))
    # our_model.add(Dense(30, activation='relu'))
    our_model.add(Dropout(0.4))
    our_model.add(Dense(2, activation='softmax'))
    our_model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    plot(our_model, to_file='cnn_model.png')
    return our_model


def ResModel(n_channels=6):
    """
    residual model structure:
    input +
    [conv 3*3*n_channels + conv 3*3*n_channels] * num_res_blocks +
    flatten +
    softmax
    """
    our_model = Sequential()
    num_res_clocks = 3
    kernel_size = (3, 3)
    # build first layer
    our_model.add(building_residual_block(INPUT_SHAPE, n_channels, kernel_size))
    for i in range(num_res_clocks - 1):
        our_model.add(building_residual_block((INPUT_SHAPE[0], INPUT_SHAPE[1], n_channels), n_channels, kernel_size))
    our_model.add(Flatten())
    our_model.add(Dropout(0.2))
    # our_model.add(Dense(20, activation='relu'))
    # our_model.add(Dropout(0.25))
    our_model.add(Dense(2, activation='softmax'))
    our_model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    plot(our_model, to_file='res_model.png')
    return our_model


def construct_X_Y_day_level(evaluation_data):
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


def construct_X_Y_car_level(evaluation_data):
    # each car mostly has 8-day traces
    X_data = []
    Y_data = []
    cnt = 0
    for each in evaluation_data.keys():
        dat = evaluation_data[each]
        len_feature = len(dat['feature'])
        if len_feature == 0:
            continue
        elif len_feature > 8:
            len_feature = 8
            print ("more than 8 day traces!")
        elif len_feature < 8:  # use the existing day trace as the rest days
            for i in range(8 - len_feature):
                X_data.append(np.array(dat['feature'][i % len_feature]).flatten())
        for i in range(len_feature):
            d = dat['feature'][i]
            X_data.append(np.array(d).flatten())
        Y_data.append(np.array(dat['label']).flatten())
        cnt += 1
    X_data = np.array(X_data).reshape(cnt, 24, 24, 8)
    Y_data = np.array(Y_data).reshape(cnt, 2)
    return X_data, Y_data


def test_car_level(X_test, Y_test, model):
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score[1])
    return score[1]  # accuracy


def test_day_level(test_data, model):
    test_num = 0
    test_hit = 0
    for car_id in test_data.keys():
        all_traces = test_data[car_id]['feature']
        if len(all_traces) == 0:
            continue
        test_num += 1
        Y_true = test_data[car_id]['label']
        Y_predictions = np.zeros((len(all_traces), len(Y_true)))
        X = []
        for each_day_trace in all_traces:
            X.append(np.array(each_day_trace).flatten())
        X = np.array(X).reshape(len(all_traces), 24, 24, 1)
        Y_predictions = model.predict(X)
        Y_pred_probs = np.mean(Y_predictions, axis=0)
        if np.argmax(Y_pred_probs) == np.argmax(Y_true):
            test_hit += 1
    acc = test_hit * 1.0 / test_num
    print(acc)
    return acc


if __name__ == "__main__":
    np.random.seed(123)  # expect reproduction, but Keras@Tensorflow still has problems (Keras Issue#2280)
    data = loader.mapFile()

    N = 5
    kf = KFold(n_splits=N)
    eval_measures = np.zeros((N, 1))

    car_level_input = False  # car_level or day_level
    num_epoch = 10
    num_batch = 1

    if car_level_input:
        # each car as an input
        INPUT_SHAPE = (24, 24, 8)
        X_data, Y_data = construct_X_Y_car_level(data)
        # cross-validation
        cnt = 0
        for train_idx, test_idx in kf.split(X_data):
            # our_model = ResModel(4)
            # our_model = basicModel()
            our_model = CnnModel()
            X_train, X_test, Y_train, Y_test = X_data[train_idx], X_data[test_idx], Y_data[train_idx], Y_data[test_idx]
            our_model.fit(X_train, Y_train, batch_size=num_batch,
                          nb_epoch=num_epoch, verbose=1)
            eval_measures[cnt, :] = test_car_level(X_test, Y_test, our_model)
            cnt += 1
        print ("average accuracy", np.mean(eval_measures, axis=0))

    else:
        # each car-day as an input
        INPUT_SHAPE = (24, 24, 1)
        all_car_ids = np.array(list(data.keys()))
        cnt = 0
        for train_idx, test_idx in kf.split(all_car_ids):
            our_model = CnnModel()
            train_car_ids, test_car_ids = all_car_ids[train_idx], all_car_ids[test_idx]
            train_data = dict((k, v) for k, v in data.items() if k in train_car_ids)
            test_data = dict((k, v) for k, v in data.items() if k in test_car_ids)
            X_train, Y_train = construct_X_Y_day_level(train_data)
            our_model.fit(X_train, Y_train, batch_size=num_batch,
                          nb_epoch=num_epoch, verbose=1)
            eval_measures[cnt, :] = test_day_level(test_data, our_model)
            cnt += 1
        print ("average accuracy", np.mean(eval_measures, axis=0))
