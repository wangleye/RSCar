import dataLoader as loader
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
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
    our_model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=INPUT_SHAPE))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Convolution2D(8, 3, 3, activation='relu'))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Convolution2D(16, 3, 3, activation='relu'))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Flatten())
    # our_model.add(Dropout(0.25))
    our_model.add(Dense(16, activation='relu'))
    our_model.add(Dropout(0.4))
    our_model.add(Dense(2, activation='softmax'))
    our_model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    plot(our_model, to_file='cnn_model.png')
    return our_model


def ResModel(n_channels=3):
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
    our_model.add(Dense(16, activation='relu'))
    our_model.add(Dropout(0.4))
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


def test_car_level(test_data, model):
    X_test, Y_test = construct_X_Y_car_level(test_data)
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


def test_ensemble_model(test_data, day_model, car_model):
    test_num = 0
    test_hit = 0
    for car_id in test_data.keys():
        all_traces = test_data[car_id]['feature']
        if len(all_traces) == 0:
            continue
        test_num += 1
        Y_true = test_data[car_id]['label']
        test_data_car = {}
        test_data_car[car_id] = test_data[car_id]
        Y_pred_probs = predict_one_car_ensemble(test_data_car, day_model, car_model)
        if np.argmax(Y_pred_probs) == np.argmax(Y_true):
            test_hit += 1
    acc = test_hit * 1.0 / test_num
    print(acc)
    return acc


def predict_one_car_ensemble(car_data, day_model, car_model):
    """
    ensemble two models for prediction
    """
    X_day, Y_day = construct_X_Y_day_level(car_data)
    Y_predictions = day_model.predict(X_day)
    Y_pred_day = np.mean(Y_predictions, axis=0)

    X_car, Y_car = construct_X_Y_car_level(car_data)
    Y_predictions = car_model.predict(X_car)
    Y_pred_car = np.mean(Y_predictions, axis=0)

    Y_pred_ensemble = (Y_pred_car + Y_pred_day) / 2.0
    return Y_pred_ensemble


def train(X, Y, model, num_batch, num_epoch, is_augmentation, augmentation_times):
    """
    if augmented, use augmented data to train the model
    """
    if is_augmentation:
        train_data_generator = ImageDataGenerator(rotation_range=40, horizontal_flip=True)  # , shear_range=0.2
        train_generator = train_data_generator.flow(X, Y, batch_size=num_batch)
        model.fit_generator(train_generator, samples_per_epoch=augmentation_times * X.shape[0], nb_epoch=num_epoch)
    else:
        model.fit(X_train, Y_train, batch_size=num_batch, nb_epoch=num_epoch, verbose=1)
    return model


if __name__ == "__main__":
    np.random.seed(123)  # expect reproduction, but Keras@Tensorflow still has problems (Keras Issue#2280)
    data = loader.mapFile()

    N = 5
    kf = KFold(n_splits=N)
    eval_measures = np.zeros((N, 3))  # index 0: ensemble acc, 1: car_level acc, 2: day_level acc

    ensemble = True  # use both car_level and day_level model?
    car_level = False  # car_level or day_level, only valid when ensemble is False
    num_epoch = 20
    num_batch = 4

    used_model = CnnModel  # choose from ResModel, CnnModel, basicModel

    is_data_augmentation = False  # use data augementation?
    times_of_augmentation = 3  # how many times more training pictures generated compared to original data

    # cross validation
    all_car_ids = np.array(list(data.keys()))
    cnt = 0
    for train_idx, test_idx in kf.split(all_car_ids):
        train_car_ids, test_car_ids = all_car_ids[train_idx], all_car_ids[test_idx]
        train_data = dict((k, v) for k, v in data.items() if k in train_car_ids)
        test_data = dict((k, v) for k, v in data.items() if k in test_car_ids)

        # car model
        if ensemble or car_level:
            INPUT_SHAPE = (24, 24, 8)
            X_train, Y_train = construct_X_Y_car_level(train_data)
            car_model = used_model()
            car_model = train(X_train, Y_train, car_model, num_batch, num_epoch,
                              is_data_augmentation, times_of_augmentation)
            eval_measures[cnt, 1] = test_car_level(test_data, car_model)

        # day model
        if ensemble or (not car_level):
            INPUT_SHAPE = (24, 24, 1)
            X_train, Y_train = construct_X_Y_day_level(train_data)
            day_model = used_model()
            day_model = train(X_train, Y_train, day_model, num_batch, num_epoch,
                              is_data_augmentation, times_of_augmentation)
            eval_measures[cnt, 2] = test_day_level(test_data, day_model)

        # ensemble
        if ensemble:
            eval_measures[cnt, 0] = test_ensemble_model(test_data, day_model, car_model)

        cnt += 1

    print ("average accuracy", np.mean(eval_measures, axis=0))
