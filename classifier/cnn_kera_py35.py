import dataLoader as loader
import numpy as np
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D

consts = {
    'modelPath': './model/cnnmodel.ckpt',
    'maxepoch': 25,
}


def basicModel():
    """
    basic model structure:
    conv 5*5*5 + maxpooling 2*2 + conv 5*5*15 + maxpooling 3*3 + flatten + dense + softmax
    """
    our_model = Sequential()
    our_model.add(Convolution2D(5, 5, 5, activation='relu', input_shape=(24, 24, 1)))
    # max pooling tends to be better than avg pooling
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    # our_model.add(Dropout(0.25))
    our_model.add(Convolution2D(15, 5, 5, activation='relu'))
    our_model.add(MaxPooling2D(pool_size=(3, 3)))
    our_model.add(Flatten())
    our_model.add(Dense(30, activation='relu'))
    our_model.add(Dropout(0.25))
    our_model.add(Dense(2, activation='softmax'))
    our_model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
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


def train(train_data, model):
    X_train, Y_train = construct_X_Y(train_data)
    model.fit(X_train, Y_train, batch_size=1,
              nb_epoch=consts['maxepoch'], verbose=1)


def test(test_data, model):
    X_test, Y_test = construct_X_Y(test_data)
    score = model.evaluate(X_test, Y_test, verbose=1)
    print(score[1])


if __name__ == "__main__":
    data = loader.mapFile()
    kf = KFold(n=len(data.keys()), n_folds=5, shuffle=True)
    for tr, tst in kf:
        trk = np.asarray(list(data.keys()))[tr]
        tstk = np.asarray(list(data.keys()))[tst]
        train_data = dict((k, v) for k, v in data.items() if k in trk)
        test_data = dict((k, v) for k, v in data.items() if k in tstk)
        our_model = basicModel()
        train(train_data, our_model)
        test(test_data, our_model)
