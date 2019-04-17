import pandas as pd

def get_data():
    data = pd.read_csv('AMZN.csv').values[:, 1:]

    X_train = data[:-1, 1:]
    y_train = data[1:, 0]


    return X_train, y_train
