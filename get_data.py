import numpy as np
import pandas as pd

data = pd.read_csv('AMZN.csv').values[:, 1:]

X_train = data[:-1, 1:]
y_train = data[1:, 0]