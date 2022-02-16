import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def sample_data(training_data_list):

    print('\nStart Sampling Data')

    training_data_list_sampled = []
    ros = RandomOverSampler()

    for X_train, y_train in training_data_list:
        # resampling X, y
        X_ros, y_ros = ros.fit_resample(X_train, y_train)
        training_data_list_sampled.append([X_ros, y_ros])
        print(y_ros.value_counts())

    return training_data_list_sampled
