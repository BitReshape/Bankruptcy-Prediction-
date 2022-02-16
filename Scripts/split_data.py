import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def return_class_distribution(df):
    df_num_rows         = df.shape[0]
    df_class_counts     = df['class'].value_counts()
    num_class_0         = df_class_counts[0]
    num_class_1         = df_class_counts[1]
    return 1-num_class_1/df_num_rows, num_class_1/df_num_rows

def split_data_based_on_distribution(list_df):
    training_data_list  = []
    testing_data_list   = []
    for df in list_df:
        df_X = df.drop('class', axis=1)
        df_y = df['class']
        print(return_class_distribution(df))
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, stratify=df_y)
        print(y_train.value_counts(), y_test.value_counts())
        training_data_list.append([X_train, y_train])
        testing_data_list.append([X_test, y_test])
    return training_data_list, testing_data_list


def split_data(df_year_1,
                df_year_2,
                df_year_3,
                df_year_4,
                df_year_5):

    print('\nSplit data started')

    training_data_list, testing_data_list = split_data_based_on_distribution([df_year_1,
                                                                                df_year_2,
                                                                                df_year_3,
                                                                                df_year_4,
                                                                                df_year_5])

    return training_data_list, testing_data_list