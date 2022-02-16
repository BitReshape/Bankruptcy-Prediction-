import numpy as np
import pandas as pd
from torch import float64, int64
import matplotlib.pyplot as plt
from scipy import stats

'''
This function checks NaN values within a data set
'''
def detect_outliers(df_list):
    
    list_attr = list(df_list[0].head())
    for attr in list_attr:
        plt.subplot(2, 3, 1)
        plt.boxplot(df_list[0][attr])
        plt.subplot(2, 3, 2)
        plt.boxplot(df_list[1][attr])
        plt.subplot(2, 3, 3)
        plt.boxplot(df_list[2][attr])
        plt.subplot(2, 3, 4)
        plt.boxplot(df_list[3][attr])
        plt.subplot(2, 3, 5)
        plt.boxplot(df_list[4][attr])
        plt.title(str(attr))
        plt.show()
    pass

def remove_outliers(df_list,
                    list_attr):
    df_list_return = []
    for df in df_list:
        df = df.quantile(0.10)
        df_list_return.append(df)
    return df_list_return

def nan_values (df):
    nan_index_dict    = {}
    for attr in df.head():
        nan_index_dict[attr] = {'index_list': [],
                                'length of list': 0,
                                'total df row number': df.shape[0]}
        nan_values = df[df[attr].isna()]
        nan_index_dict[attr]['index_list'] = list(nan_values.index)
        nan_index_dict[attr]['length of list'] = len(nan_index_dict[attr]['index_list'])
    return nan_index_dict
 
def remove_columns_nan_values (list_df,
                                list_nan_index_dict,
                                max_row_percentage_removal,
                                ):
    removing_attr_list = []
    attr_list = list(list_nan_index_dict[0].keys())
    for nan_index_dict in list_nan_index_dict:
        for attr in attr_list:
            if nan_index_dict[attr]['length of list']/ nan_index_dict[attr]['total df row number'] > max_row_percentage_removal:
                if attr not in removing_attr_list:
                    removing_attr_list.append(attr)
    # remove columns
    for df in list_df:
        df.drop(removing_attr_list, axis=1, inplace=True)
    
def remove_nan_values (list_df, 
                        list_nan_index_dict):
    attr_list = list(list_nan_index_dict[0].keys())
    for i in range(len(list_df)):
        df              = list_df[i]
        nan_index_dict  = list_nan_index_dict[i]
        # remove rows
        removing_rows = []
        for attr in attr_list:
            index_list = nan_index_dict[attr]['index_list']
            removing_rows += index_list
        df.drop(removing_rows, axis=0, inplace=True)
    pass 

def print_shape_classes(list_df):
    for df in list_df:
        print(df.shape)
        print(df['class'].value_counts())
    
def change_types(list_df):
    for df in list_df:
        df['class'] = df['class'].astype(int)

def preparing_data(df_year_1,
                    df_year_2,
                    df_year_3,
                    df_year_4,
                    df_year_5,
                    max_row_percentage_removal,
                    var_times):
    
    print('\nPreparing data started')
    list_attr = list(df_year_1.head())

    # removing rows and columns from dataframe containing NaN values
    nan_index_dict_year_1 = nan_values(df_year_1)
    nan_index_dict_year_2 = nan_values(df_year_2)
    nan_index_dict_year_3 = nan_values(df_year_3)
    nan_index_dict_year_4 = nan_values(df_year_4)
    nan_index_dict_year_5 = nan_values(df_year_5)

    removing_attr_list = remove_columns_nan_values([df_year_1,
                                                    df_year_2,
                                                    df_year_3,
                                                    df_year_4,
                                                    df_year_5],
                                                    [nan_index_dict_year_1, 
                                                    nan_index_dict_year_2, 
                                                    nan_index_dict_year_3, 
                                                    nan_index_dict_year_4, 
                                                    nan_index_dict_year_5], 
                                                    max_row_percentage_removal)
    
    nan_index_dict_year_1 = nan_values(df_year_1)
    nan_index_dict_year_2 = nan_values(df_year_2)
    nan_index_dict_year_3 = nan_values(df_year_3)
    nan_index_dict_year_4 = nan_values(df_year_4)
    nan_index_dict_year_5 = nan_values(df_year_5)
    
    remove_nan_values([df_year_1,
                        df_year_2,
                        df_year_3,
                        df_year_4,
                        df_year_5],
                        [nan_index_dict_year_1, 
                        nan_index_dict_year_2, 
                        nan_index_dict_year_3, 
                        nan_index_dict_year_4, 
                        nan_index_dict_year_5])
    
    # print new sizes of dataframes and the counts of classes
    print_shape_classes([df_year_1,
                        df_year_2,
                        df_year_3,
                        df_year_4,
                        df_year_5])

    # change class dtype (object) to int
    change_types([df_year_1,
                    df_year_2,
                    df_year_3,
                    df_year_4,
                    df_year_5])

    df_list_return = remove_outliers([df_year_1,
                                        df_year_2,
                                        df_year_3,
                                        df_year_4,
                                        df_year_5],
                                        list_attr)
    
    print_shape_classes([df_year_1,
                        df_year_2,
                        df_year_3,
                        df_year_4,
                        df_year_5])

    return df_list_return[0], df_list_return[1],df_list_return[2],df_list_return[3],df_list_return[4]
