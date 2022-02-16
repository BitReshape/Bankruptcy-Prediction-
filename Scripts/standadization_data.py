import numpy as np
import pandas as pd
from Scripts import preparing_data, split_data
from sklearn.preprocessing import StandardScaler

def get_varianz (training_data_list):
    list_var_of_attr = []
    for X_train,_ in training_data_list:
        list_var_of_attr.append(list(X_train.var()))
        
    return list_var_of_attr

def scaling (training_data_list):
    
    scalar = StandardScaler()
    training_data_list_scaled = []

    for X_train,y_train in training_data_list:
        #print(X_train.var())
        training_data_list_scaled.append([pd.DataFrame(scalar.fit_transform(X_train), columns=X_train.columns), y_train])
    
    return training_data_list_scaled

def normalise (training_data_list):
    
    training_data_list_scaled = []

    for X_train,y_train in training_data_list:
        X_train=(X_train-X_train.mean())/X_train.std()
        training_data_list_scaled.append([X_train, y_train])
    
    return training_data_list_scaled
    

def standardization(training_data_list, 
                    method):

    print('\nstart scaling data')

    list_var_of_attr = get_varianz(training_data_list)

    if method == 'scaling':
        training_data_list_scaled = scaling (training_data_list)
    elif method == 'normalise':
        training_data_list_scaled = normalise(training_data_list)

    return training_data_list_scaled