import numpy as np
import pandas as pd
from scipy.io import arff
from Scripts import preparing_data, split_data, standadization_data, sample_data, feature_selection, model_selection, shuffle_data

# load data using scipy (converst data to numpy array)
data_year_1 = arff.loadarff('./data/1year.arff')
data_year_2 = arff.loadarff('./data/2year.arff')
data_year_3 = arff.loadarff('./data/3year.arff')
data_year_4 = arff.loadarff('./data/4year.arff')
data_year_5 = arff.loadarff('./data/5year.arff')

# data (numpy array) to dataframe
df_year_1 = pd.DataFrame(data_year_1[0]) # the data contains financial rates from 1st year of the forecasting period and corresponding class label that indicates bankruptcy status after 5 years
df_year_2 = pd.DataFrame(data_year_2[0]) # the data contains financial rates from 2nd year of the forecasting period and corresponding class label that indicates bankruptcy status after 4 years.
df_year_3 = pd.DataFrame(data_year_3[0]) # the data contains financial rates from 3rd year of the forecasting period and corresponding class label that indicates bankruptcy status after 3 years. 
df_year_4 = pd.DataFrame(data_year_4[0]) # the data contains financial rates from 4th year of the forecasting period and corresponding class label that indicates bankruptcy status after 2 years. 
df_year_5 = pd.DataFrame(data_year_5[0]) # the data contains financial rates from 5th year of the forecasting period and corresponding class label that indicates bankruptcy status after 1 year. 

# understanding data
print(df_year_1.describe())
print(df_year_2.describe())
print(df_year_3.describe())
print(df_year_4.describe())
print(df_year_5.describe())

# removing missing data
'''
1. removing columns that have more that max_row_percentage_removal % of nans
2. removing rows with nans
3. removing outliers
'''
preparing_data.preparing_data(df_year_1, df_year_2, df_year_3, df_year_4, df_year_5, 0.01, 10)

# split data into training and test sets
training_data_list, testing_data_list = split_data.split_data(df_year_1, df_year_2, df_year_3, df_year_4, df_year_5)

# sample data
training_data_list = sample_data.sample_data(training_data_list)

# normalize data
#training_data_list_scaled, testing_data_list_scaled = training_data_list, testing_data_list
training_data_list_scaled = standadization_data.standardization(training_data_list, 'normalise')

# featuring
training_data_list_feature_selected, testing_data_list_feature_selected = feature_selection.feature_selection(training_data_list_scaled, testing_data_list)

# Feature Selection Manutally
#feature_selection.manual_feature_selection(training_data_list_scaled, testing_data_list_scaled)

# shuffle
training_dataset_shuffled, testing_datasets_shuffled = shuffle_data.shuffle_data(training_data_list_feature_selected, testing_data_list_feature_selected)

# Model Selection
#model_selection.model_selection(training_data_list_feature_selected, testing_data_list_feature_selected, 0)
#model_selection.model_selection(training_data_list_feature_selected, testing_data_list_feature_selected, 1)
model_selection.model_selection(training_dataset_shuffled, testing_datasets_shuffled, 2)
model_selection.model_selection(training_dataset_shuffled, testing_datasets_shuffled, 3)
model_selection.model_selection(training_dataset_shuffled, testing_datasets_shuffled, 4)