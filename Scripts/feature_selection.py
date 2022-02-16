from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np

def get_features (X_train, X_train_selected_features):
    list_attr               = list(X_train.head())
    not_selected_featured = []
    for row in range(X_train.shape[0]):
        selected_features   = []
        X_train_selected_row   = X_train.iloc[row]
        X_train_selected_features_selected_row = X_train_selected_features[row]
        for elem_train in range(len(X_train_selected_row)):
            for elem_feature in X_train_selected_features_selected_row:
                if X_train_selected_row[elem_train] == elem_feature:
                    selected_features.append(list_attr[elem_train])
                    break
        if len(selected_features) == len(X_train_selected_features_selected_row):
            not_selected_featured = [ elem for elem in list_attr if elem not in selected_features]
            return selected_features, not_selected_featured
        elif len(selected_features) > len(X_train_selected_features_selected_row):
            selected_features       = [ elem for elem in selected_features if selected_features.index(elem) < len(X_train_selected_features_selected_row)]
            not_selected_featured   = [ elem for elem in list_attr if elem not in selected_features]
            return selected_features, not_selected_featured
    raise ValueError("No Features found!")

def feature_selection_testing_data (not_selected_featured, X_test):
    X_test.drop(not_selected_featured, axis=1, inplace=True)

def Lasso_feature_selection():

    pass

def feature_selection(training_data_list, 
                        testing_data_list):

    print('\nstarting feature selection')

    training_data_list_feature_selected = []
    testing_data_list_feature_selected  = []

    for i in range(len(training_data_list)):
        X_train, y_train    = training_data_list[i]
        X_test, y_test      = testing_data_list[i]
        X_train_feature_selected = SelectKBest(mutual_info_classif, k=15).fit_transform(X_train, y_train)
        training_data_list_feature_selected.append([X_train_feature_selected, y_train])
        selected_features, not_selected_featured = get_features(X_train, X_train_feature_selected)
        feature_selection_testing_data (not_selected_featured, X_test)
        testing_data_list_feature_selected.append([X_test, y_test])
        print(selected_features)

    return training_data_list_feature_selected, testing_data_list_feature_selected

def good_class_separation(training_data_list,
                            list_attr):
    year_counter = 0
    for i in range(len(list_attr)-1):
        attr = list_attr[i]
        plt.subplot(2, 3, 1)
        plt.scatter(training_data_list[0][0][attr], np.zeros_like(training_data_list[0][0][attr]), c=training_data_list[0][1].map({0:'blue', 1: 'red'}))
        plt.subplot(2, 3, 2)
        plt.scatter(training_data_list[1][0][attr], np.zeros_like(training_data_list[1][0][attr]), c=training_data_list[1][1].map({0:'blue', 1: 'red'}))
        plt.subplot(2, 3, 3)
        plt.scatter(training_data_list[2][0][attr], np.zeros_like(training_data_list[2][0][attr]), c=training_data_list[2][1].map({0:'blue', 1: 'red'}))
        plt.subplot(2, 3, 4)
        plt.scatter(training_data_list[3][0][attr], np.zeros_like(training_data_list[3][0][attr]), c=training_data_list[3][1].map({0:'blue', 1: 'red'}))
        plt.subplot(2, 3, 5)
        plt.scatter(training_data_list[4][0][attr], np.zeros_like(training_data_list[4][0][attr]), c=training_data_list[4][1].map({0:'blue', 1: 'red'}))
        plt.title('Year datatset'+  str(year_counter) + ' Class Separation for attr: ' + str(attr), size = 15)
        plt.show()
        year_counter += 1
    pass

def same_beavhiour_feature(training_data_list,
                            list_attr):
    year_counter = 0
    for X_train, y_train  in training_data_list:
        for i in range(len(list_attr)-1):
            attr = list_attr[i]
            j = i + 1
            while j < len(list_attr):
                attr2 = list_attr[j]
                if attr != attr2:
                    plt.figure()
                    plt.title('Year datatset'+  str(year_counter) + ' Class Separation for attr: ' + str(attr) + ' and ' +str(attr2), size = 15)
                    plt.scatter(X_train[attr], X_train[attr2], c=y_train.map({0:'red', 1: 'blue'}))
                    plt.show()
                j += 1
        year_counter += 1
    pass

def manual_feature_selection(training_data_list, 
                                testing_data_list):
    list_attr = list(training_data_list[0][0].head())
    # Good Class Separation 
    #good_class_separation(training_data_list, list_attr)
    # Throw away Features that have the same behaviour
    #same_beavhiour_feature(training_data_list, list_attr)


    # Get all Correlation of feature Loanamount mit allen anderen
    #df[df.columns[1:]].corr()['LoanAmount'][:]

    pass

# Manual Feature Selection

# Correlation 
'''#df_year_3=(df_year_3-df_year_3.mean())/df_year_3.std()
print(df_year_3[df_year_3.columns[1:]].corr()['class'][:])
#df_year_4=(df_year_4-df_year_4.mean())/df_year_4.std()
print(df_year_4[df_year_4.columns[1:]].corr()['class'][:])
print(df_year_5[df_year_5.columns[1:]].corr()['class'][:])
'''

# Heatmap
'''import matplotlib.pyplot as plt
import seaborn as sns
data = df_year_3
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),cmap="RdYlGn")
plt.show()'''

# Lasso Feature Selection 
'''from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
num_features = 15
model1 = Lasso(alpha=.1)
rfe = RFE(estimator=model1, n_features_to_select=num_features)
rfe.fit(training_data_list[2][0], training_data_list[2][1])
ranking = rfe.ranking_
training_data_list[2][0] = rfe.transform(training_data_list[2][0])
#print(training_data_list[2][0])'''

