# Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

def create_confusion_matrix(y_test, prediction, method_title):
    # Confusion Matrix
    cm = metrics.confusion_matrix(y_test, prediction)
    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix ' + str(method_title), size = 15)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"], rotation=45, size = 10)
    plt.yticks(tick_marks, ["0", "1"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual label', size = 15)
    plt.xlabel('Predicted label', size = 15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
            horizontalalignment='center',
            verticalalignment='center')
    plt.show()

def change_y_train(y_train, y_test):
    new_y_train = []
    new_y_test  = []
    new_y_train = [[1,0] if elem==0 else [0,1] for elem in y_train]
    new_y_test  = [[1,0] if elem==0 else [0,1] for elem in y_test]
    return new_y_train, new_y_test

def rechange_y_train(y_train, y_test):
    new_y_train = []
    new_y_test  = []
    new_y_train = [0 if elem[0]==1 else 1 for elem in y_train]
    new_y_test  = [0 if elem[0]==1 else 1 for elem in y_test]
    return new_y_train, new_y_test

def model_selection(training_data_list_feature_selected, testing_data_list_feature_selected, data_set_year):

    X_train, y_train    = training_data_list_feature_selected[data_set_year]
    X_test, y_test      = testing_data_list_feature_selected[data_set_year]

    # Decision Tree
    print('\nDecision Tree:')
    clf = DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(X_train, y_train)
    print("Decision Tree: Accuracy:")
    prediction = clf.predict(X_test)
    print(1 - (sum(1 for i in (prediction == y_test) if i==False)/len(prediction)))
    create_confusion_matrix(y_test, prediction, 'Decision Tree')
    dump(clf,'./Saved_Models/DT_'+ str(data_set_year) + '.joblib') 
    #pickle.dump(clf2, open('.Saved_Models/DT_'+ str(data_set_year) + '.sav', 'wb'))

    #Random Forest
    print('\nRandom Forest:')
    clf=RandomForestClassifier(n_estimators=100) # Gaussian Classifier
    clf.fit(X_train,y_train)
    prediction = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, prediction))
    create_confusion_matrix(y_test, prediction, 'Random Forest')
    dump(clf,'./Saved_Models/RF_'+ str(data_set_year) + '.joblib') 

    # Logistic regression 
    print('\nLogistic regression :')
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    prediction = logisticRegr.predict(X_test)
    score = logisticRegr.score(X_test, y_test)
    print('Logistic Regression Accuracy:', score)
    create_confusion_matrix(y_test, prediction, 'Logistic regression')
    dump(logisticRegr,'./Saved_Models/LG_'+ str(data_set_year) + '.joblib') 

    # Neural Network - FNN
    print('\nNeural Network:')
    model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(15,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    #model.fit(X_train, np.array(list(y_train)), epochs=15)
    y_train, y_test = change_y_train(y_train, y_test)
    model.fit( X_train, np.array(list(y_train)),
          batch_size=5,
          epochs=10)
    _, accuracy = model.evaluate(X_test, np.array(list(y_test)))
    print(accuracy)
    prediction = model.predict(X_test)
    _, prediction = rechange_y_train(y_train, prediction)
    _, new_y_test = rechange_y_train(y_train, y_test)
    create_confusion_matrix(new_y_test, prediction, 'Neural Network')
    model.save('./Saved_Models/NN_'+ str(data_set_year) + '.h5')

    pass