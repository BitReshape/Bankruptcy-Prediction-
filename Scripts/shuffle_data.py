from sklearn.utils import shuffle

def shuffle_data(training_dataset, testing_datasets):
    training_dataset_shuffled = []
    testing_datasets_shuffled = []
    for X_train, y_train in training_dataset:
        X_train, y_train = shuffle(X_train, y_train)
        training_dataset_shuffled.append([X_train, y_train])

    for X_test, y_test in testing_datasets:
        X_test, y_test = shuffle(X_test, y_test)
        testing_datasets_shuffled.append([X_test, y_test])

    return training_dataset_shuffled, testing_datasets_shuffled