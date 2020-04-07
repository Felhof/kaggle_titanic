""" Kaggle Titanic Model

    Uses a support vector machine to predict survival on the titanic
    using this dataset: https://www.kaggle.com/c/titanic/data

    Usage Example:
        to train with default svm parameters run: python titanic_svm.py
        to do a hyperparametersearch using sklearn first:
            python titanic_svm.py hps
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import resample

import numpy as np
import pandas as pd
import sys

numeric = ["Pclass","Age","SibSp","Parch","Fare"]

"""use trained model to create prediction from test data and save result to csv.

    Args:
        model (SVC): the support vector machine fitted to the training data
        evaluation_df (DataFrame): test data
        passenger_ids (Series): ids correspondong to test data
"""
def create_submission(model, evaluation_df, passenger_ids):
    test_pred = model.predict(evaluation_df)
    submission = pd.DataFrame({"PassengerId" : passenger_ids, "Survived": test_pred})
    submission.to_csv("submission.csv", index=False)
    print("Created and saved submission.csv")


"""Run the model on the training data and use ground truth to print various
    evalutation metrics

    Args:
        model (SVC): the support vector machine fitted to the training data
        X_test (DataFrame): test data
        y_test (Series): test ground truth

"""
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    confusion = confusion_matrix(y_test,y_pred)
    died_ratios = confusion[0,:] / sum(confusion[0])
    survived_ratios = confusion[1,:] / sum(confusion[1])
    ratios = np.vstack((died_ratios, survived_ratios))
    confusion_df = pd.DataFrame({" " : ["Died","Survived"], "Died" : ratios[:,0], "Survived" : ratios[:,1]})
    print("-------------Model Evaluation-------------------")
    print()
    print("Confusion matrix:")
    print()
    print(confusion_df.to_string(index=False))
    print()
    print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))
    print()


"""find best parameters to fit a svm on the given data using cross validation

    Args:
        X_train (DataFrame): training data
        y_train (Series): ground truth of training data

    Returns:
        Dict: the optimal parameters, a mapping from parameter names to values
"""
def hyperparmeter_search(X_train, y_train):
    parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf','poly']}
    ]
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)
    return clf.best_params_


"""process data so it can be modelled using a svm. This includes binarising
    categorical values and filling in empty values.

    Args:
        data_raw (DataFrame): the unprocessed data

    Returns:
        DataFrame: the processed data
"""
def preprocess(data_raw):
    data_raw.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis="columns", inplace=True)

    filled_data = pd.DataFrame()
    columns_to_drop = []
    for column_name, column_data in data_raw.iteritems():
        if column_data.isnull().sum() > 0:
            if column_name in numeric:
                data_mean = round(column_data.mean(), 3)
                filled_data[column_name] = column_data.fillna(data_mean)
            else:
                filled_data[column_name] = column_data.fillna("")
            columns_to_drop.append(column_name)
    data_raw.drop(columns_to_drop, axis="columns", inplace=True)
    data_raw = data_raw.join(filled_data)

    scaler = MinMaxScaler()
    data_raw[numeric] = scaler.fit_transform(data_raw[numeric])

    lb = LabelBinarizer()

    embarked_binarized = lb.fit_transform(data_raw["Embarked"])
    data_raw.drop("Embarked", axis="columns", inplace=True)
    data_raw = data_raw.join(pd.DataFrame(embarked_binarized))

    data_raw["Sex"] = lb.fit_transform(data_raw["Sex"])

    return data_raw


"""upsample an unbalanced dataset. This is to mitigate the model being biased
    in faver of predicting passengers dying as the majority in the training data
    dies

    Args:
        data (DataFrame): the data to upsample
        feature: the binary feature that is unbalanced
        minority: the value of the feature that is in the minority

    Returns:
        DataFrame: the upsampled data
"""
def upsample(data, feature, minority=1):
    data_majority = data[data[feature]==1-minority]
    data_minority = data[data[feature]==minority]

    data_minority_upsampled = resample(data_minority,
                                        replace=True,
                                        n_samples=len(data_majority))

    data_upsampled = pd.concat([data_majority, data_minority])
    return data_upsampled


if __name__ == "__main__":

    train_df=pd.read_csv("train.csv")
    evaluation_df=pd.read_csv("test.csv")
    test_ids = evaluation_df["PassengerId"]

    train_df = upsample(train_df, "Survived")
    data_raw = pd.concat([train_df, evaluation_df], ignore_index=True, sort=False)
    data_processed = preprocess(data_raw)
    train_df, evaluation_df = np.split(data_processed, [len(train_df)])
    evaluation_df.drop("Survived", axis="columns", inplace=True)

    features = train_df.drop("Survived", axis="columns")
    target = train_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(features, target)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    if "hps" in sys.argv[1:]:
        best_params = hyperparmeter_search(X_train, y_train)
        print("Found best hyperparameters:")
        print(best_params)
        best_svc = SVC(C=best_params["C"],kernel=best_params["kernel"], gamma=best_params.get("gamma", 'auto'))
    else:
        print("Using default parameters: C=10, linear kernel")
        best_svc = SVC(C=10,kernel="linear")

    best_svc.fit(X_train, y_train)

    evaluate(best_svc, X_test, y_test)

    create_submission(best_svc, evaluation_df, test_ids)
