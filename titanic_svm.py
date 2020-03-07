from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import resample

import numpy as np
import pandas as pd
import sys

numeric = ["Pclass","Age","SibSp","Parch","Fare"]


def upsample(data, feature, minority=1):
    data_majority = data[data[feature]==1-minority]
    data_minority = data[data[feature]==minority]

    data_minority_upsampled = resample(data_minority,
                                        replace=True,
                                        n_samples=len(data_majority))

    data_upsampled = pd.concat([data_majority, data_minority])
    return data_upsampled


def preprocess(data_raw):
    data_raw.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis="columns", inplace=True)

    age_mean = round(data_raw["Age"].mean(), 3)
    data_raw["Age"] = data_raw["Age"].fillna(age_mean)

    scaler = MinMaxScaler()
    data_raw[numeric] = scaler.fit_transform(data_raw[numeric])

    lb = LabelBinarizer()

    data_raw["Embarked"] = data_raw["Embarked"].fillna("")
    embarked_binarized = lb.fit_transform(data_raw["Embarked"])
    data_raw.drop("Embarked", axis="columns", inplace=True)
    data_raw = data_raw.join(pd.DataFrame(embarked_binarized))

    data_raw["Sex"] = lb.fit_transform(data_raw["Sex"])

    return data_raw

def hyperparmeter_search(X_train, y_train):
    parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf','poly']}
    ]
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)
    return clf.best_params_



def evaluate(model, features, target):
    y_pred = model.predict(features)
    confusion = confusion_matrix(target,y_pred)
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
    print(classification_report(target, y_pred, target_names=["Died", "Survived"]))


if __name__ == "__main__":

    train_df=pd.read_csv("train.csv")
    test_df=pd.read_csv("test.csv")

    train_df = upsample(train_df, "Survived")
    train_df = preprocess(train_df)

    features = train_df.drop("Survived", axis="columns")
    target = train_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(features, target)

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
