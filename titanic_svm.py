from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.svm import SVC

import numpy as np
import pandas as pd

numeric = ["Pclass","Age","SibSp","Parch","Fare"]

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


if __name__ == "__main__":

    train_df=pd.read_csv("train.csv")
    test_df=pd.read_csv("test.csv")

    train_df = preprocess(train_df)

    features = train_df.drop("Survived", axis="columns")
    target = train_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(features, target)

    model = SVC()
    model.fit(X_train, y_train)

    kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts
    y_pred = cross_val_predict(model,features,target,cv=10)
    confusion = confusion_matrix(target,y_pred)
    cunfusion_df = pd.DataFrame({" " : ["Died","Survived"], "Died" : confusion[:,0], "Survived" : confusion[:,1]})
    print("-------------Model Evaluation-------------------")
    print()
    print("Confusion matrix:")
    print()
    print(cunfusion_df.to_string(index=False))
    print()
    print(classification_report(target, y_pred, target_names=["Died", "Survived"]))
