from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import resample

import numpy as np
import pandas as pd

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


if __name__ == "__main__":

    train_df=pd.read_csv("train.csv")
    test_df=pd.read_csv("test.csv")

    train_df = upsample(train_df, "Survived")
    train_df = preprocess(train_df)

    features = train_df.drop("Survived", axis="columns")
    target = train_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(features, target)

    model = SVC()
    model.fit(X_train, y_train)

    kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts
    y_pred = cross_val_predict(model,features,target,cv=10)
    confusion = confusion_matrix(target,y_pred)
    died_ratios = [round(x, 3) for x in confusion[:,0] / sum(confusion[0])]
    survived_ratios = [round(x, 3) for x in confusion[:,1] / sum(confusion[1])]
    cunfusion_df = pd.DataFrame({" " : ["Died","Survived"], "Died" : died_ratios, "Survived" : survived_ratios})
    print("-------------Model Evaluation-------------------")
    print()
    print("Confusion matrix:")
    print()
    print(cunfusion_df.to_string(index=False))
    print()
    print(classification_report(target, y_pred, target_names=["Died", "Survived"]))
