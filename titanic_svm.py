from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

import numpy as np
import pandas as pd

numeric = ["Pclass"],["Age"],["SibSp"],["Parch"],["Fare"]

def preprocess(data_raw):
    data_raw.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis="columns")

    age_mean = round(data_raw["Age"].mean(), 3)
    data_raw["Age"] = data_raw["Age"].fillna(age_mean)

    scaler = MinMaxScaler()
    data_raw[numeric] = scaler.fit_transform(data_raw[numeric])

    lb = LabelBinarizer()

    data_raw["Embarked"] = data_raw["Embarked"].fillna(0)
    embarked_binarized = lb.fit_transform(data_raw["Embarked"])
    data_raw.drop("Embarked")
    data_raw = data_raw.join(embarked_binarized)

    data_raw["Sex"] = lb.fit_transform(data_raw["Sex"])


if __name__ == "main":

    train_df=pd.read_csv("train.csv")
    test_df=pd.read_csv("test.csv")
