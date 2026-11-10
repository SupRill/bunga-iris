import joblib
import pandas as pd

FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(pd.DataFrame(data, columns=FEATURES))
