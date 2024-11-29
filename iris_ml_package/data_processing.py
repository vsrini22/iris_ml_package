from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    iris = load_iris(as_frame=True)
    data = iris['data']
    data['target'] = iris['target']
    return data

def clean_data(dataframe):
    dataframe.dropna(inplace=True)
    return dataframe
