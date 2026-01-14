import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    @staticmethod
    def load_data(path):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def train_test_split(df, target):
        X = df.drop(columns=target)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test