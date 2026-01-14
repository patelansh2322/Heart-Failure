import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Features:
    def convert_cate_to_num(df, col_name, cate1, cate2):
        df[col_name] = df[col_name].map({cate1: 1, cate2: 0}).astype(int)
        return df

    def preprocessing(X_train, X_test):
        numerical = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        binary = ['Sex', 'ExerciseAngina']
        cate = ['ChestPainType', 'RestingECG', 'ST_Slope']

        preprocessor = ColumnTransformer(
            transformers = [
                ("Numerical", StandardScaler(), numerical),
                ("Binary", "passthrough", binary),
                ("Categorical", OneHotEncoder(handle_unknown='ignore'), cate)
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        return X_train, X_test, preprocessor

