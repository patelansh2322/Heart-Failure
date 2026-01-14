import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

from data import Data
from features import Features

df = Data.load_data('data.csv')

df = Features.convert_cate_to_num(df, 'Sex', 'M', 'F')
df = Features.convert_cate_to_num(df, 'ExerciseAngina', 'Y', 'N')

X_train, X_test, y_train, y_test = Data.train_test_split(df, 'HeartDisease')

X_train, X_test, preprocessor = Features.preprocessing(X_train, X_test)

def model_accuracy(og, pred):
    report = classification_report(og, pred)
    f1 = f1_score(og, pred)
    recall = recall_score(og, pred)
    acc = accuracy_score(og, pred)
    return report, f1, recall, acc

classifiers = {"RandomForestClassifier": RandomForestClassifier(), 
"GradientBoostingClassifier": GradientBoostingClassifier(), 
"AdaBoostClassifier": AdaBoostClassifier(), 
"LogisticRegression": LogisticRegression()}

for i in range(len(list(classifiers))):
    model = list(classifiers.values())[i]
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    report, f1, recall, acc = model_accuracy(y_train, y_train_pred)
    report1, f11, recall1, acc1 = model_accuracy(y_test, y_test_pred)

    print(list(classifiers.keys())[i])
    print('Model performance for Training set')
    print(f"- Accuracy: {acc:0.4f}")
    print(f"- F1 score: {f1:0.4f}")
    print(f"- Recall: {recall:0.4f}")
    print(f"- Report:\n{report}")

    print('-----------------------------------')

    print('Model performance for Testing set')
    print(f"- Accuracy: {acc1:0.4f}")
    print(f"- F1 score: {f11:0.4f}")
    print(f"- Recall: {recall1:0.4f}")
    print(f"- Report:\n{report1}")


    print('='*35)
    print('\n')

random_forest = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

gradient_boost = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.85, 1.0]
}

ada_boost = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'estimator': [DecisionTreeClassifier(max_depth=1),
                       DecisionTreeClassifier(max_depth=2),
                       DecisionTreeClassifier(max_depth=3)]
}

logistic = {
    'C': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0, 1],
    'solver': ['liblinear']
}

hyper = {"RandomForestClassifier": [RandomForestClassifier(random_state=42), random_forest], 
"GradientBoostingClassifier": [GradientBoostingClassifier(random_state=42), gradient_boost], 
"AdaBoostClassifier": [AdaBoostClassifier(random_state=42), ada_boost], 
"LogisticRegression": [LogisticRegression(random_state=42), logistic]}
best = {}
for i in range(len(list(hyper))):
    model = list(hyper.values())[i][0]
    random_search = RandomizedSearchCV(
    estimator = model,
    param_distributions = list(hyper.values())[i][1],
    n_iter = 20,
    scoring = 'f1',
    cv = 5,
    verbose = 2,
    random_state = 42,
    n_jobs = -1
    )
    random_search.fit(X_train, y_train)
    best[list(hyper.keys())[i]] = random_search.best_params_
for keys, values in best.items():
    print(f"{keys}: {values}")

after_hyper = {"RandomForestClassifier": RandomForestClassifier(n_estimators = 300, min_samples_split = 10, min_samples_leaf = 1, max_features = 'sqrt', max_depth= 20), 
"GradientBoostingClassifier": GradientBoostingClassifier(subsample = 0.85, n_estimators = 100, min_samples_split = 5, min_samples_leaf = 2, max_depth = 3, learning_rate = 0.05), 
"AdaBoostClassifier": AdaBoostClassifier(n_estimators = 100, learning_rate = 0.05, estimator = DecisionTreeClassifier(max_depth=3)),
"LogisticRegression": LogisticRegression(solver = 'liblinear', l1_ratio = 0, C = 1)}
for i in range(len(list(after_hyper))):
    model = list(after_hyper.values())[i]
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    report, f1, recall, acc = model_accuracy(y_train, y_train_pred)
    report1, f11, recall1, acc1 = model_accuracy(y_test, y_test_pred)

    print(list(after_hyper.keys())[i])
    print('Model performance for Training set')
    print(f"- Accuracy: {acc:0.4f}")
    print(f"- F1 score: {f1:0.4f}")
    print(f"- Recall: {recall:0.4f}")
    print(f"- Report:\n{report}")

    print('-----------------------------------')

    print('Model performance for Testing set')
    print(f"- Accuracy: {acc1:0.4f}")
    print(f"- F1 score: {f11:0.4f}")
    print(f"- Recall: {recall1:0.4f}")
    print(f"- Report:\n{report1}")


    print('='*35)
    print('\n')