from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

def model_accuracy(og, pred):
    report = classification_report(og, pred)
    f1 = f1_score(og, pred)
    recall = recall_score(og, pred)
    acc = accuracy_score(og, pred)
    return report, f1, recall, acc

def build_and_evaluate_model(X_train, y_train, X_test, y_test):
    classifiers = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, min_samples_split=10, min_samples_leaf=1,
            max_features='sqrt', max_depth=20, random_state=42
        ), 
        "GradientBoostingClassifier": GradientBoostingClassifier(
            subsample=0.85, n_estimators=100, min_samples_split=5,
            min_samples_leaf=2, max_depth=3, learning_rate=0.05, random_state=42
        ), 
        "AdaBoostClassifier": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.05,
            estimator=DecisionTreeClassifier(max_depth=3), random_state=42
        ),
        "LogisticRegression": LogisticRegression(solver='liblinear', C=1, random_state=42)
    }

    best_score = -1
    best_model_name = None
    best_metrics = None
    best_model = None

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        report_train, f1_train, recall_train, acc_train = model_accuracy(y_train, y_train_pred)
        report_test, f1_test, recall_test, acc_test = model_accuracy(y_test, y_test_pred)

        overfit_penalty = abs(f1_train - f1_test)
        score = f1_test / (1 + overfit_penalty)

        if score > best_score:
            best_score = score
            best_model_name = name
            best_metrics = {
                "train": {
                    "accuracy": acc_train,
                    "f1": f1_train,
                    "recall": recall_train,
                    "report": report_train
                },
                "test": {
                    "accuracy": acc_test,
                    "f1": f1_test,
                    "recall": recall_test,
                    "report": report_test
                }
            }
            best_model = model

    print(f"Best Model: {best_model_name}")
    print('Model performance for Training set')
    print(f"- Accuracy: {best_metrics['train']['accuracy']:.4f}")
    print(f"- F1 score: {best_metrics['train']['f1']:.4f}")
    print(f"- Recall: {best_metrics['train']['recall']:.4f}")
    print(f"- Report:\n{best_metrics['train']['report']}")
    print('-----------------------------------')
    print('Model performance for Testing set')
    print(f"- Accuracy: {best_metrics['test']['accuracy']:.4f}")
    print(f"- F1 score: {best_metrics['test']['f1']:.4f}")
    print(f"- Recall: {best_metrics['test']['recall']:.4f}")
    print(f"- Report:\n{best_metrics['test']['report']}")

    return best_model, best_model_name, best_metrics
