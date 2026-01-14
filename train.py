from data import Data
from features import Features
from model import Model

df = Data.load_data('data.csv')

df = Features.convert_cate_to_num(df, 'Sex', 'M', 'F')
df = Features.convert_cate_to_num(df, 'ExerciseAngina', 'Y', 'N')

X_train, X_test, y_train, y_test = Data.train_test_split(df, 'HeartDisease')

X_train, X_test, preprocessor = Features.preprocessing(X_train, X_test)

best_model, best_model_name, best_metrics = Model.build_and_evaluate_model(X_train, y_train, X_test, y_test)
