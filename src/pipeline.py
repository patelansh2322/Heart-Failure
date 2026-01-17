from src.data import load_data, split_data, save_data
from src.features import convert_cate_to_num, preprocessing
from src.model import build_and_evaluate_model
import pickle
import os
import pandas as pd

def run_pipeline(data_path, user_path, save_path="output/"):
    df = load_data(data_path)
    df_new_user = load_data(user_path)
    df = pd.concat([df, df_new_user], ignore_index=True)

    df = convert_cate_to_num(df)
    
    save_data(df, "data/new_data.csv")
    
    X_train, X_test, y_train, y_test = split_data(df, "HeartDisease")
    
    X_train, X_test, preprocessor = preprocessing(X_train, X_test)
    
    best_model, best_model_name, metrics = build_and_evaluate_model(X_train, y_train, X_test, y_test)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "model_pipeline.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "preprocessor": preprocessor}, f)
    
    return best_model_name, metrics
