from src.data import load_data, split_data, save_data
from src.features import convert_cate_to_num, preprocessing
from src.model import build_and_evaluate_model
import pickle

def main():
    df = load_data("data/data.csv")

    df = convert_cate_to_num(df)

    save_data(df, 'data/new_data.csv')

    X_train, X_test, y_train, y_test = split_data(df, "HeartDisease")

    X_train, X_test, preprocessor = preprocessing(X_train, X_test)

    best_model, name, metrics = build_and_evaluate_model(X_train, y_train, X_test, y_test)

    with open("output/model_pipeline.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "preprocessor": preprocessor
    }, f)


if __name__ == "__main__":
    main()
