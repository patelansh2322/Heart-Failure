from src.pipeline import run_pipeline

if __name__ == "__main__":
    model_name, metrics = run_pipeline("data/data.csv", "output/")
    print(f"Trained model: {model_name}")
