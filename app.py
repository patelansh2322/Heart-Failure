from flask import Flask, render_template, request 
import pickle 
import pandas as pd 
from src.data_logger import init_csv, log_row

app = Flask(__name__)
init_csv()

with open("output/model_pipeline.pkl", "rb") as f:
    saved = pickle.load(f)

pre = saved["preprocessor"]
mod = saved["model"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    risk = None

    if request.method == "POST":
        sex_map = {'M': 1, 'F': 0}
        angina_map = {'Y': 1, 'N': 0}

        f1 = int(request.form['Age'])
        f2 = sex_map[request.form['Sex']]
        f3 = request.form['ChestPainType']
        f4 = int(request.form['RestingBP'])
        f5 = int(request.form['Cholesterol'])
        f6 = int(request.form['FastingBS'])
        f7 = request.form['RestingECG']
        f8 = int(request.form['MaxHR'])
        f9 = angina_map[request.form['ExerciseAngina']]
        f10 = float(request.form['Oldpeak'])
        f11 = request.form['ST_Slope']

        data = {
            'Age': f1,
            'Sex': f2,
            'ChestPainType': f3,
            'RestingBP': f4,
            'Cholesterol': f5,
            'FastingBS': f6,
            'RestingECG': f7,
            'MaxHR': f8,
            'ExerciseAngina': f9,
            'Oldpeak': f10,
            'ST_Slope': f11
        }

        df = pd.DataFrame([data])

        X = pre.transform(df)
        pred = mod.predict(X)[0]

        prediction = "Heart Disease" if pred == 1 else "No Heart Disease"
        risk = round(mod.predict_proba(X)[0][1]*100, 1)

        log_row(f1, request.form['Sex'], f3, f4, f5, f6, f7, f8, request.form['ExerciseAngina'], f10, f11, pred)

    return render_template("index.html", prediction=prediction, risk=risk)

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8000)