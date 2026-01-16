from flask import Flask, render_template, request 
import pickle 
import pandas as pd 

app = Flask(__name__)

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

        data = {
            'Age': int(request.form['Age']),
            'Sex': sex_map[request.form['Sex']],
            'ChestPainType': request.form['ChestPainType'],
            'RestingBP': int(request.form['RestingBP']),
            'Cholesterol': int(request.form['Cholesterol']),
            'FastingBS': int(request.form['FastingBS']),
            'RestingECG': request.form['RestingECG'],
            'MaxHR': int(request.form['MaxHR']),
            'ExerciseAngina': angina_map[request.form['ExerciseAngina']],  # ✅ FIX
            'Oldpeak': float(request.form['Oldpeak']),
            'ST_Slope': request.form['ST_Slope']
        }

        df = pd.DataFrame([data])

        X = pre.transform(df)
        pred = mod.predict(X)[0]

        prediction = "Heart Disease" if pred == 1 else "No Heart Disease"
        risk = round(mod.predict_proba(X)[0][1]*100, 1)

    return render_template("index.html", prediction=prediction, risk=risk)

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8000)