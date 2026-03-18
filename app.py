import os
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import json
from bckgrd import backgrdrmv

app = Flask(__name__)

# Load model
model = load_model(r"C:\projects\deeplnproject\pokedex_model.h5")

# Load labels
with open("labels.json", "r") as f:
    dataset = json.load(f)

index_to_name = {v: k for k, v in dataset.items()}

df = pd.read_csv(r"C:\projects\deeplnproject\final_cleaned.csv")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def enter():
    return render_template("index.html")


@app.route("/output", methods=["POST"])
def out():
    file = request.files["pokemon_image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = backgrdrmv(filepath)

    prediction = model.predict(img)
    predict = np.argmax(prediction)

    predicted_name = index_to_name.get(predict)

    # 🔥 GET DATA FROM CSV
    row = df[df['Name'].str.lower() == predicted_name.lower()]

    if not row.empty:
        pokemon_data = row.iloc[0].to_dict()
    else:
        pokemon_data = {
            "Name": predicted_name,
            "HP": "N/A",
            "Attack": "N/A",
            "Defense": "N/A",
            "Speed": "N/A"
        }

    return render_template(
        "indexs.html",
        image=file.filename,
        data=pokemon_data
    )


if __name__ == "__main__":
    app.run(debug=True)