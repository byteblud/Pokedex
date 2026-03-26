import os
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import json
from bckgrd import backgrdrmv

os.environ["TF_USE_LEGACY_KERAS"] = "1"
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "pokedex_model.h5")
model = load_model(model_path,compile=False)

labels_path = os.path.join(BASE_DIR, "labels.json")
with open(labels_path, "r") as f:
    dataset = json.load(f)

index_to_name = {v: k for k, v in dataset.items()}

csv_path = os.path.join(BASE_DIR, "final_cleaned.csv")
df = pd.read_csv(csv_path)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def enter():
    return render_template("index.html")


@app.route("/output", methods=["POST"])
def out():
    file = request.files["pokemon_image"]

    if file.filename == "":
        return "No file uploaded"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = backgrdrmv(filepath)

    prediction = model.predict(img)
    predict = np.argmax(prediction)

    predicted_name = index_to_name.get(predict, "Unknown")

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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
