import os
import uuid
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import json
from bckgrd import backgrdrmv

app = Flask(__name__)

# ✅ FIX
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = load_model("model.keras")

# Load labels
with open("labels.json", "r") as f:
    dataset = json.load(f)

index_to_name = {v: k for k, v in dataset.items()}

# Load CSV
df = pd.read_csv("final_cleaned.csv")

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def enter():
    return render_template("index.html")


@app.route("/output", methods=["POST"])
def out():
    file = request.files.get("pokemon_image")

    if not file or file.filename == "":
        return "No file uploaded"

    # ✅ unique filename
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Image preprocessing
    img = backgrdrmv(filepath)

    img = np.array(img)
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    predict = np.argmax(prediction)

    print("Prediction:", prediction)

    predicted_name = index_to_name.get(predict, "Unknown")

    # CSV match
    df['Name'] = df['Name'].astype(str)
    row = df[df['Name'].str.lower().str.strip() == predicted_name.lower().strip()]

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
        image=filename,
        data=pokemon_data
    )


if __name__ == "__main__":
    app.run(debug=True)
