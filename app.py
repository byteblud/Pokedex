import os

# ✅ Fix Keras compatibility (VERY IMPORTANT)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import json
from bckgrd import backgrdrmv

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 🔥 LOAD MODEL (WITH DEBUG)
# =========================
model_path =  "pokedex_model.keras"

try:
    model = load_model(model_path, compile=False,safe_mode=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None


# =========================
# 🔥 LOAD LABELS (WITH DEBUG)
# =========================
labels_path = os.path.join(BASE_DIR, "labels.json")

try:
    with open(labels_path, "r") as f:
        dataset = json.load(f)
    print("✅ Labels loaded successfully")
except Exception as e:
    print("❌ Labels loading failed:", e)
    dataset = {}

index_to_name = {v: k for k, v in dataset.items()}


# =========================
# 🔥 LOAD CSV (WITH DEBUG)
# =========================
csv_path = os.path.join(BASE_DIR, "final_cleaned.csv")

try:
    df = pd.read_csv(csv_path)
    print("✅ CSV loaded successfully")
except Exception as e:
    print("❌ CSV loading failed:", e)
    df = pd.DataFrame()


# =========================
# 📁 UPLOAD FOLDER
# =========================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# 🌐 ROUTES
# =========================
@app.route("/")
def enter():
    return render_template("index.html")


@app.route("/output", methods=["POST"])
def out():
    if model is None:
        return "Model not loaded. Check logs."

    file = request.files.get("pokemon_image")

    if not file or file.filename == "":
        return "No file uploaded"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = backgrdrmv(filepath)
    except Exception as e:
        return f"Image processing error: {e}"

    try:
        prediction = model.predict(img)
        predict = np.argmax(prediction)
    except Exception as e:
        return f"Prediction error: {e}"

    predicted_name = index_to_name.get(predict, "Unknown")

    try:
        row = df[df['Name'].str.lower() == predicted_name.lower()]
    except Exception as e:
        return f"CSV processing error: {e}"

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


# =========================
# 🚀 RUN APP (for local only)
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
