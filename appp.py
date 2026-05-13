import os
import json
import uuid
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from bckgrd import backgrdrmv

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Pokemon Vision AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

body{
    overflow-x:hidden;
}

.stApp{
    background:
    radial-gradient(circle at top left, #1e293b 0%, #0f172a 40%, #020617 100%);
    color:white;
}

/* Hide Streamlit Elements */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}


/* Sidebar */
section[data-testid="stSidebar"]{
    background: rgba(15,23,42,0.95);
    border-right:1px solid rgba(255,255,255,0.08);
}

/* Main Hero */
.hero{
    text-align:center;
    margin-top:20px;
    animation: fadeIn 1s ease;
}

.hero-title{
    font-size:70px;
    font-weight:800;
    color:white;
    line-height:1;
}

.hero-title span{
    color:#38bdf8;
}

.hero-sub{
    margin-top:15px;
    color:#cbd5e1;
    font-size:22px;
    font-weight:400;
}

.ai-badge{
    display:inline-block;
    margin-top:20px;
    padding:10px 25px;
    border-radius:999px;
    background:rgba(56,189,248,0.15);
    border:1px solid rgba(56,189,248,0.4);
    color:#7dd3fc;
    font-size:15px;
    font-weight:600;
}

/* Upload Box */
.upload-box{
    margin-top:40px;
    padding:45px;
    border-radius:30px;
    background:rgba(255,255,255,0.05);
    border:2px dashed rgba(255,255,255,0.15);
    backdrop-filter: blur(18px);
    box-shadow:0 10px 40px rgba(0,0,0,0.35);
    transition:0.4s;
}

.upload-box:hover{
    border:2px dashed #38bdf8;
    transform:translateY(-3px);
}

/* Pokemon Name */
.pokemon-name{
    font-size:60px;
    font-weight:800;
    color:white;
}

.pokemon-name span{
    color:#38bdf8;
}

/* Confidence */
.confidence-text{
    font-size:28px;
    color:#7dd3fc;
    margin-top:10px;
}

/* Stat Card */
.stat-card{
    background:rgba(255,255,255,0.05);
    padding:22px;
    border-radius:22px;
    border:1px solid rgba(255,255,255,0.08);
    transition:0.3s;
    margin-top:15px;
}

.stat-card:hover{
    transform:scale(1.03);
    background:rgba(255,255,255,0.08);
}

.stat-title{
    color:#94a3b8;
    font-size:18px;
    font-weight:500;
}

.stat-value{
    color:white;
    font-size:34px;
    font-weight:700;
    margin-top:5px;
}

/* Prediction Ring */
.ring{
    width:220px;
    height:220px;
    border-radius:50%;
    background:conic-gradient(
        #38bdf8 var(--percent),
        rgba(255,255,255,0.08) 0deg
    );
    display:flex;
    align-items:center;
    justify-content:center;
    margin:auto;
}

.ring-inner{
    width:170px;
    height:170px;
    border-radius:50%;
    background:#0f172a;
    display:flex;
    align-items:center;
    justify-content:center;
    flex-direction:column;
}

.ring-number{
    font-size:42px;
    font-weight:800;
    color:white;
}

.ring-label{
    color:#94a3b8;
    font-size:16px;
}

/* Loader */
.loader-card{
    padding:60px;
    text-align:center;
    border-radius:30px;
    background:rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
}

.loader-title{
    color:white;
    font-size:45px;
    font-weight:800;
}

.loader-sub{
    color:#cbd5e1;
    font-size:20px;
}

/* Footer */
.footer{
    margin-top:50px;
    text-align:center;
    color:#64748b;
    font-size:17px;
}

/* Animation */
@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(20px);
    }
    to{
        opacity:1;
        transform:translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FILE PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.keras")
labels_path = os.path.join(BASE_DIR, "labels.json")
csv_path = os.path.join(BASE_DIR, "final_cleaned.csv")

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(
    model_path,
    compile=False
)

# =========================================================
# LOAD LABELS
# =========================================================
with open(labels_path, "r") as f:
    dataset = json.load(f)

index_to_name = {v: k for k, v in dataset.items()}

# =========================================================
# LOAD CSV
# =========================================================
df = pd.read_csv(csv_path)

# =========================================================
# SESSION STATE
# =========================================================
if "show_result" not in st.session_state:
    st.session_state.show_result = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("⚡ Vision AI")

    st.markdown("---")

    st.markdown("### 🤖 Model Info")

    st.info("""
Deep Learning CNN Model

✔ Pokémon Recognition
✔ Background Removal
✔ Neural Prediction
✔ Confidence Scoring
""")

    st.markdown("---")

    st.markdown("### 📊 System")

    st.metric("Dataset", "800+ Pokémon")
    st.metric("Framework", "TensorFlow")
    st.metric("Prediction Speed", "< 3 sec")

    st.markdown("---")

    st.success("AI System Online")

    st.markdown("---")

    with st.expander("📜 Prediction History", expanded=True):

        if len(st.session_state.prediction_history) == 0:

            st.caption("No predictions yet")

        else:

            for item in st.session_state.prediction_history:

                st.markdown(f"""
                <div style="
                padding:12px;
                border-radius:15px;
                background:rgba(255,255,255,0.05);
                margin-bottom:10px;
                border:1px solid rgba(255,255,255,0.08);
                ">

                <div style="
                font-size:18px;
                font-weight:700;
                color:white;
                ">
                ⚡ {item['name']}
                </div>

                <div style="
                color:#7dd3fc;
                font-size:14px;
                margin-top:5px;
                ">
                Confidence: {item['confidence']}
                </div>

                </div>
                """, unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero">

<div class="hero-title">
⚡ Pokémon <span>Vision AI</span>
</div>

<div class="hero-sub">
Professional Deep Learning Powered Pokémon Recognition System
</div>

<div class="ai-badge">
Neural Network Active
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# UPLOADER
# =========================================================
if not st.session_state.show_result:

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Drag & Drop Pokémon Image",
        type=["png", "jpg", "jpeg"],
        key=st.session_state.uploader_key
    )

    st.markdown('</div>', unsafe_allow_html=True)

else:
    uploaded_file = st.session_state.uploaded_file

# =========================================================
# PREDICTION
# =========================================================
if uploaded_file is not None:

    st.session_state.uploaded_file = uploaded_file
    st.session_state.show_result = True

    filename = str(uuid.uuid4()) + "_" + uploaded_file.name
    save_path = os.path.join(BASE_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    placeholder = st.empty()

    # =====================================================
    # LOADER
    # =====================================================
    with placeholder.container():

        st.markdown("""
        <div class="loader-card">

        <img src="https://media.tenor.com/xVfFIHxAzW4AAAAi/pokemon-pikachu.gif"
        width="220">

        <div class="loader-title">
        Scanning Pokémon...
        </div>

        <div class="loader-sub">
        AI Neural Network is analyzing your image
        </div>

        </div>
        """, unsafe_allow_html=True)

        progress = st.progress(0)

        stages = [
            "Removing Background...",
            "Extracting Features...",
            "Running CNN Layers...",
            "Matching Pokémon...",
            "Generating Confidence..."
        ]

        status = st.empty()

        for i in range(5):

            status.info(stages[i])

            for j in range(20):

                progress.progress((i * 20) + j + 1)
                time.sleep(0.03)

    # =====================================================
    # IMAGE PROCESSING
    # =====================================================
    img = backgrdrmv(save_path)

    img = np.array(img)

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predict = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    predicted_name = index_to_name.get(predict, "Unknown")

    # =====================================================
    # SAVE HISTORY
    # =====================================================
    history_item = {
        "name": predicted_name,
        "confidence": f"{confidence:.2f}%"
    }

    if (
        len(st.session_state.prediction_history) == 0
        or
        st.session_state.prediction_history[0]["name"] != predicted_name
    ):
        st.session_state.prediction_history.insert(0, history_item)

    # =====================================================
    # FIND DATA
    # =====================================================
    df['Name'] = df['Name'].astype(str)

    row = df[
        df['Name'].str.lower().str.strip()
        ==
        predicted_name.lower().strip()
    ]

    if not row.empty:

        pokemon_data = row.iloc[0].to_dict()

    else:

        pokemon_data = {
            "HP": "N/A",
            "Attack": "N/A",
            "Defense": "N/A",
            "Speed": "N/A"
        }

    placeholder.empty()

    # =====================================================
    # RESULT SECTION
    # =====================================================
    col1, col2 = st.columns([1, 1])

    # LEFT
    with col1:

        pokemon_name_url = predicted_name.lower().replace(" ", "-")

        image_url = f"https://img.pokemondb.net/artwork/large/{pokemon_name_url}.jpg"

        st.image(image_url, width=350)

    # RIGHT
    with col2:

        st.markdown(f"""
        <div class="pokemon-name">
        ⚡ <span>{predicted_name}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="confidence-text">
        AI Confidence Score
        </div>
        """, unsafe_allow_html=True)

        degree = int((confidence / 100) * 360)

        st.markdown(f"""
        <div class="ring" style="--percent:{degree}deg;">
            <div class="ring-inner">
                <div class="ring-number">
                {confidence:.1f}%
                </div>
                <div class="ring-label">
                Accuracy
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # STATS
    # =====================================================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">❤️ HP</div>
            <div class="stat-value">{pokemon_data.get("HP","N/A")}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">⚔ Attack</div>
            <div class="stat-value">{pokemon_data.get("Attack","N/A")}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">🛡 Defense</div>
            <div class="stat-value">{pokemon_data.get("Defense","N/A")}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">⚡ Speed</div>
            <div class="stat-value">{pokemon_data.get("Speed","N/A")}</div>
        </div>
        """, unsafe_allow_html=True)

    st.balloons()

    # =====================================================
    # RESET BUTTON
    # =====================================================
    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        if st.button(
            "🔄 Predict Another Pokémon",
            use_container_width=True
        ):

            st.session_state.show_result = False
            st.session_state.uploaded_file = None
            st.session_state.uploader_key += 1

            st.rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
Made with ❤️ using Streamlit • TensorFlow • Deep Learning
</div>
""", unsafe_allow_html=True)
