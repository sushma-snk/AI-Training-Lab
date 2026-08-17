
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 🧠 AI TRAINING LAB — Teach it. Test it. Fool it.
# Classroom game for B.Des Product Design / Interaction Design
#
# Architecture:
#   image -> pretrained MobileNetV2 visual features
#          -> lightweight KNN classifier
#
# The deep network is NOT retrained after each example.
# Student labels teach a small classifier using the learned
# visual features. This is a fast, honest transfer-learning demo.
# ============================================================

st.set_page_config(
    page_title="AI Training Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------- CSS / animation -----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
      radial-gradient(circle at 10% 10%, rgba(124,58,237,.10), transparent 28%),
      radial-gradient(circle at 90% 20%, rgba(14,165,233,.10), transparent 25%),
      linear-gradient(180deg, #fbfbff 0%, #f6f7fb 100%);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 2rem 2.2rem;
    border-radius: 30px;
    background: linear-gradient(135deg, #17132e, #30245e 55%, #5537a8);
    color: white;
    box-shadow: 0 20px 50px rgba(40,30,90,.20);
    margin-bottom: 1.4rem;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.1rem;
    margin: 0;
    letter-spacing: -1.5px;
}

.hero p {
    font-size: 1.15rem;
    opacity: .82;
    margin: .45rem 0 0;
}

.pill {
    display: inline-block;
    padding: .35rem .75rem;
    border-radius: 999px;
    background: rgba(255,255,255,.13);
    font-size: .82rem;
    margin-bottom: .7rem;
}

.score-card {
    background: white;
    border: 1px solid #e8e7ef;
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(30,25,60,.06);
    min-height: 105px;
}

.score-label {
    color: #74717f;
    font-size: .82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-top: .15rem;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    margin: 1.4rem 0 .7rem;
}

.action-card {
    background: white;
    border: 2px solid #ebe9f3;
    border-radius: 24px;
    padding: 1.4rem;
    box-shadow: 0 10px 30px rgba(30,25,60,.06);
}

.action-card:hover {
    border-color: #9b7bea;
}

.prediction {
    padding: 1.5rem;
    border-radius: 24px;
    background: linear-gradient(135deg,#f2edff,#ffffff);
    border: 1px solid #ddd3ff;
    text-align: center;
    animation: pop .45s ease-out;
}

.prediction .emoji {
    font-size: 3rem;
}

.prediction .label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 800;
}

.prediction .confidence {
    font-size: 1rem;
    color: #6d5aa8;
    font-weight: 700;
}

@keyframes pop {
    0% { transform: scale(.94); opacity: 0; }
    70% { transform: scale(1.02); }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes pulse {
    0%,100% { transform: scale(1); }
    50% { transform: scale(1.035); }
}

.learn-card {
    padding: 1rem;
    border-radius: 18px;
    background: #fff;
    border: 1px solid #e7e4ee;
    text-align: center;
}

.memory-dot {
    font-size: 1.25rem;
    letter-spacing: 3px;
}

.challenge {
    padding: 1.3rem;
    border-radius: 24px;
    background: linear-gradient(135deg,#fff3df,#fff);
    border: 2px solid #ffd98f;
}

.progress-track {
    height: 12px;
    background: #e8e7ef;
    border-radius: 999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg,#7c3aed,#06b6d4);
    border-radius: 999px;
    transition: width .6s ease;
}

.small-note {
    color: #777381;
    font-size: .9rem;
}

.stButton > button {
    border-radius: 15px !important;
    min-height: 48px !important;
    font-weight: 800 !important;
    border: 1px solid #dedbe8 !important;
    transition: all .18s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(40,30,90,.12);
}

div[data-testid="stFileUploader"] {
    border-radius: 18px;
}

@media (max-width: 700px) {
    .hero h1 { font-size: 2.25rem; }
    .hero { padding: 1.4rem; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------- State -----------------------
state_defaults = {
    "score": 0,
    "correct": 0,
    "wrong": 0,
    "attempts": 0,
    "training_features": [],
    "training_labels": [],
    "round_complete": set(),
}
for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------- Models -----------------------
@st.cache_resource(show_spinner="🧠 Loading the pretrained vision model...")
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

@st.cache_resource(show_spinner="🤖 Loading the ImageNet prediction model...")
def load_imagenet():
    return MobileNetV2(weights="imagenet")

extractor = load_feature_extractor()

def feature_from_image(image):
    img = image.convert("RGB").resize((224,224))
    arr = np.asarray(img).astype("float32")
    arr = preprocess_input(np.expand_dims(arr, 0))
    return extractor.predict(arr, verbose=0)[0]

def imagenet_predict(image):
    model = load_imagenet()
    img = image.convert("RGB").resize((224,224))
    arr = np.asarray(img).astype("float32")
    arr = preprocess_input(np.expand_dims(arr, 0))
    pred = model.predict(arr, verbose=0)
    return [
        (name.replace("_"," ").title(), float(score))
        for _, name, score in decode_predictions(pred, top=5)[0]
    ]

def classroom_model():
    labels = st.session_state.training_labels
    if len(labels) < 2 or len(set(labels)) < 2:
        return None, None
    X = np.asarray(st.session_state.training_features)
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    clf = KNeighborsClassifier(
        n_neighbors=min(3, len(X)),
        weights="distance"
    )
    clf.fit(X, y)
    return clf, enc

def classroom_predict(image):
    clf, enc = classroom_model()
    if clf is None:
        return None
    feat = feature_from_image(image).reshape(1,-1)
    probs = clf.predict_proba(feat)[0]
    best = int(np.argmax(probs))
    return {
        "label": str(enc.inverse_transform([best])[0]),
        "confidence": float(probs[best]),
        "probabilities": {
            str(enc.inverse_transform([i])[0]): float(probs[i])
            for i in range(len(probs))
        }
    }

def add_example(image, label):
    st.session_state.training_features.append(feature_from_image(image))
    st.session_state.training_labels.append(label.strip().lower())

def reset_all():
    st.session_state.score = 0
    st.session_state.correct = 0
    st.session_state.wrong = 0
    st.session_state.attempts = 0
    st.session_state.training_features = []
    st.session_state.training_labels = []
    st.session_state.round_complete = set()
    st.rerun()

def flash_success(message, points):
    st.balloons()
    st.success(f"✨ {message}  **+{points} points**")

# ----------------------- Hero -----------------------
st.markdown("""
<div class="hero">
  <div class="pill">CLASSROOM EXPERIMENT • AI + ML + DL</div>
  <h1>🧠 AI TRAINING LAB</h1>
  <p>Teach it. Test it. Fool it.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------- Scoreboard -----------------------
s1,s2,s3,s4 = st.columns(4)
cards = [
    ("🏆 CLASS SCORE", st.session_state.score),
    ("🎯 AI CORRECT", st.session_state.correct),
    ("🕵️ AI CAUGHT", st.session_state.wrong),
    ("🧪 ATTEMPTS", st.session_state.attempts),
]
for col,(label,value) in zip([s1,s2,s3,s4], cards):
    with col:
        st.markdown(
            f'<div class="score-card"><div class="score-label">{label}</div>'
            f'<div class="score-number">{value}</div></div>',
            unsafe_allow_html=True
        )

# ----------------------- Navigation -----------------------
st.markdown('<div class="section-title">🎮 Choose your mission</div>', unsafe_allow_html=True)

missions = [
    ("🟢", "MEET THE AI", "See what a pretrained AI thinks.", "meet"),
    ("🟡", "BEAT THE AI", "Guess first. Reveal later.", "beat"),
    ("🔴", "TEACH THE AI", "Give it labelled examples.", "teach"),
    ("🔵", "FINAL EXAM", "Test it on unseen images.", "exam"),
    ("😈", "FOOL THE AI", "Find a confident mistake.", "fool"),
]
mission_names = [f"{e} {title}" for e,title,_,_ in missions]
selected = st.radio(
    "Mission",
    mission_names,
    horizontal=True,
    label_visibility="collapsed"
)
mode = missions[mission_names.index(selected)][3]

# ----------------------- Image input -----------------------
st.markdown('<div class="section-title">📸 Give the AI an image</div>', unsafe_allow_html=True)

input_choice = st.radio(
    "Choose your input",
    ["📁 UPLOAD", "📷 CAMERA"],
    horizontal=True,
    label_visibility="collapsed"
)

if input_choice == "📁 UPLOAD":
    source = st.file_uploader(
        "Drop an image here",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed"
    )
else:
    source = st.camera_input("Capture an image")

if source is None:
    st.markdown("""
    <div class="action-card">
      <h3>👋 Ready?</h3>
      <p>Upload an image or use your phone camera. Then choose a mission above.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

image = Image.open(io.BytesIO(source.getvalue())).convert("RGB")

img_col, game_col = st.columns([.9,1.1])
with img_col:
    st.image(image, use_container_width=True)
    st.caption("Your image")

with game_col:

    # ========================================================
    # MEET
    # ========================================================
    if mode == "meet":
        st.markdown('<div class="action-card">', unsafe_allow_html=True)
        st.markdown("### 🔮 ASK THE AI")
        st.write("This is the original pretrained ImageNet model.")
        if st.button("✨ TAP TO REVEAL AI'S GUESS", type="primary", use_container_width=True):
            with st.spinner("🔍 Analysing → 🧠 finding patterns → 🤖 guessing..."):
                time.sleep(.5)
                results = imagenet_predict(image)
            st.session_state.attempts += 1
            st.markdown(
                f'<div class="prediction"><div class="emoji">🤖</div>'
                f'<div class="label">{results[0][0]}</div>'
                f'<div class="confidence">{results[0][1]*100:.1f}% confident</div></div>',
                unsafe_allow_html=True
            )
            df = pd.DataFrame({
                "Prediction": [x[0] for x in results],
                "Confidence": [x[1]*100 for x in results]
            }).set_index("Prediction")
            st.bar_chart(df)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # BEAT
    # ========================================================
    elif mode == "beat":
        st.markdown('<div class="action-card">', unsafe_allow_html=True)
        st.markdown("### 🧑‍🎨 YOU GO FIRST")
        guess = st.text_input(
            "Your prediction",
            placeholder="I think this is...",
            label_visibility="collapsed"
        )
        if st.button("🔮 REVEAL AI", type="primary", use_container_width=True):
            results = imagenet_predict(image)
            st.session_state.attempts += 1
            st.markdown(f"### 🧑 You: **{guess or 'No guess'}**")
            st.markdown(f"### 🤖 AI: **{results[0][0]}**")
            st.progress(results[0][1], text=f"AI confidence: {results[0][1]*100:.1f}%")
            st.write("### Did AI get it right?")
            yes,no = st.columns(2)
            with yes:
                if st.button("🎉 YES — AI GOT IT", use_container_width=True):
                    st.session_state.correct += 1
                    st.session_state.score += 10
                    flash_success("AI got it!",10)
            with no:
                if st.button("😈 NOPE — I CAUGHT IT", use_container_width=True):
                    st.session_state.wrong += 1
                    st.session_state.score += 20
                    flash_success("AI caught! You are an AI detective.",20)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # TEACH
    # ========================================================
    elif mode == "teach":
        st.markdown('<div class="action-card">', unsafe_allow_html=True)
        st.markdown("### 👩‍🏫 TEACH THE AI")
        st.write("What is this actually?")
        label = st.text_input(
            "Correct label",
            placeholder="e.g. water bottle",
            label_visibility="collapsed"
        )
        if st.button("🧠 TEACH IT!", type="primary", use_container_width=True):
            if not label.strip():
                st.error("Give the image a label first.")
            else:
                add_example(image, label)
                st.session_state.score += 5
                st.success(f"✨ AI memory updated: **{label.strip()}**  +5 points")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        labels = st.session_state.training_labels
        if labels:
            st.markdown("### 🧠 CLASSROOM MEMORY")
            counts = pd.Series(labels).value_counts()
            cols = st.columns(min(4, len(counts)))
            for i,(name,count) in enumerate(counts.items()):
                with cols[i % len(cols)]:
                    dots = "● " * min(count,8)
                    st.markdown(
                        f'<div class="learn-card"><b>{name.title()}</b>'
                        f'<div class="memory-dot">{dots}</div>'
                        f'<small>{count} example(s)</small></div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No examples yet. Start teaching!")

        st.caption(
            "Technical note: the pretrained deep network is used as a feature extractor. "
            "A small KNN classifier learns the classroom labels."
        )

    # ========================================================
    # EXAM
    # ========================================================
    elif mode == "exam":
        st.markdown('<div class="action-card">', unsafe_allow_html=True)
        st.markdown("### 🧪 FINAL EXAM")
        st.write("No hints. No labels. Let's see what the classroom AI learned.")
        result = classroom_predict(image)
        if result is None:
            st.warning("Teach at least TWO different categories first.")
        else:
            st.markdown(
                f'<div class="prediction"><div class="emoji">🤖</div>'
                f'<div class="label">{result["label"].title()}</div>'
                f'<div class="confidence">{result["confidence"]*100:.1f}% confident</div></div>',
                unsafe_allow_html=True
            )
            st.write("### Your verdict?")
            yes,no = st.columns(2)
            with yes:
                if st.button("🎯 CORRECT! +10", use_container_width=True):
                    st.session_state.correct += 1
                    st.session_state.attempts += 1
                    st.session_state.score += 10
                    flash_success("The AI generalised!",10)
            with no:
                if st.button("🕵️ WRONG! +10", use_container_width=True):
                    st.session_state.wrong += 1
                    st.session_state.attempts += 1
                    st.session_state.score += 10
                    st.warning("🕵️ AI detective point earned! Now ask why it failed.")
            probs = pd.DataFrame({
                "Class": list(result["probabilities"].keys()),
                "Score": [v*100 for v in result["probabilities"].values()]
            }).set_index("Class")
            st.bar_chart(probs)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # FOOL
    # ========================================================
    else:
        st.markdown('<div class="challenge">', unsafe_allow_html=True)
        st.markdown("### 😈 FOOL THE AI")
        st.write("Mission: make the AI confidently wrong.")
        result = classroom_predict(image)
        if result is None:
            st.warning("Teach at least two classes first.")
        else:
            st.markdown(
                f'<div class="prediction"><div class="emoji">😎</div>'
                f'<div class="label">{result["label"].title()}</div>'
                f'<div class="confidence">{result["confidence"]*100:.1f}% confident</div></div>',
                unsafe_allow_html=True
            )
            if result["confidence"] >= .8:
                st.warning("🔥 HIGH CONFIDENCE — can you prove it wrong?")
            else:
                st.info("AI is uncertain. What makes this image difficult?")
            actual = st.text_input(
                "Correct label if AI is wrong",
                placeholder="What is it really?",
                label_visibility="collapsed"
            )
            if st.button("🕵️ I FOOLED THE AI!", type="primary", use_container_width=True):
                if actual.strip():
                    st.session_state.score += 20
                    st.session_state.wrong += 1
                    st.session_state.attempts += 1
                    st.balloons()
                    st.success(
                        f"🚨 CONFIDENCE TRAP! AI said **{result['label']}**, "
                        f"but the correct answer is **{actual.strip()}**. +20"
                    )
                else:
                    st.error("Tell us what the image really is.")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Learning reveal -----------------------
st.divider()
st.markdown("### 🎓 You just experienced AI without a lecture")

learn = [
    ("📸", "DATA", "Images students provide"),
    ("🏷️", "LABELLING", "Correct answers students give"),
    ("🧠", "DEEP LEARNING", "Pretrained visual feature extractor"),
    ("🤖", "MACHINE LEARNING", "Small classifier learns classroom categories"),
    ("🧪", "TESTING", "New images not used for teaching"),
    ("🎯", "EVALUATION", "Correct vs wrong"),
    ("😈", "LIMITATIONS", "Finding failure cases"),
]
cols = st.columns(4)
for i,(emoji,title,desc) in enumerate(learn):
    with cols[i % 4]:
        st.markdown(
            f'<div class="learn-card"><div style="font-size:2rem">{emoji}</div>'
            f'<b>{title}</b><br><small>{desc}</small></div>',
            unsafe_allow_html=True
        )

# ----------------------- Reset -----------------------
st.divider()
if st.button("♻️ Reset entire classroom experiment"):
    reset()
