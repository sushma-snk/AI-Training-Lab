# 🧠 AI Training Lab — Teach It. Test It. Fool It.

A playful classroom Streamlit app for a first Digital Fluency session with B.Des Product Design and Interaction Design students.

## The classroom idea

Students do not sit through a long AI lecture. They interact with an AI, challenge it, teach it, test it and try to fool it.

### Missions

1. 🟢 **Meet the AI** — see what a pretrained ImageNet model predicts.
2. 🟡 **Beat the AI** — students guess first, then reveal the AI prediction.
3. 🔴 **Teach the AI** — students provide labelled examples.
4. 🔵 **Final Exam** — test the classroom-trained classifier on new images.
5. 😈 **Fool the AI** — find confident failure cases.

Images can be provided by **Upload** or **Camera**.

## How the learning system works

The app does NOT retrain the full deep neural network after every student example.

Instead:

```text
Student image
     ↓
Pretrained MobileNetV2
     ↓
Visual feature vector
     ↓
KNN classifier
     ↓
Student-defined classroom category
```

This makes the learning interaction fast enough for a live classroom.

The educational explanation is:

> "The deep model has already learned general visual patterns. We are using those learned features and teaching a smaller model how our classroom categories look."

This is a simplified demonstration of **transfer learning**.

## Important limitation

The classroom examples are kept in Streamlit session memory. They are not a permanent shared dataset. If the app session restarts, the classroom memory resets.

This is intentional for a simple first-session demo and avoids permanently storing student images.

## Local installation

Use **Python 3.12**.

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Community Cloud

When deploying:

- Repository: your GitHub repository
- Branch: `main`
- Main file: `app.py`
- Python: **3.12** in Advanced settings

The first startup may take longer because TensorFlow and the pretrained MobileNetV2 weights need to be loaded.

## Suggested 3-hour classroom flow

### 1. Meet the AI — 10–15 min
Ask students to predict what AI will say.

### 2. Beat the AI — 15–20 min
Students guess first. Award points for spotting mistakes.

### 3. Teach the AI — 30–45 min
Groups collect examples of a few classroom categories.

Suggested categories:
- bottle
- shoe
- bag
- phone
- cup
- notebook

Use different viewpoints, lighting and backgrounds.

### 4. Final Exam — 15–20 min
Use new images. Ask whether AI generalised.

### 5. Fool the AI — 15–20 min
Students deliberately search for unusual images and confident mistakes.

### 6. Reveal the concepts — 15–20 min

Connect the game to:

```text
DATA
 ↓
LABELLING
 ↓
TRAINING
 ↓
MODEL
 ↓
PREDICTION
 ↓
TESTING
 ↓
EVALUATION
 ↓
LIMITATIONS
```

Then introduce:

**AI → Machine Learning → Deep Learning**

without equations.

## Why this is suitable for design students

The activities focus on:
- observation
- visual patterns
- human vs AI judgement
- ambiguity
- interaction
- feedback
- model limitations
- designing better examples

These are easier entry points into AI for students who do not need mathematical depth in the first session.
