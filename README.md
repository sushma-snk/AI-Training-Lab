# AI Training Lab

Interactive Streamlit classroom demo: **Teach it. Test it. Fool it.**

## Technical approach
MobileNetV2 pretrained on ImageNet is used as a feature extractor. Student-labelled examples are stored in session memory. A small KNN classifier learns the classroom labels from those features.

The deep neural network itself is not retrained after every student example. This makes the classroom demo fast and technically honest.

## Run
Recommended Python: 3.12.

```bash
py -3.12 -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
Select Python 3.12 in Advanced settings. Main file: `app.py`.

## Classroom sequence
1. Meet the AI
2. Beat the AI
3. Teach the AI
4. Final Exam
5. Fool the AI

Students experience data, labelling, transfer learning, machine learning, testing, evaluation, confidence, generalisation and model limitations.
