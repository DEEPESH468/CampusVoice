# CampusVoice — Student Feedback Intelligence System

CampusVoice is a local-first Streamlit PBL application that turns comments into sentiment labels, explainable service categories, recurring NMF topics, and evidence-linked suggestions for human review. It does not send data to an API. It is a decision-support demonstration, not a validated institutional measurement system.

## Run

Python 3.11 is recommended (3.10+ should work). From this directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Click **Load synthetic demo data** in the sidebar to start the demonstration. The bundled rows and labels are synthetic; they are not real student feedback. To upload feedback, supply a CSV with a `comment` column; `feedback_id`, `department`, `semester`, and `date` are optional. Blank comments and exact duplicate comments are skipped and counted.

For standard VADER scores, make sure the NLTK `vader_lexicon` resource is installed (`python -c "import nltk; nltk.download('vader_lexicon')"`). In an offline installation without it, CampusVoice clearly identifies and uses a small built-in fallback lexicon; its outputs are not VADER scores.

## Train and evaluate

Upload a labelled CSV under Model comparison with `comment,sentiment` columns; sentiment values must be Positive, Neutral, or Negative. CampusVoice uses a stratified holdout split (25%, fixed random seed), fits TF-IDF and Logistic Regression only on training comments, and compares it with VADER on the same test comments. At least eight rows and two examples per class are required by the safe split check. The split can be rerun reproducibly via `src.evaluation.compare_models`. Synthetic results demonstrate the mechanics only; do not claim real-world accuracy from them. For genuine evaluation, obtain consent, anonymize comments, sample across departments and terms, label with a documented process, and keep the test set untouched until final evaluation.

To train a session-only classifier for feedback analysis, choose **Logistic Regression** and upload a labelled CSV in the Model comparison tab; a labelled input dataset can also be used directly where a sentiment column is included. Do not deploy or share its predictions as validated until representative evaluation is complete.

Run tests with:

```bash
python -m pytest
```

## Modules

- `src/data_loader.py`: CSV checks, blank and duplicate handling.
- `src/preprocessing.py`: normalization and tokenization (negation preserved).
- `src/sentiment.py`: VADER and TF-IDF + Logistic Regression.
- `src/evaluation.py`: stratified evaluation and classification metrics.
- `src/aspects.py`: editable phrase maps and evidence phrases.
- `src/topic_modeling.py`: TF-IDF and NMF topic discovery.
- `src/insights.py`: transparent suggestions tied to observed negative mentions.
- `src/reporting.py`: Markdown export.
- `app.py`: Streamlit user interface.

See `docs/architecture.md`, `docs/project_report.md`, `docs/viva_questions.md`, `docs/presentation_outline.md`, and `TEAM_CONTRIBUTIONS.md` for the PBL materials.
