# CampusVoice AI — Student Feedback Intelligence

CampusVoice is a local-first Streamlit dashboard for exploring student feedback. It preserves CSV metadata, displays VADER predictions for unlabelled feedback, and can train a session-only TF-IDF + Logistic Regression classifier from a separate human-labelled dataset. Data is not sent to an external API.

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the project checks with `python -m pytest -q`.

## Dashboard pages

Use the **Workspace** menu in the sidebar to open Executive Overview, Sentiment Analytics, Aspect Analysis, Topic Discovery, Department Comparison, Feedback Explorer, Actionable Insights, Train Sentiment Model & Evaluation, Reports & Export, or Methodology. Upload feedback or load the synthetic dashboard demo from the sidebar; filters apply across the analysis pages and exports.

## Feedback CSV schema

The required column is `comment`. The rich schema below is accepted as-is; every supplied field is retained for preview, filters, exploration, and export. All metadata fields are optional.

`feedback_id,comment,department,program,semester,academic_year,course,category_hint,date,source`

An optional `sentiment` column is reserved for verified labels: `Positive`, `Negative`, or `Neutral`. Case and whitespace variations are normalized; `pos`, `neg`, and `neu` are also accepted. `category_hint` is only a topic/category field and is never interpreted as sentiment. Blank comments are skipped. Repeated comment text is retained and reported because it may represent independent feedback. The sidebar's **Remove exact duplicate rows** option removes only records identical across all supplied fields; repeated feedback IDs are reported for review. Invalid dates are reported and set to empty while retaining the comment row. CSVs without labels remain usable for analysis.

## Analyze unlabelled feedback

Upload the rich CSV from the sidebar. CampusVoice predicts sentiment with the VADER baseline (or its explicitly identified small offline fallback if the NLTK VADER lexicon is unavailable). When a session model has been trained, it can be selected in the sidebar and applied to this unlabelled dataset. Predictions are model outputs, not ground truth. The sidebar's **Load synthetic demo** button loads unlabelled dashboard feedback; it is separate from the labelled training demo.

## Train a supervised model

Open **Train Sentiment Model & Evaluation**. Upload a separate CSV with `comment,sentiment`; extra metadata is allowed. An optional `topic_group` column keeps related examples together during a reproducible stratified holdout. Without groups, evaluation uses an 80/20 stratified split; with groups, whole groups are assigned to one partition, so the exact ratio depends on group sizes. Every class must appear in both partitions. Missing groups, insufficient data, or impossible group/class splits make evaluation unavailable and show a warning; the app does not silently switch to a row-level split. Exact duplicate comments are removed before evaluation; conflicting duplicate labels/groups are rejected. The vectorizer and classifier used for holdout metrics are fitted only on the training partition. After evaluation, a separate active model is fitted on all labelled rows and retained in the current Streamlit session for subsequent feedback predictions. Changing the training file marks the prior model stale until retrained.

The page also provides a downloadable labelled template and an optional 300-row balanced synthetic demonstration set covering ten feedback topics. Use **Load Labelled Demo Dataset** on the training page to load it. Synthetic examples are fictional and are not evidence of real-world model performance. A model can be trained with two or more classes even when the sample is too small for valid holdout metrics; in that case the app explains why evaluation is unavailable rather than inventing scores. One-class training data is rejected. Small held-out sets are explicitly flagged.

## Reading evaluation results

Accuracy is the share of held-out predictions that match labels. Per-class precision measures how many predictions of that class were correct; recall measures how many true examples of that class were found; F1 balances precision and recall. Macro F1 weights classes equally. The confusion matrix shows actual labels by predicted labels. These values describe only the supplied held-out split, not future cohorts. VADER and Logistic Regression comparisons use the same test comments. Do not treat classifier scores as calibrated probabilities or synthetic-data metrics as real validation.

For meaningful claims, collect consented and anonymized feedback, create human-verified labels using a documented process, and evaluate on a representative set kept separate from training. Review the methodology page before sharing results; the system is decision support, not a tool to rank or penalize individual students or staff.

## Modules

- `src/data_loader.py`: schema preservation, comment validation, date and label normalization.
- `src/sentiment.py`: VADER and TF-IDF + Logistic Regression pipeline.
- `src/evaluation.py`: reproducible stratified or group-aware held-out evaluation and classification metrics.
- `src/aspects.py`: editable phrase-based aspect detection.
- `src/topic_modeling.py`: TF-IDF and NMF topic exploration.
- `src/insights.py`, `src/reporting.py`: evidence-linked review prompts and exports.
- `app.py`: Streamlit dashboard and session workflow.
- `data/demo_labeled_training.csv`: clearly synthetic labelled training demonstration.
