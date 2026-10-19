"""CampusVoice: local-first student feedback intelligence dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.aspects import detect_aspects
from src.data_loader import manual_feedback, read_feedback_csv
from src.evaluation import compare_models
from src.insights import generate_insights
from src.reporting import markdown_report
from src.sentiment import predict_classifier, train_classifier, vader_score
from src.topic_modeling import discover_topics

ROOT = Path(__file__).parent
DEMO_PATH = ROOT / "data" / "demo_feedback.csv"
st.set_page_config(page_title="CampusVoice", page_icon="🎓", layout="wide")

st.title("🎓 CampusVoice")
st.caption("Student Feedback Intelligence System · local NLP for evidence-led review")
st.info("Privacy: feedback is processed in this Python session. No API or external service is used. Avoid uploading names, roll numbers, emails, or other identifying details.")

with st.sidebar:
    st.header("Feedback input")
    uploaded = st.file_uploader("Upload feedback CSV", type=["csv"], help="Required: comment. Optional: feedback_id, department, semester, date.")
    load_demo = st.button("Load synthetic demo data", use_container_width=True)
    st.caption("Demo rows are synthetic and are not real college feedback.")
    st.divider()
    st.subheader("Add one comment")
    with st.form("manual"):
        one_comment = st.text_area("Comment")
        dept = st.text_input("Department (optional)")
        sem = st.text_input("Semester (optional)")
        add_comment = st.form_submit_button("Analyze comment")
    st.divider()
    st.subheader("Sentiment approach")
    method = st.radio("Predictions for feedback", ["VADER baseline", "Logistic Regression"], help="The trained model is for demonstration unless trained and checked on representative labelled data.")

if "feedback" not in st.session_state:
    st.session_state.feedback = pd.DataFrame()
if load_demo:
    st.session_state.feedback, st.session_state.ingest = read_feedback_csv(DEMO_PATH)
    st.session_state.labels = pd.read_csv(DEMO_PATH)["sentiment"].tolist()
if uploaded is not None:
    try:
        st.session_state.feedback, st.session_state.ingest = read_feedback_csv(uploaded)
        st.session_state.labels = None
    except ValueError as exc:
        st.error(str(exc))
if add_comment:
    if not one_comment.strip():
        st.sidebar.error("Enter a comment before analysis.")
    else:
        new = manual_feedback(one_comment, dept, sem)
        st.session_state.feedback = pd.concat([st.session_state.feedback, new], ignore_index=True).drop_duplicates("comment")
        st.session_state.labels = None

base = st.session_state.feedback.copy()
if base.empty:
    st.markdown("### Start with feedback")
    st.write("Load the synthetic demonstration data or upload a CSV with a `comment` column. You can also add one comment in the sidebar.")
    st.code("comment,department,semester\nThe laboratory equipment needs maintenance,Engineering,4", language="csv")
    st.stop()

if "ingest" in st.session_state:
    stats = st.session_state.ingest
    st.caption(f"Input rows: {stats['input_rows']} · accepted: {stats['accepted_rows']} · skipped blank: {stats['missing_comments']} · skipped duplicates: {stats['duplicate_comments']}")

texts = base["comment"].astype(str).tolist()
trained_model = None
if method == "Logistic Regression":
    labelled = None
    if "sentiment" in base.columns:
        labelled = base["sentiment"].astype(str).str.title().tolist()
        base = base.drop(columns=["sentiment"])
    elif "labels" in st.session_state and st.session_state.labels and len(st.session_state.labels) == len(base):
        labelled = st.session_state.labels
    if labelled:
        try:
            trained_model = train_classifier(texts, labelled)
            preds = predict_classifier(trained_model, texts)
        except ValueError as exc:
            st.warning(f"Could not train classifier: {exc}. Using VADER for this view.")
            preds = [vader_score(t) for t in texts]
    else:
        st.warning("No labelled training CSV is loaded. Upload labelled data below to train a session-only classifier; falling back to VADER for current comments.")
        preds = [vader_score(t) for t in texts]
else:
    preds = [vader_score(t) for t in texts]

analysis = base.copy()
analysis["sentiment"] = [p["sentiment"] for p in preds]
analysis["confidence"] = [p.get("confidence") for p in preds]
aspect_rows = [detect_aspects(t) for t in texts]
analysis["aspects"] = [", ".join(a["aspect"] for a in row) for row in aspect_rows]
analysis["aspect_evidence"] = ["; ".join(f"{a['aspect']} [{a['sentiment']}]: {a['evidence']}" for a in row) for row in aspect_rows]
analysis["topic_id"] = None
topics = discover_topics(texts, analysis["sentiment"].tolist())
if topics:
    # NMF assignments reproduced from representative membership are deliberately not implied in explorer;
    # topic counts and examples below are the fitted discovery output.
    pass
insights = generate_insights(analysis.assign(aspects=analysis["aspects"].str.split(", ")))

overview_tab, explorer_tab, topics_tab, actions_tab, models_tab, export_tab, method_tab = st.tabs(["Overview", "Feedback explorer", "Topic analysis", "Actionable insights", "Model comparison", "Export", "Methodology"])
with overview_tab:
    cols = st.columns(4)
    counts = analysis["sentiment"].value_counts()
    cols[0].metric("Comments analysed", len(analysis))
    for col, label in zip(cols[1:], ["Positive", "Negative", "Neutral"]):
        col.metric(label, int(counts.get(label, 0)))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.pie(names=counts.index, values=counts.values, hole=.45, title="Sentiment distribution"), use_container_width=True)
    with right:
        cat = analysis["aspects"].str.split(", ").explode().value_counts().rename_axis("Category").reset_index(name="Comments")
        st.plotly_chart(px.bar(cat, x="Comments", y="Category", orientation="h", title="Mentioned categories"), use_container_width=True)
    st.subheader("Most common negative topics")
    if topics:
        st.dataframe(pd.DataFrame([{k: t[k] for k in ["name", "count", "negative_count", "negative_proportion"]} for t in topics]).sort_values("negative_count", ascending=False), hide_index=True, use_container_width=True)
    else:
        st.caption("Add at least four varied comments to attempt topic discovery.")

with explorer_tab:
    c1, c2, c3, c4 = st.columns(4)
    query = c1.text_input("Search comments")
    sentiments = c2.multiselect("Sentiment", sorted(analysis.sentiment.unique()), default=sorted(analysis.sentiment.unique()))
    categories = sorted(set(x for row in analysis.aspects.str.split(", ") for x in row))
    cats = c3.multiselect("Category", categories, default=categories)
    depts = [x for x in analysis.department.dropna().astype(str).unique()] if "department" in analysis else []
    selected_depts = c4.multiselect("Department", depts, default=depts)
    filtered = analysis[analysis.sentiment.isin(sentiments) & analysis.aspects.apply(lambda x: bool(set(x.split(", ")) & set(cats))) & analysis.comment.str.contains(query, case=False, na=False)]
    if depts:
        filtered = filtered[filtered.department.astype(str).isin(selected_depts)]
    if "semester" in analysis and analysis.semester.notna().any():
        semesters = [x for x in analysis.semester.dropna().unique()]
        picked = st.multiselect("Semester", semesters, default=semesters)
        filtered = filtered[filtered.semester.isna() | filtered.semester.isin(picked)]
    st.dataframe(filtered, hide_index=True, use_container_width=True)

with topics_tab:
    if not topics:
        st.warning("There are too few comments or too little vocabulary for stable topics. Try a larger and more varied dataset.")
    else:
        for i, topic in enumerate(topics):
            title = st.text_input(f"Topic {i+1} label (interpret or rename)", value=str(topic["name"]), key=f"topic_{i}")
            st.markdown(f"**{title}** · {topic['count']} comments · negative share {topic['negative_proportion']:.0%}")
            st.caption("Key terms: " + ", ".join(topic["terms"]))
            for comment in topic["representatives"]:
                st.write("• " + comment)

with actions_tab:
    st.caption("Suggestions are transparent prompts for human review, not proven conclusions.")
    if not insights:
        st.info("No category has at least two negative mentions yet.")
    for item in insights:
        with st.container(border=True):
            st.markdown(f"**{item['aspect']}** · {item['negative_count']} negative comments / {item['total_count']} mentions ({item['negative_proportion']:.0%})")
            st.write(item["suggestion"])
            st.caption("Supporting comments: " + " | ".join(item["evidence"]))

with models_tab:
    st.write("VADER uses a sentiment lexicon and rules. Compound scores ≥ 0.05 are Positive, ≤ −0.05 Negative, and between them Neutral. Logistic Regression learns word and phrase weights from labelled examples using TF-IDF.")
    st.caption("Model confidence is the classifier's maximum class probability; it is not a guarantee of correctness. VADER displays absolute compound score, not a calibrated probability.")
    engines = sorted({str(p.get("engine", "Logistic Regression")) for p in preds})
    if any("fallback" in engine for engine in engines):
        st.warning("The NLTK VADER lexicon is not installed and this offline session used the small built-in fallback lexicon. Install the NLTK VADER lexicon for normal VADER analysis; fallback scores are not VADER results.")
    st.subheader("Train/evaluate on a labelled CSV")
    labelled_file = st.file_uploader("Labelled CSV (comment, sentiment)", type="csv", key="labels_csv")
    if labelled_file:
        try:
            labelled_df = pd.read_csv(labelled_file)
            if not {"comment", "sentiment"}.issubset(labelled_df.columns):
                raise ValueError("Required columns are comment and sentiment.")
            labelled_df = labelled_df.dropna(subset=["comment", "sentiment"])
            normalized_labels = labelled_df.sentiment.astype(str).str.title()
            if not set(normalized_labels).issubset({"Positive", "Negative", "Neutral"}):
                raise ValueError("Sentiment values must be Positive, Negative, or Neutral.")
            result = compare_models(labelled_df.comment.astype(str).tolist(), normalized_labels.tolist())
            st.success(f"Stratified evaluation completed: {result['train_size']} train / {result['test_size']} test. This result reflects only this supplied dataset.")
            for key in ["vader", "logistic_regression"]:
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                metrics = result[key]
                st.json({k: metrics[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class"]})
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(metrics["confusion_matrix"], display_labels=metrics["labels"]).plot(ax=ax, colorbar=False)
                st.pyplot(fig)
            st.warning("Synthetic-data performance does not establish real-world accuracy. Report representative consented evaluation separately.")
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    elif "sentiment" in pd.read_csv(DEMO_PATH).columns:
        st.caption("The bundled synthetic demo labels allow an illustrative stratified comparison when demo data is active; upload your own labelled CSV to evaluate your data.")
        demo_df = pd.read_csv(DEMO_PATH)
        if st.button("Evaluate synthetic demo split"):
            try:
                result = compare_models(demo_df.comment.tolist(), demo_df.sentiment.tolist())
                st.warning("Synthetic demonstration only — not validated on real student feedback.")
                st.json({k: result[k]["macro_f1"] for k in ["vader", "logistic_regression"]})
            except ValueError as exc:
                st.error(str(exc))

with export_tab:
    st.download_button("Download filtered feedback and analysis CSV", filtered.to_csv(index=False).encode("utf-8"), "campusvoice_analysis.csv", "text/csv")
    report = markdown_report(analysis, topics, insights, "Original feedback is retained. VADER uses compound thresholds (+/-0.05); optional Logistic Regression uses TF-IDF unigrams/bigrams. Aspect categories use editable keyword phrase rules. NMF discovers recurring word patterns.")
    st.download_button("Download Markdown summary report", report, "campusvoice_report.md", "text/markdown")

with method_tab:
    st.markdown("""- **Sentiment:** VADER is a transparent lexicon baseline. Logistic Regression uses TF-IDF features and must be trained/evaluated on labelled examples.
- **Aspects:** phrase rules can detect multiple categories; VADER estimates polarity on each matching clause and retains it as evidence. Extend `ASPECT_KEYWORDS` in `src/aspects.py`; rule matching and clause polarity can miss paraphrases and context.
- **Topics:** NMF decomposes comment TF-IDF vectors into word patterns. Labels are top terms, not guaranteed semantic categories; interpret or rename them manually.
- **Preprocessing:** whitespace is normalized, original comments remain unchanged, negation is retained for tokenization, and stop-word removal is limited to topics. No stemming is used.
- **Privacy and responsible use:** use consented, anonymized feedback; review small group disclosures before sharing. Do not use automated output to rank or penalize individual staff or students.
- **Real evaluation:** use an independently held, representative labelled test set; check class balance and report class-wise precision/recall/F1. Split before fitting vectorization to avoid leakage.""")
