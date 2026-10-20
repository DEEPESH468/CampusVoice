"""CampusVoice AI: local-first student feedback analytics dashboard."""
from __future__ import annotations

from pathlib import Path

from hashlib import sha256
from io import BytesIO
from importlib import reload
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.aspects import detect_aspects
from src import data_loader
from src.evaluation import compare_models
from src.insights import generate_insights
from src.reporting import markdown_report
from src.sentiment import predict_classifier, train_classifier, vader_score
from src.topic_modeling import discover_topics

ROOT = Path(__file__).resolve().parent
data_loader = reload(data_loader)
EXPECTED_DATA_LOADER = (ROOT / "src" / "data_loader.py").resolve()
if Path(data_loader.__file__).resolve() != EXPECTED_DATA_LOADER:
    raise ImportError(f"CampusVoice loaded an unexpected data loader: {data_loader.__file__}; expected {EXPECTED_DATA_LOADER}")
DEMO_PATH = ROOT / "data" / "demo_feedback.csv"
COLORS = {"Positive": "#8eae91", "Negative": "#d88778", "Neutral": "#a3a9a4", "Unclassified": "#717873"}
NAV = ["Executive Overview", "Sentiment Analytics", "Aspect Analysis", "Topic Discovery",
       "Department Comparison", "Feedback Explorer", "Actionable Insights", "Train Sentiment Model & Evaluation",
       "Reports & Export", "Methodology"]

st.set_page_config(page_title="CampusVoice AI", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#eeeFE8; --muted:#a4aaa4; --sage:#9ab49c; --panel:#202522; --line:#343b36; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#171b19; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(23,27,25,.92); }
h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em; color:#f0f1eb!important; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; min-height:112px; }
[data-testid="stMetricLabel"] { color:#a4aaa4!important; font-size:.82rem!important; }
[data-testid="stMetricValue"] { color:#eff1eb!important; font-family:'Manrope',sans-serif; }
[data-testid="stSidebar"] { background:#1d221f; border-right:1px solid #303732; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { letter-spacing:-.02em; }
[data-testid="stTabs"] button { color:#aab0aa; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#b9cfba; border-bottom-color:#9ab49c; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; }
[data-testid="stPlotlyChart"] { background:#202522; border:1px solid #343b36; border-radius:12px; padding:8px; }
div.stButton > button, div.stDownloadButton > button { border-radius:8px; border:1px solid #607763; background:#34483a; color:#eff4ed; }
.cv-subtitle { color:#a4aaa4; margin-top:-12px; }
.cv-status { float:right; color:#b9cfba; background:#27332a; border:1px solid #3a5140; padding:6px 12px; border-radius:99px; font-size:.82rem; }
.cv-model-status { float:right; color:#e4e8e1; background:#292f2b; border:1px solid #454d47; padding:6px 12px; border-radius:99px; font-size:.82rem; margin-right:8px; }
.cv-empty { background:#202522; border:1px solid #343b36; border-radius:14px; padding:30px; color:#bdc3bc; }
.cv-warning { background:#242923; border:1px solid #716c4b; border-left:4px solid #b9a77a; border-radius:10px; padding:14px 18px; color:#e8e8dc; margin:12px 0 18px; }
</style>
""", unsafe_allow_html=True)

# Shared chart defaults keep all pages visually consistent.
def style_fig(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=18, r=18, t=54, b=24),
        paper_bgcolor="#202522", plot_bgcolor="#202522", font=dict(family="DM Sans", color="#e8ebe5", size=12),
        title_font=dict(family="Manrope", size=16, color="#eef0e9"), legend_title_text="",
        hoverlabel=dict(bgcolor="#171b19", font_color="#f0f1eb"))
    fig.update_xaxes(gridcolor="#343b36", zerolinecolor="#343b36", linecolor="#343b36")
    fig.update_yaxes(gridcolor="#343b36", zerolinecolor="#343b36", linecolor="#343b36")
    return fig


def chart(frame, **kwargs):
    st.plotly_chart(style_fig(px.bar(frame, **kwargs)), use_container_width=True)


def metric_cols(frame):
    counts = frame.sentiment.value_counts()
    total = len(frame)
    positives = int(counts.get("Positive", 0)); negatives = int(counts.get("Negative", 0)); neutrals = int(counts.get("Neutral", 0))
    aspects = frame.aspect_rows.explode().dropna() if "aspect_rows" in frame else pd.Series(dtype=object)
    aspects = aspects.apply(lambda row: row.get("aspect") if isinstance(row, dict) else None).dropna()
    top_aspect = aspects.value_counts().index[0] if len(aspects) else "—"
    cols = st.columns(6)
    metrics = [("Comments analyzed", f"{total:,}"), ("Positive", f"{positives/total:.1%}" if total else "0%"),
               ("Negative", f"{negatives/total:.1%}" if total else "0%"), ("Neutral", f"{neutrals/total:.1%}" if total else "0%"),
               ("Top aspect", top_aspect), ("Negative comments", f"{negatives:,}")]
    for col, (label, value) in zip(cols, metrics): col.metric(label, value)


def sentiment_donut(frame, title="Sentiment distribution"):
    counts = frame.sentiment.value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0).rename_axis("Sentiment").reset_index(name="Comments")
    fig = px.pie(counts, names="Sentiment", values="Comments", hole=.66, color="Sentiment", color_discrete_map=COLORS,
                 custom_data=[counts.Comments])
    fig.update_traces(textinfo="percent", textfont_size=13, hovertemplate="%{label}<br>%{value:,} comments · %{percent}<extra></extra>")
    fig.update_layout(title=title, showlegend=True, annotations=[dict(text=f"{len(frame):,}<br><span style='font-size:11px'>comments</span>", x=.5, y=.5, showarrow=False, font=dict(size=20, color="#eef0e9"))])
    st.plotly_chart(style_fig(fig), use_container_width=True)


def aspect_frame(frame):
    rows = []
    for _, record in frame.iterrows():
        for item in record.get("aspect_rows", []):
            if item.get("aspect") != "Other or unclassified":
                rows.append({"aspect": item["aspect"], "sentiment": item["sentiment"], "comment": record.comment,
                             "department": record.get("department"), "evidence": item.get("evidence", "")})
    return pd.DataFrame(rows, columns=["aspect", "sentiment", "comment", "department", "evidence"])


def sentiment_over_time(frame):
    if "date" not in frame: return None
    dates = pd.to_datetime(frame.date, errors="coerce")
    valid = frame.loc[dates.notna()].copy()
    if len(valid) < 2: return None
    valid["date"] = pd.to_datetime(valid.date, errors="coerce")
    span = (valid.date.max() - valid.date.min()).days
    valid["period"] = valid.date.dt.to_period("M" if span > 90 else "W" if span > 21 else "D").dt.start_time
    grouped = valid.groupby(["period", "sentiment"]).size().reset_index(name="Comments")
    fig = px.line(grouped, x="period", y="Comments", color="sentiment", markers=True, color_discrete_map=COLORS,
                  title="Sentiment over time", labels={"period":"Date", "Comments":"Comments", "sentiment":"Sentiment"})
    return style_fig(fig, 330)


if "feedback" not in st.session_state: st.session_state.feedback = pd.DataFrame()
if "source_name" not in st.session_state: st.session_state.source_name = "No data loaded"
if st.session_state.pop("activate_trained_model_next_run", False):
    st.session_state["use_trained_model"] = True

with st.sidebar:
    st.markdown("## ◉ CampusVoice AI")
    st.caption("STUDENT FEEDBACK INTELLIGENCE")
    st.divider()
    page = st.radio("Workspace", NAV, label_visibility="collapsed")
    st.divider()
    st.subheader("Your data")
    remove_exact_duplicates = st.checkbox("Remove exact duplicate rows", value=False,
        help="Repeated comment text is retained by default. This option removes rows identical across all supplied fields.")
    uploaded = st.file_uploader("Upload a feedback CSV", type=["csv"], help="Required: comment. Optional: feedback_id, department, semester, date, sentiment.")
    if st.button("Load synthetic demo", use_container_width=True):
        st.session_state.feedback, st.session_state.ingest = data_loader.read_feedback_csv(DEMO_PATH, remove_exact_duplicates=remove_exact_duplicates)
        st.session_state.source_name = "Synthetic demo"
    upload_signature = sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None
    if uploaded is not None and st.session_state.get("loaded_upload_signature") != (upload_signature, remove_exact_duplicates):
        try:
            st.session_state.feedback, st.session_state.ingest = data_loader.read_feedback_csv(uploaded, remove_exact_duplicates=remove_exact_duplicates)
            st.session_state.source_name = uploaded.name
            st.session_state.loaded_upload_signature = (upload_signature, remove_exact_duplicates)
            st.session_state.pop("labels", None)
        except ValueError as exc: st.error(str(exc))
    st.caption("Synthetic demo rows are fictional.")
    with st.expander("Analyze one comment", expanded=False):
        with st.form("manual"):
            one_comment = st.text_area("Comment", height=90)
            dept = st.text_input("Department (optional)")
            sem = st.text_input("Semester (optional)")
            add = st.form_submit_button("Analyze comment", use_container_width=True)
        if add:
            if not one_comment.strip(): st.error("Enter a comment first.")
            else:
                new = data_loader.manual_feedback(one_comment, dept, sem)
                st.session_state.feedback = pd.concat([st.session_state.feedback, new], ignore_index=True)
                st.session_state.source_name = "Current session + manual entry"
    trained_available = st.session_state.get("trained_model") is not None and st.session_state.get("trained_model_valid", False)
    if trained_available:
        st.caption("Model status: **Trained TF-IDF + Logistic Regression**")
    elif st.session_state.get("trained_model") is not None:
        st.caption("Model status: **Previous model stale · VADER active**")
    else:
        st.caption("Model status: **VADER baseline**")
    use_trained = st.checkbox("Use trained model for predictions", value=trained_available, key="use_trained_model", disabled=not trained_available) and trained_available
    st.divider()
    st.caption("Local processing · no external API")

raw = st.session_state.feedback.copy()
active_model_name = "Trained TF-IDF + Logistic Regression" if use_trained else "VADER"
st.markdown(f"<span class='cv-status'>● {st.session_state.source_name}</span><span class='cv-model-status'>Model · {active_model_name}</span>", unsafe_allow_html=True)
st.title("CampusVoice AI")
st.markdown("<div class='cv-subtitle'>A clear view of what students are saying, and where attention may help.</div>", unsafe_allow_html=True)
label_state = "Labelled data" if "sentiment" in raw and raw.sentiment.notna().any() else "Unlabelled feedback"
st.caption(f"Dataset status: **{label_state}** · Prediction source: **{'Trained TF-IDF + Logistic Regression' if use_trained else 'VADER baseline'}**")
if not raw.empty and ("sentiment" not in raw or not raw.sentiment.notna().any()):
    st.markdown("<div class='cv-warning'><strong>Unlabelled feedback · predictions only</strong><br>Analysis can continue, but this CSV has no verified sentiment labels for supervised training. Predictions use the active model shown above. To train a classifier, open <b>Train Sentiment Model &amp; Evaluation</b> and upload a separately labelled CSV. The <code>category_hint</code> field is treated only as topic metadata.</div>", unsafe_allow_html=True)
if raw.empty:
    st.markdown("<div class='cv-empty'><h3>Start with student feedback</h3><p>Load synthetic demo data or upload a CSV with a <code>comment</code> column. You can also add one comment from the sidebar.</p><p>Optional columns: department, semester, date, feedback_id, sentiment.</p></div>", unsafe_allow_html=True)
    st.code("comment,department,semester,date\nThe laboratory equipment needs maintenance,Engineering,4,2026-02-10", language="csv")
    if page != "Train Sentiment Model & Evaluation":
        st.stop()

if "ingest" in st.session_state:
    stats = st.session_state.ingest
    st.caption(f"{stats['accepted_rows']:,} accepted · {stats['missing_comments']:,} blank skipped · {stats['duplicate_comments']:,} repeated comments retained · {stats['exact_duplicate_rows']:,} exact duplicate rows found ({stats['exact_duplicates_removed']:,} removed) · {stats['duplicate_feedback_ids']:,} repeated feedback IDs · {stats.get('invalid_dates', 0):,} invalid dates retained without dates")
with st.expander("Preview loaded dataset", expanded=False):
    st.dataframe(raw.head(10), hide_index=True, use_container_width=True)

base = raw.copy()
if "comment" not in base.columns:
    base = pd.DataFrame(columns=["comment"])
texts = base.comment.astype(str).tolist()
trained_model = st.session_state.get("trained_model") if use_trained else None
if use_trained and trained_model is not None:
    preds = predict_classifier(trained_model, texts)
else:
    preds = [vader_score(t) for t in texts]
analysis = base.copy()
if "sentiment" in base:
    analysis["ground_truth_sentiment"] = base["sentiment"]
analysis["sentiment"] = [p["sentiment"] for p in preds]
analysis["compound"] = [p.get("compound") for p in preds]
analysis["model_score"] = [p.get("confidence") for p in preds]
analysis["engine"] = [p.get("engine", "Logistic Regression") for p in preds]
analysis["aspect_rows"] = [detect_aspects(t) for t in texts]
analysis["aspects"] = [", ".join(a["aspect"] for a in row) for row in analysis.aspect_rows]
analysis["aspect_evidence"] = ["; ".join(f"{a['aspect']} [{a['sentiment']}]: {a['evidence']}" for a in row) for row in analysis.aspect_rows]
topics = discover_topics(texts, analysis.sentiment.tolist())
aspects = aspect_frame(analysis)
insights = generate_insights(analysis.assign(aspects=analysis.aspects.str.split(", "))) if not analysis.empty else []

# Global filters apply to all analytical pages.
with st.sidebar:
    st.subheader("Global filters")
    filter_values = {}
    for column in ["department", "program", "semester", "academic_year", "course", "category_hint", "source"]:
        values = sorted(analysis[column].dropna().astype(str).unique()) if column in analysis else []
        if values:
            filter_values[column] = st.multiselect(column.replace("_", " ").title(), values, default=values, key=f"filter_{column}")
    date_range = None
    if "date" in analysis:
        parsed_dates = pd.to_datetime(analysis.date, errors="coerce").dropna()
        if len(parsed_dates):
            bounds = (parsed_dates.min().date(), parsed_dates.max().date())
            date_range = st.date_input("Date range", value=bounds if bounds[0] != bounds[1] else (bounds[0], bounds[0]))

filtered = analysis.copy()
for column, values in filter_values.items():
    filtered = filtered[filtered[column].astype(str).isin(values)]
if date_range and len(date_range) == 2 and "date" in filtered:
    parsed = pd.to_datetime(filtered.date, errors="coerce")
    filtered = filtered[parsed.isna() | parsed.dt.date.between(date_range[0], date_range[1])]
f_aspects = aspect_frame(filtered)
topics = discover_topics(filtered.comment.astype(str).tolist(), filtered.sentiment.tolist())

if page == "Executive Overview":
    st.subheader("Executive overview")
    metric_cols(filtered)
    left, right = st.columns(2)
    with left: sentiment_donut(filtered)
    with right:
        counts = f_aspects.aspect.value_counts().rename_axis("Aspect").reset_index(name="Mentions") if not f_aspects.empty else pd.DataFrame(columns=["Aspect", "Mentions"])
        if len(counts):
            fig = px.bar(counts.sort_values("Mentions"), x="Mentions", y="Aspect", orientation="h", title="Feedback categories", color_discrete_sequence=["#9ab49c"], text="Mentions")
            fig.update_traces(textposition="outside"); st.plotly_chart(style_fig(fig), use_container_width=True)
        else: st.info("No configured aspect keywords matched this feedback.")
    if "category_hint" in filtered and filtered.category_hint.notna().any():
        category_counts = filtered.category_hint.fillna("Unspecified").astype(str).value_counts().rename_axis("Category").reset_index(name="Comments")
        st.plotly_chart(style_fig(px.bar(category_counts.sort_values("Comments"), x="Comments", y="Category", orientation="h", title="Uploaded feedback categories", color_discrete_sequence=["#b9a77a"])), use_container_width=True)
    trend = sentiment_over_time(filtered)
    if trend is not None: st.plotly_chart(trend, use_container_width=True)
    else: st.caption("Add valid dates across multiple comments to see sentiment trends over time.")
    if not f_aspects.empty:
        pivot = f_aspects.pivot_table(index="aspect", columns="sentiment", values="comment", aggfunc="count", fill_value=0).reset_index()
        cols = [c for c in ["Positive", "Neutral", "Negative"] if c in pivot]
        fig = px.bar(pivot, x=cols, y="aspect", orientation="h", title="Sentiment by category", color_discrete_map=COLORS, labels={"aspect":"Aspect", "value":"Comments"})
        st.plotly_chart(style_fig(fig), use_container_width=True)
    st.subheader("Recent feedback")
    recent_cols = [c for c in ["date", "comment", "sentiment", "aspects", "department"] if c in filtered]
    st.dataframe(filtered[recent_cols].tail(8).iloc[::-1], hide_index=True, use_container_width=True)

elif page == "Sentiment Analytics":
    st.subheader("Sentiment analytics")
    sentiment_donut(filtered, "Overall sentiment in current selection")
    left, right = st.columns(2)
    if "department" in filtered and filtered.department.notna().any():
        dept = filtered.groupby(["department", "sentiment"]).size().reset_index(name="Comments")
        with left: st.plotly_chart(style_fig(px.bar(dept, x="department", y="Comments", color="sentiment", barmode="group", color_discrete_map=COLORS, title="Sentiment by department", labels={"department":"Department"})), use_container_width=True)
    if "semester" in filtered and filtered.semester.notna().any():
        sem = filtered.groupby([filtered.semester.astype(str), "sentiment"]).size().reset_index(name="Comments").rename(columns={"semester":"Semester"})
        with right: st.plotly_chart(style_fig(px.bar(sem, x="Semester", y="Comments", color="sentiment", barmode="group", color_discrete_map=COLORS, title="Sentiment by semester")), use_container_width=True)
    trend = sentiment_over_time(filtered)
    if trend is not None: st.plotly_chart(trend, use_container_width=True)
    scores = pd.to_numeric(filtered.compound, errors="coerce").dropna()
    if len(scores): st.plotly_chart(style_fig(px.histogram(pd.DataFrame({"Compound score":scores}), x="Compound score", nbins=20, title="VADER compound score distribution", color_discrete_sequence=["#9ab49c"])), use_container_width=True)
    negatives = filtered[filtered.sentiment == "Negative"].copy(); positives = filtered[filtered.sentiment == "Positive"].copy()
    a,b = st.columns(2)
    with a: st.markdown("**Negative feedback examples**"); st.dataframe(negatives[[c for c in ["comment","department","semester"] if c in negatives]].head(10), hide_index=True, use_container_width=True)
    with b: st.markdown("**Positive feedback examples**"); st.dataframe(positives[[c for c in ["comment","department","semester"] if c in positives]].head(10), hide_index=True, use_container_width=True)

elif page == "Aspect Analysis":
    st.subheader("Aspect based opinion analysis")
    st.caption("Aspect categories use editable keyword phrase rules. Sentiment is estimated from the matching phrase's clause; this is a heuristic, not a trained aspect sentiment model.")
    if f_aspects.empty: st.info("No aspect phrases matched. Try another dataset or review src/aspects.py keyword mappings.")
    else:
        counts = f_aspects.groupby(["aspect","sentiment"]).size().reset_index(name="Comments")
        l,r = st.columns(2)
        with l: st.plotly_chart(style_fig(px.bar(counts, x="Comments", y="aspect", color="sentiment", orientation="h", barmode="stack", color_discrete_map=COLORS, title="Sentiment by aspect")), use_container_width=True)
        with r:
            freq = f_aspects.aspect.value_counts().sort_values().rename_axis("Aspect").reset_index(name="Mentions")
            st.plotly_chart(style_fig(px.bar(freq, x="Mentions", y="Aspect", orientation="h", title="Most frequently mentioned aspects", color_discrete_sequence=["#9ab49c"])), use_container_width=True)
        heat = counts.pivot(index="aspect", columns="sentiment", values="Comments").fillna(0)
        st.plotly_chart(style_fig(px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale=["#202522","#77977b"], title="Aspect and sentiment heatmap", labels={"x":"Sentiment","y":"Aspect","color":"Comments"})), use_container_width=True)
        negative = f_aspects.assign(negative=f_aspects.sentiment.eq("Negative")).groupby("aspect").negative.mean().mul(100).round(1).rename("Negative share (%)").reset_index()
        st.markdown("**Negative feedback share by aspect**")
        st.dataframe(negative.sort_values("Negative share (%)", ascending=False), hide_index=True, use_container_width=True)
        st.markdown("**Department comparison**")
        dept_aspects = f_aspects.dropna(subset=["department"]).groupby(["department","aspect"]).size().reset_index(name="Mentions")
        if not dept_aspects.empty: st.plotly_chart(style_fig(px.bar(dept_aspects, x="department", y="Mentions", color="aspect", barmode="group", title="Aspect mentions by department")), use_container_width=True)
        selected = st.selectbox("Drill into aspect", sorted(f_aspects.aspect.unique()))
        st.dataframe(f_aspects[f_aspects.aspect == selected][["aspect","sentiment","department","evidence","comment"]], hide_index=True, use_container_width=True)

elif page == "Topic Discovery":
    st.subheader("Topic discovery")
    st.caption("Topics are unsupervised NMF word patterns. Labels are generated from representative terms and can be renamed below.")
    if not topics: st.info("There are too few varied comments for stable topics. Try at least four comments with different vocabulary.")
    else:
        topic_df = pd.DataFrame([{"Topic":t["name"],"Comments":t["count"],"Negative share":t["negative_proportion"]} for t in topics])
        st.plotly_chart(style_fig(px.bar(topic_df.sort_values("Comments"), x="Comments", y="Topic", orientation="h", title="Topic prevalence", color_discrete_sequence=["#9ab49c"])), use_container_width=True)
        selected_idx = st.selectbox("Inspect topic", range(len(topics)), format_func=lambda i: f"{topics[i]['name']} · {topics[i]['count']} comments")
        topic = topics[selected_idx]
        st.markdown("**Representative terms** · " + " · ".join(topic["terms"]))
        term_rows = pd.DataFrame({"Term": topic["terms"], "NMF term weight": topic["term_weights"]})
        st.plotly_chart(style_fig(px.bar(term_rows.sort_values("NMF term weight"), x="NMF term weight", y="Term", orientation="h", title="Representative term weights", color_discrete_sequence=["#9ab49c"])), use_container_width=True)
        st.metric("Negative share in assigned comments", f"{topic['negative_proportion']:.1%}", help="Calculated over the topic's assigned comments")
        members = filtered.iloc[topic["member_indices"]]
        dist_left, dist_right = st.columns(2)
        with dist_left:
            topic_sent = members.sentiment.value_counts().rename_axis("Sentiment").reset_index(name="Comments")
            st.plotly_chart(style_fig(px.bar(topic_sent, x="Sentiment", y="Comments", color="Sentiment", color_discrete_map=COLORS, title="Topic by sentiment")), use_container_width=True)
        with dist_right:
            if "department" in members and members.department.notna().any():
                topic_dept = members.department.fillna("Unspecified").value_counts().rename_axis("Department").reset_index(name="Comments")
                st.plotly_chart(style_fig(px.bar(topic_dept, x="Comments", y="Department", orientation="h", title="Topic by department", color_discrete_sequence=["#9ab49c"])), use_container_width=True)
        st.markdown("**Representative comments**")
        st.dataframe(members[[c for c in ["comment", "sentiment", "department", "semester"] if c in members]].head(10), hide_index=True, use_container_width=True)
        all_examples = pd.DataFrame([{"Topic": t["name"], "Representative terms": ", ".join(t["terms"]), "Assigned comments": t["count"], "Negative comments": t["negative_count"], "Negative share": t["negative_proportion"]} for t in topics])
        st.dataframe(all_examples, hide_index=True, use_container_width=True)

elif page == "Department Comparison":
    st.subheader("Department comparison")
    if "department" not in filtered or filtered.department.notna().sum() == 0: st.info("Department values are not present in the current data selection.")
    else:
        dept_sent = filtered.dropna(subset=["department"]).groupby(["department","sentiment"]).size().reset_index(name="Comments")
        l,r = st.columns(2)
        with l: st.plotly_chart(style_fig(px.bar(dept_sent, x="department", y="Comments", color="sentiment", barmode="stack", color_discrete_map=COLORS, title="Sentiment counts by department")), use_container_width=True)
        pct = dept_sent.copy(); pct["Share"] = pct.Comments / pct.groupby("department").Comments.transform("sum") * 100
        with r: st.plotly_chart(style_fig(px.bar(pct, x="department", y="Share", color="sentiment", barmode="stack", color_discrete_map=COLORS, title="Sentiment mix by department", labels={"Share":"Share (%)"})), use_container_width=True)
        if not f_aspects.empty:
            top = f_aspects.groupby(["department","aspect"]).size().reset_index(name="Mentions")
            st.plotly_chart(style_fig(px.bar(top, x="department", y="Mentions", color="aspect", barmode="group", title="Aspect mentions by department")), use_container_width=True)
        st.dataframe(filtered.groupby("department").agg(Comments=("comment","size"), Negative=("sentiment",lambda x:(x=="Negative").sum()), Positive=("sentiment",lambda x:(x=="Positive").sum())).reset_index(), hide_index=True, use_container_width=True)

elif page == "Feedback Explorer":
    st.subheader("Feedback explorer")
    c1,c2,c3 = st.columns([2,1,1])
    query = c1.text_input("Search comment text")
    selected_sent = c2.multiselect("Sentiment", ["Positive","Negative","Neutral"], default=["Positive","Negative","Neutral"])
    available = sorted(set(a for rows in filtered.aspect_rows for a in [x["aspect"] for x in rows]))
    selected_aspects = c3.multiselect("Aspect", available, default=available)
    view = filtered[filtered.sentiment.isin(selected_sent) & filtered.comment.str.contains(query, case=False, na=False)]
    view = view[view.aspect_rows.apply(lambda rows: any(x["aspect"] in selected_aspects for x in rows))]
    st.caption(f"Showing {len(view):,} of {len(filtered):,} comments")
    cols = [c for c in ["feedback_id","date","comment","sentiment","ground_truth_sentiment","aspects","aspect_evidence",
                        "department","program","semester","academic_year","course","category_hint","source"] if c in view]
    st.dataframe(view[cols], hide_index=True, use_container_width=True)

elif page == "Actionable Insights":
    st.subheader("Areas for human review")
    st.caption("These suggestions are transparent prompts tied to negative feedback. They are not proven causes or automated decisions.")
    scoped = generate_insights(filtered.assign(aspects=filtered.aspects.str.split(", ")))
    if not scoped: st.info("No aspect currently has at least two negative comments.")
    for item in scoped:
        with st.container(border=True):
            st.markdown(f"### {item['aspect']} · {item['negative_count']} negative / {item['total_count']} mentions ({item['negative_proportion']:.0%})")
            st.write(item["suggestion"])
            st.caption("Supporting comments")
            for quote in item["evidence"]: st.markdown(f"- {quote}")

elif page == "Train Sentiment Model & Evaluation":
    st.subheader("Train Sentiment Model")
    st.caption("Use verified sentiment labels to fit a classifier, then evaluate it on held-out examples before applying it to feedback.")
    st.write("Train a local TF-IDF + Logistic Regression pipeline from human-labelled examples. The optional stratified holdout evaluates a separate model fitted only on the training partition; after that, the active session model is fitted on all labelled training rows.")
    template = pd.DataFrame(columns=["comment", "sentiment"])
    st.download_button("Download labelled CSV template", template.to_csv(index=False).encode("utf-8"), "campusvoice_labelled_template.csv", "text/csv")
    load_training_demo = st.button("Load Labelled Demo Dataset", help="36 hand-labelled synthetic examples; not real student feedback or validation data.")
    if load_training_demo:
        demo = pd.read_csv(ROOT / "data" / "demo_labeled_training.csv")
        st.session_state.training_data = demo
        st.session_state.training_hash = sha256(demo.to_csv(index=False).encode()).hexdigest()
        st.session_state.training_is_synthetic = True
        st.info("Synthetic labelled demonstration data loaded. These examples are fictional and evaluation results do not represent real student feedback.")
    labelled_file = st.file_uploader("Upload labelled CSV (required columns: comment, sentiment)", type="csv", key="training_csv")
    if labelled_file is not None and not load_training_demo:
        try:
            source_bytes = labelled_file.getvalue()
            source_hash = sha256(source_bytes).hexdigest()
            if st.session_state.get("last_training_upload_hash") != source_hash:
                if st.session_state.get("model_source_hash") != source_hash:
                    st.session_state.trained_model_valid = False
                st.session_state.training_data = None
                st.session_state.evaluation_result = None
                clean, stats = data_loader.read_feedback_csv(BytesIO(source_bytes))
                if "sentiment" not in clean:
                    raise ValueError("This file has no sentiment column. Add human-verified Positive, Negative, or Neutral labels to train a supervised model.")
                clean = clean.dropna(subset=["sentiment"])
                if clean.empty:
                    raise ValueError("No labelled comments remain after blank rows are removed.")
                st.session_state.training_data = clean
                st.session_state.training_hash = source_hash
                st.session_state.last_training_upload_hash = source_hash
                st.session_state.training_is_synthetic = False
                st.session_state.training_ingest = stats
                st.session_state.evaluation_result = None
                if st.session_state.get("model_source_hash") != source_hash:
                    st.session_state.trained_model_valid = False
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(str(exc))
    if st.session_state.get("trained_model") is not None and not st.session_state.get("trained_model_valid", False):
        st.warning("The labelled training dataset changed. The previous trained model is marked stale; train again before using it for predictions.")
    training_data = st.session_state.get("training_data")
    if training_data is not None:
        labels = training_data.sentiment.tolist()
        class_counts = pd.Series(labels).value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
        st.caption(f"Training data: {'Synthetic demonstration' if st.session_state.get('training_is_synthetic') else 'Uploaded labelled dataset'} · {len(training_data):,} labelled examples")
        train_stats = st.session_state.get("training_ingest")
        if train_stats and not st.session_state.get("training_is_synthetic"):
            st.caption(f"Training file: {train_stats['accepted_rows']:,} accepted · {train_stats['missing_comments']:,} blank skipped · {train_stats['duplicate_comments']:,} repeated comments retained · {train_stats['exact_duplicate_rows']:,} exact duplicate rows found · {train_stats['duplicate_feedback_ids']:,} repeated feedback IDs · {train_stats['invalid_dates']:,} invalid dates retained without dates")
        if st.session_state.get("trained_model_valid") and st.session_state.get("model_source_hash") == st.session_state.get("training_hash"):
            status = st.session_state.get("training_status", {})
            st.success(f"Active model trained in this session on {status.get('rows', len(training_data)):,} labelled examples across {status.get('classes', len(set(labels)))} classes.")
        st.dataframe(training_data.head(8), hide_index=True, use_container_width=True)
        dist_a, dist_b = st.columns([1,2])
        with dist_a: st.dataframe(class_counts.rename_axis("Sentiment").reset_index(name="Examples"), hide_index=True, use_container_width=True)
        with dist_b: st.plotly_chart(style_fig(px.bar(class_counts.rename_axis("Sentiment").reset_index(name="Examples"), x="Sentiment", y="Examples", color="Sentiment", color_discrete_map=COLORS, title="Label distribution")), use_container_width=True)
        if len(set(labels)) < 2:
            st.error("Training needs at least two sentiment classes. This dataset currently has only one.")
        else:
            st.caption("Training readiness: at least two classes detected. Held-out metrics require at least 8 rows and enough examples in each class.")
            if st.button("Train model and evaluate holdout", type="primary"):
                try:
                    evaluation = None
                    try:
                        evaluation = compare_models(training_data.comment.astype(str).tolist(), labels)
                    except ValueError as exc:
                        st.warning(f"Model can train, but held-out evaluation is unavailable: {exc}")
                    model = train_classifier(training_data.comment.astype(str).tolist(), labels)
                    st.session_state.trained_model = model
                    st.session_state.model_source_hash = st.session_state.training_hash
                    st.session_state.trained_model_valid = True
                    st.session_state.evaluation_result = evaluation
                    st.session_state.training_status = {"rows": len(training_data), "classes": len(set(labels))}
                    st.session_state.activate_trained_model_next_run = True
                    st.success(f"Training complete: TF-IDF + Logistic Regression fitted on {len(training_data):,} labelled examples across {len(set(labels))} classes. This model is available for subsequent predictions in this session.")
                except ValueError as exc:
                    st.error(f"Training could not complete: {exc}")
        if st.session_state.get("evaluation_result"):
            result = st.session_state.evaluation_result
            st.markdown(f"### Held-out evaluation · {result['train_size']} train / {result['test_size']} test")
            st.caption("Metrics below use only held-out comments for scoring. The active model above is separately fitted on all labelled rows.")
            metrics = result["logistic_regression"]
            mcols = st.columns(5)
            mcols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
            mcols[1].metric("Macro precision", f"{metrics['macro_precision']:.3f}")
            mcols[2].metric("Macro recall", f"{metrics['macro_recall']:.3f}")
            mcols[3].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
            mcols[4].metric("Test rows", result["test_size"])
            report_rows = []
            for label, values in metrics["per_class"].items():
                report_rows.append({"Sentiment": label, "Precision": values["precision"], "Recall": values["recall"], "F1": values["f1"], "Support": values["support"]})
            st.dataframe(pd.DataFrame(report_rows), hide_index=True, use_container_width=True)
            cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
            st.plotly_chart(style_fig(px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale=["#202522", "#9ab49c"], title="Confusion matrix · held-out test", labels={"x":"Predicted sentiment","y":"Actual sentiment","color":"Comments"})), use_container_width=True)
            st.caption("VADER and classifier comparison uses the exact same held-out rows.")
            compare = pd.DataFrame([{"Model": "TF-IDF + Logistic Regression", "Accuracy": result["logistic_regression"]["accuracy"], "Macro F1": result["logistic_regression"]["macro_f1"]}, {"Model": "VADER", "Accuracy": result["vader"]["accuracy"], "Macro F1": result["vader"]["macro_f1"]}])
            st.plotly_chart(style_fig(px.bar(compare, x="Model", y=["Accuracy", "Macro F1"], barmode="group", title="Comparable held-out metrics")), use_container_width=True)
            if st.session_state.get("training_is_synthetic"):
                st.warning("These results come from synthetic demonstration examples. They do not show performance on real student feedback.")
    elif st.session_state.get("trained_model") is not None:
        st.info("A trained model is retained in this session. Upload labelled data above to train another version.")
    else:
        st.info("Upload a labelled CSV or load the synthetic training demo. category_hint is preserved as topic metadata and is never used as sentiment.")

elif page == "Reports & Export":
    st.subheader("Reports and export")
    st.caption("Exports follow the current department, semester, and date filters.")
    st.download_button("Download filtered analysis CSV", filtered.drop(columns=["aspect_rows"], errors="ignore").to_csv(index=False).encode("utf-8"), "campusvoice_analysis.csv", "text/csv")
    report = markdown_report(filtered, topics, generate_insights(filtered.assign(aspects=filtered.aspects.str.split(", "))), "VADER compound thresholds (+/-0.05), optional TF-IDF Logistic Regression, clause-level keyword aspect rules, and TF-IDF/NMF topic discovery.")
    st.download_button("Download Markdown summary report", report, "campusvoice_report.md", "text/markdown")
    st.markdown("**Current selection summary**")
    metric_cols(filtered)
    sentiment_donut(filtered)

elif page == "Methodology":
    st.subheader("Methodology and responsible use")
    st.markdown("""
- **Sentiment:** VADER is a lexicon baseline. Logistic Regression uses TF-IDF and must be trained and evaluated on labelled examples. VADER compound values are continuous scores, not probabilities.
- **Aspects:** editable phrase rules can detect multiple categories in one comment. Each matched phrase is associated with its local clause and scored with VADER. Clause parsing and lexicon polarity are heuristics; they can miss paraphrases, mixed syntax, and context.
- **Topics:** NMF decomposes comment TF-IDF vectors into recurring word patterns. Labels are representative terms, not guaranteed semantic categories. Topics are fitted over the analyzed comments.
- **Data:** blank comments are skipped. Repeated comment text is retained; exact duplicate rows can be removed with the optional sidebar setting. Repeated feedback IDs are reported for review. Original comment text and metadata are retained. This app processes data locally and does not send it to an API.
- **Responsible use:** collect consent and remove identifying details. Review small-group disclosures before sharing. Use outputs to guide service review, not to rank or penalize individual staff or students.
- **Evaluation:** use an independently held, representative labelled test set; inspect class balance and class-wise precision/recall/F1. Synthetic data demonstrates mechanics and is not real-world validation.
""")
