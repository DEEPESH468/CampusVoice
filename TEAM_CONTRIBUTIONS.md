# Team contributions and work plan

This is a proposed four-person division of responsibility, not a claim that work has already been independently completed by each member. All team members should review the end-to-end flow and be prepared to explain data leakage, evaluation limits, and privacy choices.

| Member | Primary ownership | Key files | Integration dependency |
|---|---|---|---|
| 1 — Data and preprocessing | CSV schema, validation, text normalization, tokenization, ingestion tests | `src/data_loader.py`, `src/preprocessing.py`, `tests/test_data_loader.py`, `tests/test_preprocessing.py` | Supplies clean comments and row counts to sentiment and app |
| 2 — Sentiment classification | VADER, TF-IDF/Logistic Regression, labelled training and metrics | `src/sentiment.py`, `src/evaluation.py`, `tests/test_sentiment.py`, `tests/test_evaluation.py` | Consumes clean rows; returns labels and evaluation results to dashboard |
| 3 — Aspects and themes | Aspect phrase map, evidence matching, NMF topics, suggestion rules | `src/aspects.py`, `src/topic_modeling.py`, `src/insights.py`, `tests/test_aspects.py` | Consumes original comments and predicted labels; passes evidence to UI/report |
| 4 — Frontend and integration | Streamlit layout, filters, charts, exports, demo workflow, integration checks | `app.py`, `src/reporting.py`, `README.md`, docs | Integrates interfaces from members 1–3 and checks full workflow |

## Suggested Git branch workflow

1. Keep `main` runnable; create `feature/data-ingestion`, `feature/sentiment`, `feature/aspects-topics`, and `feature/dashboard` branches.
2. Agree on shared contracts early: a DataFrame containing `comment` plus optional metadata, labels from `{Positive, Neutral, Negative}`, and aspect results with `aspect` and `evidence` fields.
3. Commit small changes with descriptive messages; merge via reviewed pull requests into `main`.
4. Rebase or merge `main` before integration; resolve interface changes with the module owner.
5. Run `python -m pytest` and start `streamlit run app.py` after each integration. Tag the demo-ready commit.
