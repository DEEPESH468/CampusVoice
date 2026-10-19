# Presentation outline (9 slides)

1. **Title and team** — CampusVoice, course/project details, four team members.
2. **Problem** — Large volumes of feedback are difficult to summarize manually; show example categories.
3. **Objectives and scope** — Sentiment, aspects, recurring themes, evidence, suggestions; no individual ranking.
4. **Architecture** — Explain data sources, local pipeline, dashboard, and exports using `architecture.md` Mermaid diagram.
5. **NLP methods** — VADER thresholds, TF-IDF + Logistic Regression, aspect phrase rules, NMF topics.
6. **Implementation and demo** — Load clearly labelled synthetic demo data, filter comments, inspect matched evidence and topics.
7. **Evaluation and results** — Explain stratified holdout metrics and confusion matrix. Present only values actually produced by the chosen labelled dataset; identify synthetic results as illustrative.
8. **Limitations, privacy, and team** — Small/biased data, rule coverage, human review, consent/anonymization; describe ownership from `TEAM_CONTRIBUTIONS.md` accurately.
9. **Conclusion and future work** — Useful local triage tool; next step is representative consented labelling and independent evaluation.
