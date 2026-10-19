# CampusVoice: Student Feedback Intelligence System

## Abstract

CampusVoice is a local Python application that organizes free-text student feedback into sentiment, service aspects, recurring word themes, and evidence-linked suggestions. It combines VADER, TF-IDF with Logistic Regression, phrase-rule aspect detection, and NMF topic discovery. Outputs are intended to help humans review large feedback collections; the application does not establish causality or identify individual performance. The included demonstration data is synthetic.

## Introduction and problem statement

Colleges may receive hundreds of comments about courses, laboratories, facilities, examinations, and student services. Reading each comment manually makes it hard to see recurring concerns. The project objective is to convert unstructured feedback into understandable summaries while keeping comments and supporting evidence visible.

## Objectives and scope

- Accept manual input and CSV with required comment and optional metadata.
- Estimate Positive, Neutral, or Negative sentiment with a lexicon baseline or supervised classifier.
- Detect multiple college service aspects with matched phrase evidence.
- Discover recurring topics and show representative comments.
- Suggest possible areas for human review using observed counts and evidence.
- Export analysis and provide transparent methodology and limitations.

The system does not identify students, rank staff, infer causes, or replace institutional review. No real labelled dataset is bundled.

## Existing approach and limitations

Manual review is context-sensitive but slow and difficult to summarize consistently at scale. Simple word counts are easy to explain but ignore word order, negation, and context. Automated analysis is faster but can miss sarcasm, local terminology, and nuances; outputs therefore need human review.

## Proposed system and architecture

CSV or manual input is validated, preserving original comment text. Sentiment models and aspect/topic analysis operate on comments locally. A dashboard combines counts, examples, filters, suggestions, model evaluation, and exports. The Mermaid flow is in `architecture.md`.

## NLP methodology

Whitespace is normalized for analysis and blank comments are rejected. The stored/displayed source comment is retained. Tokenization is available without stemming or lemmatization. Negators are preserved. Topic TF-IDF uses a limited English stop-word removal step, while sentiment consumes the original text because negation and function words can matter.

VADER provides positive, neutral, negative, and compound lexicon scores. CampusVoice uses the documented compound thresholds +0.05 and −0.05. Logistic Regression learns class weights on unigram and bigram TF-IDF features. The vectorizer is part of a scikit-learn pipeline and is fitted only on the training split. Its maximum class probability is shown as model confidence, but probability calibration is not guaranteed.

Aspect analysis uses editable phrase mappings. A comment can match multiple categories. The matching clause is shown as evidence and passed to VADER to estimate local polarity; a comment with no matching phrase is classified as Other or unclassified. This is a heuristic aspect-polarity estimate, not a trained aspect sentiment model, and clause boundaries, vocabulary, and contextual references can produce errors.

Topic analysis fits NMF to TF-IDF, names topics using high-weight terms, assigns each comment to its highest-weight topic, and shows representative comments. NMF labels are word summaries, not verified themes. Fewer than four comments or insufficient vocabulary yields no topics.

## Dataset description

`data/demo_feedback.csv` contains illustrative synthetic records authored for demonstrations, with synthetic sentiment labels. It is not a sample of real students and cannot validate institutional accuracy. Real evaluation requires consented, anonymized, representative labelled feedback. `data/README.md` documents the training CSV format.

## Algorithms and implementation

The implementation uses pandas for ingestion and tables, NLTK VADER for the lexicon baseline, scikit-learn TF-IDF and Logistic Regression for supervised sentiment, scikit-learn NMF for topics, Streamlit for the interface, and Plotly/Matplotlib for charts. Aspect mappings and action rules are maintained in separate modules for student-friendly extension.

## Experimental methodology and results

For labelled data, the reusable comparison function performs a fixed-seed stratified 75/25 split (when there are at least eight rows and at least two examples per class), then evaluates VADER and Logistic Regression on the same held-out comments. It reports accuracy, macro precision/recall/F1, per-class scores, and confusion matrices. No real-world dataset was supplied for this project, so no real-world model performance result is claimed. Synthetic demo evaluation is available only to illustrate the workflow and is not evidence of generalization.

## Limitations

VADER is English-centric and lexicon-based. Synthetic training examples do not represent naturally written feedback. Bag-of-words TF-IDF has limited semantic understanding. Keyword aspects miss synonyms and can misread context. NMF topics require interpretation and can vary with the dataset. Current aspect sentiment is the comment-level sentiment attached to each detected category, so mixed-polarity sentences can be misrepresented. Confidence is not calibrated. Small or imbalanced datasets produce unstable estimates.

## Privacy and ethical considerations

Collect feedback with consent and remove direct identifiers before upload. Avoid sharing small subgroup results that could reveal an author. Restrict access and retention in the institution's real deployment process. Use output as a prompt for contextual human review; do not automatically penalize or rank individual teachers or students. Keep student data on the local machine.

## Conclusion and future scope

CampusVoice demonstrates a modular pipeline from feedback ingestion to explainable sentiment, aspect evidence, topic patterns, and downloadable summaries. Future work can add a representative consented corpus, human-reviewed aspect-level sentiment, multilingual support, stronger privacy controls, and longitudinal evaluation.

## References

- Hutto, C. J. and Gilbert, E. (2014). “VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.” ICWSM.
- Pedregosa et al. (2011). “Scikit-learn: Machine Learning in Python.” *Journal of Machine Learning Research*, 12, 2825–2830.
- Salton, G. and Buckley, C. (1988). “Term-weighting approaches in automatic text retrieval.” *Information Processing & Management*, 24(5), 513–523.
- Lee, D. D. and Seung, H. S. (1999). “Learning the parts of objects by non-negative matrix factorization.” *Nature*, 401, 788–791.
- Official library documentation: [NLTK](https://www.nltk.org/), [scikit-learn](https://scikit-learn.org/), [Streamlit](https://docs.streamlit.io/), [Pandas](https://pandas.pydata.org/docs/).
