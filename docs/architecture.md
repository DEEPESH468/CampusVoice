# CampusVoice architecture

```mermaid
flowchart TD
    A[Manual comment] --> C[Input validation and normalization]
    B[CSV upload] --> C
    D[Synthetic demo CSV] --> C
    C --> E[Original comments plus optional metadata]
    E --> F[VADER baseline or labelled TF-IDF + Logistic Regression]
    E --> G[Aspect phrase matching and evidence]
    E --> H[TF-IDF and NMF topic discovery]
    F --> I[Analysis table]
    G --> I
    H --> J[Topic counts, terms, representative comments]
    I --> K[Evidence-linked suggestions for human review]
    I --> L[Interactive dashboard and filters]
    J --> L
    K --> L
    L --> M[CSV analysis export]
    L --> N[Markdown summary report]
    O[Labelled CSV] --> P[Stratified train/test split]
    P --> Q[Fit model on training split]
    Q --> R[Compare VADER and classifier on held-out test]
    R --> L
```

The Streamlit process performs local computation. VADER lexicon resource availability is checked at runtime; NLTK downloads its small lexicon if absent. Topic extraction uses scikit-learn. No student comments are sent to a cloud model.
