# Data

`demo_feedback.csv` is synthetic illustrative dashboard data. It does not represent real college feedback. Any supplied labels are fictional and are not representative validation.

`demo_labeled_training.csv` is a separate, explicitly synthetic 300-row dataset for demonstrating the supervised training workflow. It has 100 Positive, 100 Negative, and 100 Neutral examples across ten feedback topics; `topic_group` lets evaluation hold out complete topics. The examples are fictional and must not be presented as verified student sentiment or real-world model validation. The held-out scores describe only this synthetic dataset.

For classifier training, upload a labelled CSV with required columns `comment` and `sentiment`; labels must be Positive, Neutral, or Negative. Collect consent, remove identifying details, and use a representative sample before making real-world claims.
