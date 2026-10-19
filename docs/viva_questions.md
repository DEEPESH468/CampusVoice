# Viva preparation: CampusVoice

1. **What is NLP?** Techniques that let computers process and analyze human language.
2. **What is a token?** A unit such as a word or punctuation segment produced by tokenization.
3. **Why normalize whitespace?** It removes accidental formatting differences without changing meaning.
4. **Why preserve the original comment?** It keeps evidence auditable and avoids presenting transformed text as the student's words.
5. **Why retain “not” and “never”?** They reverse or alter sentiment, as in “not useful.”
6. **What is stemming?** Heuristic reduction of words to stems; this project avoids it to preserve evidence and avoid damaging terms.
7. **What is VADER?** A lexicon and rule-based sentiment tool designed for social text.
8. **What is VADER compound score?** A normalized summary of valence, approximately from −1 to +1.
9. **What thresholds are used?** Compound ≥ 0.05 is Positive, ≤ −0.05 Negative, otherwise Neutral.
10. **What is TF-IDF?** A term weighting that raises weight for terms frequent in a document but less common across documents.
11. **Why use unigrams and bigrams?** They capture single words and short phrases such as “not helpful.”
12. **What does Logistic Regression do?** It estimates class probabilities from a weighted combination of input features.
13. **Why is it called regression when used for classification?** The logistic function maps a linear score to class probabilities; it is commonly used as a classifier.
14. **What is a lexicon model's strength?** It works without task-specific labelled training data and is easy to explain.
15. **What is a lexicon model's weakness?** It can miss domain meaning, sarcasm, and context.
16. **What is supervised learning?** Learning a mapping from examples with known labels.
17. **Why split train and test data?** To measure performance on examples not used to fit the model.
18. **What is data leakage?** Information from test data influencing training or feature fitting, leading to overly optimistic evaluation.
19. **How does CampusVoice prevent TF-IDF leakage?** Vectorization is inside a pipeline fitted only on the training partition.
20. **What is accuracy?** The fraction of predictions that are correct.
21. **What is precision?** Of items predicted as a class, the fraction that truly belong to that class.
22. **What is recall?** Of true items in a class, the fraction correctly found.
23. **What is F1?** The harmonic mean of precision and recall.
24. **Why macro F1?** It averages class F1 equally, so a common class does not dominate the summary.
25. **What is a confusion matrix?** A table comparing true and predicted class counts.
26. **What is class imbalance?** A dataset where some labels have many more examples than others.
27. **What is aspect-based sentiment analysis?** Analysis of opinions about particular targets or aspects, not just overall polarity.
28. **How are aspects found here?** Editable phrase rules match configured vocabulary and retain a matched evidence phrase.
29. **Can one comment have multiple aspects?** Yes; each matching category is reported.
30. **What is NMF?** Non-negative Matrix Factorization approximates a non-negative document-term matrix with lower-dimensional topic and document weights.
31. **Why use TF-IDF for topics?** It represents comments as weighted terms suitable for NMF.
32. **Are topic labels ground truth?** No. Top terms and examples help a person interpret the discovered patterns.
33. **What is clustering?** Grouping examples by similarity without requiring labels; topic assignment here uses highest NMF topic weight.
34. **Can synthetic results prove model quality?** No. Synthetic data is simplified and may not represent real comments.
35. **How should genuine student feedback be collected?** With consent, clear purpose, anonymization, and representative sampling across relevant groups.
36. **Why not rank teachers automatically?** Models can be noisy and biased; comments lack the context required for fair individual judgments.
37. **What does confidence mean here?** The classifier's largest estimated class probability; it may not be calibrated or correct.
38. **Why keep analysis local?** It reduces unnecessary disclosure to external services and fits the no-cloud project scope.
39. **What happens to duplicate comments?** Exact duplicate normalized comments after the first are skipped and counted.
40. **What is a future improvement?** Evaluate with a representative human-labelled corpus and improve aspect-level polarity and multilingual support.
