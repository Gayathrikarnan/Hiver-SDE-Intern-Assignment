# Decision Log — Hiver SDE Intern Take-Home Assignment

## Project

**Retrieval-Grounded AI Support Agent for Spotify Customer Support**

This document records the important non-obvious engineering and design decisions made during the development of the project, along with the reasoning behind each decision.

---

## Decision 1 — Select Spotify as the target brand

**Decision:**  
Build the support agent specifically for Spotify rather than attempting to support multiple brands.

**Why:**
- The assignment asks us to pick one brand.
- A single-brand system allows the intent taxonomy and reply-generation logic to be tailored to one support domain.
- Spotify has a sufficiently large number of customer-support conversations for identifying recurring issues.
- Historical Spotify replies can be used directly as grounding evidence.

**Trade-off:**  
The resulting system is not immediately applicable to other brands without rebuilding the intent taxonomy and retrieval corpus.

---

## Decision 2 — Start with evaluation before building the complete agent

**Decision:**  
Create the evaluation set and baselines before focusing on the final support-agent pipeline.

**Why:**
- The assignment emphasizes that proof is more important than the system itself.
- Without a fixed evaluation set, it would be difficult to determine whether a model improvement was genuine.
- Baselines provide a reference point for judging whether the proposed classifier adds value.

**Trade-off:**  
This required spending development time on data preparation and evaluation before having a complete chatbot-style demo.

---

## Decision 3 — Use a manually labelled golden set of 200 examples

**Decision:**  
Create a 200-example manually labelled golden evaluation set.

**Why:**
- The assignment explicitly requires a golden evaluation set of 150–250 examples.
- 200 examples provide a manageable balance between labelling effort and evaluation coverage.
- Manual labelling provides a human-defined reference against which automated predictions can be measured.

**Final usable size:**  
198 examples, because two examples did not contain usable customer text.

**Trade-off:**  
A 198-example evaluation set is much smaller than the full dataset and therefore cannot represent every possible customer issue.

---

## Decision 4 — Keep the golden set separate from training

**Decision:**  
Do not use golden-set examples as training examples.

**Why:**
- Using evaluation examples during training could artificially increase the measured performance.
- The golden set should represent unseen examples for the classifier.
- Keeping the evaluation data separate makes the reported results more credible.

**Additional step:**  
Customer-message overlap was checked between the golden set and the weak-label training corpus.

**Result:**  
97 overlapping training rows were removed.

---

## Decision 5 — Remove exact text overlap rather than silently keeping it

**Decision:**  
Normalize customer messages and remove training rows whose normalized text exactly overlaps with golden examples.

**Why:**
- Duplicate or near-identical examples can make evaluation appear stronger than it really is.
- Removing exact overlaps reduces a direct form of evaluation leakage.
- The procedure is simple and reproducible.

**Limitation:**  
The current processed dataset does not retain conversation IDs. Therefore, a conversation-level train/evaluation split could not be verified.

**Future improvement:**  
Preserve the original conversation/thread identifiers and split data at the conversation level.

---

## Decision 6 — Define a small, data-derived intent taxonomy

**Decision:**  
Use 11 intents rather than creating a very large taxonomy.

**Final taxonomy:**

1. `ACCOUNT_LOGIN`
2. `APP_CRASH_PERFORMANCE`
3. `AVAILABILITY_REGION`
4. `BILLING_SUBSCRIPTION`
5. `BUG_REPORT`
6. `CONNECTIVITY_SYNC`
7. `FEATURE_REQUEST`
8. `FEEDBACK`
9. `OTHER_UNCLEAR`
10. `PLAYBACK_ISSUE`
11. `PLAYLIST_LIBRARY`

**Why:**
- A support agent needs a manageable number of operational categories.
- The categories were derived by examining recurring themes in the Spotify data.
- A very fine-grained taxonomy would make manual labelling harder and increase confusion between closely related intents.

**Trade-off:**  
Some categories still overlap semantically, especially `BUG_REPORT`, `FEEDBACK`, and `FEATURE_REQUEST`.

---

## Decision 7 — Keep OTHER_UNCLEAR as a fallback category

**Decision:**  
Retain `OTHER_UNCLEAR`, but treat it as an abstention/fallback category rather than a normal support intent.

**Why:**
- The golden set contains short and context-dependent messages such as:
  - "Yes!"
  - "Nope"
  - "check dms"
  - "where it at?"
  - device-only messages
- These messages often cannot be interpreted correctly without previous conversation turns.
- Forcing such messages into a specific support category can produce misleading predictions.

**Evidence:**  
The embedding classifier achieved 0.00 F1 for `OTHER_UNCLEAR`.

**Future improvement:**  
Use an explicit abstention mechanism based on confidence and conversation context.

---

## Decision 8 — Use weak labels for scalable training data

**Decision:**  
Use high-confidence keyword/rule-based labelling to create a larger training corpus instead of manually labelling thousands of examples.

**Why:**
- Manually labelling the entire Spotify dataset would be time-consuming.
- The assignment allows subsampling and expects practical data preparation.
- High-confidence rules can provide useful training signals when carefully designed.

**Training corpus:**  
18,631 cleaned weakly labelled examples.

**Important distinction:**  
Weak labels are not treated as ground truth. The manually labelled golden set remains the evaluation reference.

**Trade-off:**  
Errors in weak labels can propagate into the classifier.

---

## Decision 9 — Use Majority Class as the first baseline

**Decision:**  
Implement a majority-class classifier as the trivial baseline.

**Why:**
- It provides the simplest possible reference.
- It shows how much performance can be obtained without understanding the message.
- It helps demonstrate that the proposed system is doing more than simply predicting the most frequent class.

**Result:**

- Accuracy: `0.11`
- Macro F1: `0.02`

**Majority class:**  
`BILLING_SUBSCRIPTION`

---

## Decision 10 — Use TF-IDF + Logistic Regression as the simple baseline

**Decision:**  
Use TF-IDF with Logistic Regression as the second baseline.

**Configuration:**

- Maximum features: `10,000`
- N-gram range: `(1, 2)`
- Minimum document frequency: `2`
- Classifier: Logistic Regression
- `class_weight="balanced"`

**Why:**
- TF-IDF is a strong and understandable traditional text-classification baseline.
- Logistic Regression is fast and suitable for multi-class classification.
- The baseline is computationally inexpensive and easy to reproduce.

**Result:**

- Accuracy: `0.4899`
- Macro F1: `0.49`
- Weighted F1: `0.43`

**Trade-off:**  
TF-IDF relies heavily on lexical patterns and may struggle with semantically similar messages expressed using different words.

---

## Decision 11 — Use sentence embeddings for the proposed classifier

**Decision:**  
Use `all-MiniLM-L6-v2` sentence embeddings followed by Logistic Regression.

**Why:**
- Customer messages may express the same issue using different vocabulary.
- Sentence embeddings capture semantic similarity better than simple word-frequency representations.
- `all-MiniLM-L6-v2` produces compact 384-dimensional embeddings.
- The model is practical for a student prototype and does not require an external API.

**Pipeline:**

```text
Customer Message
       ↓
all-MiniLM-L6-v2
       ↓
384-dimensional embedding
       ↓
Logistic Regression
       ↓
Predicted Intent + Confidence