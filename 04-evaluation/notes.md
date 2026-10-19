# Module 4: Evaluation Metrics for Classification
## DataTalksClub Machine Learning Zoomcamp

**Official unit:** https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/04-evaluation

**Official practical notebook:** https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/04-evaluation/notebook.ipynb

---

# Table of Contents

1. Evaluation Metrics: Session Overview
2. Accuracy and the Dummy Model
3. Confusion Table
4. Precision and Recall
5. ROC Curves
6. ROC AUC
7. Cross-Validation
8. Complete Practical Implementation
9. Important Functions and Methods
10. Common Mistakes
11. Quick Revision Questions

---

# 1. Evaluation Metrics: Session Overview

## 1.1 What is model evaluation?

Model evaluation is the process of measuring how well a trained machine learning model performs on data it has not used for training.

Training a model is not enough. A model may perform well on the training data but perform poorly on new data. Therefore, we need appropriate evaluation metrics to determine whether the model is useful.

For example, suppose we build a machine learning model that predicts whether a telecommunications customer will leave the company.

The model predicts one of two outcomes:

- `0`: The customer will stay.
- `1`: The customer will churn.

After training the model, we need to determine how accurately it predicts these outcomes and what kinds of mistakes it makes.

Different metrics answer different questions. Accuracy measures the proportion of correct predictions, precision measures the reliability of positive predictions, recall measures how many actual positive cases are identified, and ROC AUC measures how well the model ranks positive cases above negative cases.

## 1.2 Classification versus regression

Machine learning problems can be divided into different categories.

### Regression

Regression predicts a numerical value.

Examples:

- Predicting the price of a house.
- Predicting the fuel efficiency of a car.
- Predicting monthly sales.

Common regression metrics include mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE).

### Classification

Classification predicts a category or class.

Examples:

- Predicting whether a customer will churn.
- Predicting whether an email is spam.
- Predicting whether a customer will convert into a buyer.
- Predicting whether a transaction is fraudulent.

This module focuses on evaluating classification models.

## 1.3 Binary classification

Binary classification involves two possible classes.

For example:

| Class | Meaning |
|---|---|
| `0` | Customer stays |
| `1` | Customer churns |

The positive class is the class we are interested in detecting. In this example, churn is the positive class.

The positive class does not always mean something good. It simply identifies the outcome that the model is designed to detect.

For a fraud detection model, fraud would normally be the positive class. For a disease detection model, the presence of the disease would normally be the positive class.

## 1.4 Probabilities and predictions

A classification model can produce a probability instead of directly predicting a class.

For example:

| Customer | Predicted probability of churn |
|---|---:|
| A | 0.12 |
| B | 0.35 |
| C | 0.72 |
| D | 0.91 |

These values represent the model's estimated probabilities of the positive class.

To convert probabilities into binary predictions, we choose a threshold.

If the threshold is `0.5`:

- Probability greater than or equal to `0.5` becomes class `1`.
- Probability below `0.5` becomes class `0`.

For example:

| Probability | Threshold | Prediction |
|---:|---:|---:|
| 0.12 | 0.5 | 0 |
| 0.35 | 0.5 | 0 |
| 0.72 | 0.5 | 1 |
| 0.91 | 0.5 | 1 |

The threshold is not necessarily required to be `0.5`. Changing it changes the model's predictions and evaluation metrics.

## 1.5 Why do we need different evaluation metrics?

Imagine a dataset containing 1,000 customers:

- 900 customers will stay.
- 100 customers will churn.

A model that predicts that every customer will stay will correctly classify 900 customers.

Its accuracy will be:

\[
\text{Accuracy}=\frac{900}{1000}=0.90
\]

The model has 90% accuracy, but it identifies none of the customers who will churn.

If our business goal is to identify customers at risk of leaving, this model is not useful despite its high accuracy.

This is why we need more than one evaluation metric.

The important metrics covered in this module are:

1. Accuracy
2. Confusion matrix
3. Precision
4. Recall
5. ROC curve
6. ROC AUC
7. Cross-validation

---

# 2. Accuracy and the Dummy Model

## 2.1 What is accuracy?

Accuracy measures the proportion of predictions that the model gets correct.

The formula is:

\[
\text{Accuracy}=
\frac{\text{Number of Correct Predictions}}
{\text{Total Number of Predictions}}
\]

For binary classification, it can also be written as:

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

Where:

- `TP`: True positives
- `TN`: True negatives
- `FP`: False positives
- `FN`: False negatives

### Example

Suppose a model makes 100 predictions:

- 40 true positives
- 45 true negatives
- 10 false positives
- 5 false negatives

The number of correct predictions is:

\[
TP+TN=40+45=85
\]

Therefore:

\[
\text{Accuracy}=\frac{85}{100}=0.85
\]

The model has 85% accuracy.

## 2.2 Interpreting accuracy

Accuracy ranges from 0 to 1.

| Accuracy | Interpretation |
|---:|---|
| 0.50 | 50% of predictions are correct |
| 0.75 | 75% of predictions are correct |
| 0.90 | 90% of predictions are correct |
| 1.00 | Every prediction is correct |

Higher accuracy is generally desirable, but accuracy alone does not reveal which mistakes the model makes.

A model can have high accuracy while failing to identify the minority class.

## 2.3 What is a dummy model?

A dummy model is a simple baseline model that uses a basic prediction strategy rather than learning meaningful relationships between input features and the target.

The purpose of a dummy model is to establish a minimum performance level that a trained model should improve upon.

One common strategy is to predict the most frequent class for every example.

For example, if 75% of customers do not churn, a majority-class dummy model predicts that every customer will stay.

Its accuracy will be approximately 75%.

A trained model should be compared against this baseline to determine whether it provides useful predictive value.

## 2.4 Why is a dummy model important?

Suppose we obtain the following results:

| Model | Accuracy |
|---|---:|
| Dummy model | 0.75 |
| Logistic regression | 0.81 |

The trained model achieves higher accuracy than the dummy model.

However, we still need to inspect precision, recall, and other metrics to understand whether the improvement is useful for the business problem.

A model can improve accuracy without improving detection of the positive class.

## 2.5 Creating a dummy model in scikit-learn

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train, y_train)

dummy_predictions = dummy_model.predict(X_val)

dummy_accuracy = accuracy_score(
    y_val,
    dummy_predictions
)

print("Dummy accuracy:", dummy_accuracy)
```

The `most_frequent` strategy always predicts the most common class observed in the training labels.

Other available strategies include:

- `most_frequent`: Always predict the most frequent class.
- `prior`: Predict the class probabilities based on training-set class frequencies.
- `stratified`: Randomly predict classes according to their training-set frequencies.
- `uniform`: Randomly predict classes with equal probability.

For a simple majority-class baseline, `most_frequent` is a useful choice.

## 2.6 The class imbalance problem

Class imbalance occurs when one class has many more examples than another class.

For example:

| Class | Number of examples |
|---|---:|
| No churn | 9,000 |
| Churn | 1,000 |
| Total | 10,000 |

Only 10% of customers churn.

A model that predicts no churn for everyone achieves 90% accuracy, even though it never identifies a customer who will churn.

This illustrates an important limitation of accuracy.

When the classes are imbalanced, examine precision, recall, the confusion matrix, and ROC AUC instead of relying exclusively on accuracy.

**Main lesson:** Always compare a trained model with a simple baseline, and do not assume that high accuracy means the model is useful.

---

# 3. Confusion Table

## 3.1 What is a confusion matrix?

A confusion matrix is a table that compares the actual classes with the classes predicted by a classification model.

It shows the number of correct and incorrect predictions for each class.

For binary classification, the four possible outcomes are:

- True positive
- True negative
- False positive
- False negative

## 3.2 The four outcomes

### True Positive (TP)

The actual class is positive, and the model predicts positive.

Example: A customer is going to churn, and the model correctly identifies the customer as at risk.

### True Negative (TN)

The actual class is negative, and the model predicts negative.

Example: A customer will stay, and the model correctly predicts that the customer will stay.

### False Positive (FP)

The actual class is negative, but the model predicts positive.

Example: A customer will stay, but the model incorrectly predicts that the customer will churn.

A false positive is also called a false alarm.

### False Negative (FN)

The actual class is positive, but the model predicts negative.

Example: A customer will churn, but the model incorrectly predicts that the customer will stay.

A false negative means that the model missed an actual positive case.

## 3.3 Confusion matrix layout

With class `0` as negative and class `1` as positive, the standard layout used by scikit-learn is:

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN | FP |
| Actual 1 | FN | TP |

The diagonal cells represent correct predictions. The other cells represent mistakes.

## 3.4 Why are false positives and false negatives important?

The consequences depend on the business problem.

### Example: Customer churn

- False positive: The company may offer a discount to a customer who would have stayed anyway.
- False negative: The company misses an opportunity to retain a customer who will leave.

### Example: Fraud detection

- False positive: A legitimate transaction is incorrectly flagged as fraudulent.
- False negative: A fraudulent transaction is not detected.

### Example: Medical screening

- False positive: A healthy person is incorrectly flagged for further examination.
- False negative: A person with a disease is incorrectly classified as healthy.

A good model is not simply one that makes fewer mistakes overall. It should also manage the types of mistakes that matter most.

## 3.5 Calculating a confusion matrix with scikit-learn

```python
from sklearn.metrics import confusion_matrix

y_pred_binary = (y_pred_proba >= 0.5).astype(int)

cm = confusion_matrix(
    y_val,
    y_pred_binary,
    labels=[0, 1]
)

print(cm)
```

The output has this structure:

```text
[[TN FP]
 [FN TP]]
```

You can extract the four values using:

```python
tn, fp, fn, tp = cm.ravel()

print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)
```

The `labels=[0, 1]` argument makes the intended class order explicit.

## 3.6 Calculating the confusion matrix manually

```python
actual_positive = y_val == 1
actual_negative = y_val == 0

predicted_positive = y_pred_proba >= 0.5
predicted_negative = y_pred_proba < 0.5

tp = (predicted_positive & actual_positive).sum()
tn = (predicted_negative & actual_negative).sum()

fp = (predicted_positive & actual_negative).sum()
fn = (predicted_negative & actual_positive).sum()

print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
```

This code illustrates how the confusion matrix is constructed by comparing actual labels with predicted labels.

## 3.7 Accuracy from the confusion matrix

The formula is:

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

```python
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("Accuracy:", accuracy)
```

Accuracy counts both true positives and true negatives as correct predictions.

## 3.8 Normalized confusion matrices

A normalized confusion matrix expresses counts as proportions rather than raw numbers.

This can make class-specific performance easier to compare when classes have different numbers of examples.

```python
cm_normalized = confusion_matrix(
    y_val,
    y_pred_binary,
    labels=[0, 1],
    normalize="true"
)

print(cm_normalized)
```

With `normalize="true"`, each row is divided by the number of actual examples in that class.

For example, the second row shows how actual positive examples are divided between predicted negative and predicted positive outcomes.

**Main lesson:** The confusion matrix tells us which kinds of mistakes the model makes, not just how many predictions are correct.

---

# 4. Precision and Recall

## 4.1 What is precision?

Precision answers this question:

**Of all examples predicted as positive, what fraction were actually positive?**

The formula is:

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

Precision focuses on the reliability of positive predictions.

High precision means that when the model predicts the positive class, its prediction is usually correct.

### Example

Suppose the model flags 100 customers as likely to churn.

Among these customers:

- 80 actually churn.
- 20 do not churn.

Then:

\[
\text{Precision}=\frac{80}{80+20}=0.80
\]

The precision is 80%.

This means that 80% of the customers flagged by the model actually churned.

### When is precision important?

Precision matters when false positives are expensive.

Examples:

- Fraud detection systems that block transactions.
- Marketing campaigns that offer costly discounts.
- Spam filters where legitimate emails must not be marked as spam.
- Automated systems that trigger expensive investigations.

## 4.2 What is recall?

Recall answers this question:

**Of all examples that were actually positive, what fraction did the model correctly identify?**

The formula is:

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

Recall measures how effectively the model finds the positive class.

### Example

Suppose 200 customers actually churn.

The model correctly identifies 150 of them but misses 50.

Then:

\[
\text{Recall}=\frac{150}{150+50}=0.75
\]

The recall is 75%.

This means the model identifies 75% of all customers who actually churn.

### When is recall important?

Recall matters when missing a positive case is expensive.

Examples:

- Detecting serious diseases.
- Identifying fraudulent transactions.
- Finding customers at high risk of leaving.
- Detecting security threats.

## 4.3 Precision versus recall

Precision and recall measure different aspects of classification performance.

| Metric | Main question | Error emphasized |
|---|---|---|
| Precision | How reliable are positive predictions? | False positives |
| Recall | How many actual positives did we identify? | False negatives |

A model can have high precision but low recall if it makes positive predictions only when it is very confident.

A model can have high recall but low precision if it identifies many positive cases but also produces many false alarms.

Neither metric is universally more important. The choice depends on the cost of each type of error.

## 4.4 Precision and recall using scikit-learn

```python
from sklearn.metrics import precision_score, recall_score

precision = precision_score(
    y_val,
    y_pred_binary,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred_binary,
    zero_division=0
)

print("Precision:", precision)
print("Recall:", recall)
```

The `zero_division=0` argument specifies what to return when a metric's denominator is zero, such as when there are no positive predictions.

## 4.5 The relationship between precision, recall, and threshold

Recall the rule for converting probabilities into predictions:

```python
y_pred_binary = (y_pred_proba >= threshold).astype(int)
```

The threshold controls how easily the model predicts the positive class.

### Lower threshold

A lower threshold makes the model more willing to predict positive.

Usually:

- More examples are predicted positive.
- Recall tends to increase.
- False positives may increase.
- Precision may decrease.

### Higher threshold

A higher threshold makes the model more selective.

Usually:

- Fewer examples are predicted positive.
- False positives may decrease.
- Precision may increase.
- Recall may decrease.

These are typical trends, not strict guarantees for every individual threshold change.

## 4.6 Evaluating several thresholds

```python
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score

thresholds = np.linspace(0, 1, 101)

rows = []

for threshold in thresholds:
    predictions = (
        y_pred_proba >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    rows.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall
    })

df_thresholds = pd.DataFrame(rows)

print(df_thresholds.head())
```

This code calculates precision and recall at thresholds from 0.00 through 1.00 in increments of 0.01.

## 4.7 Plotting precision and recall

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    df_thresholds["threshold"],
    df_thresholds["precision"],
    label="Precision"
)

plt.plot(
    df_thresholds["threshold"],
    df_thresholds["recall"],
    label="Recall"
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall at Different Thresholds")
plt.legend()
plt.grid(True)
plt.show()
```

The graph helps show how the balance between precision and recall changes as the threshold changes.

The point where precision and recall are equal can be useful to identify, but it is not automatically the best operating threshold.

## 4.8 F1 score

F1 score combines precision and recall using their harmonic mean.

The formula is:

\[
F_1=2\cdot\frac{P\cdot R}{P+R}
\]

Where:

- \(P\) is precision.
- \(R\) is recall.

The harmonic mean gives a low score when either precision or recall is low. As a result, F1 rewards models that perform reasonably well on both.

### Example

Suppose:

- Precision = 0.80
- Recall = 0.60

Then:

\[
F_1=2\cdot\frac{0.80\times0.60}{0.80+0.60}
\]

\[
F_1=\frac{0.96}{1.40}\approx0.686
\]

The F1 score is approximately 0.686.

### Calculating F1 in Python

```python
from sklearn.metrics import f1_score

f1 = f1_score(
    y_val,
    y_pred_binary,
    zero_division=0
)

print("F1 score:", f1)
```

### When is F1 useful?

F1 is useful when both false positives and false negatives matter and you want to balance precision and recall.

However, F1 does not include true negatives directly, and it assumes precision and recall should be balanced. If the business problem strongly favors one metric, optimizing F1 may not be the right choice.

## 4.9 Finding the threshold with the highest F1

```python
from sklearn.metrics import f1_score

results = []

for threshold in np.linspace(0, 1, 101):
    predictions = (
        y_pred_proba >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

df_f1 = pd.DataFrame(results)

best_row = df_f1.loc[df_f1["f1"].idxmax()]

print(best_row)
```

This finds the best threshold among the thresholds tested. It does not prove that the threshold is the best possible value over every real number between 0 and 1.

The threshold should be selected using validation data or cross-validation, not by repeatedly checking the final test set.

---

# 5. ROC Curves

## 5.1 What is an ROC curve?

ROC stands for **Receiver Operating Characteristic**.

A ROC curve evaluates a classification model across different probability thresholds.

It plots:

- True Positive Rate (TPR) on the vertical axis.
- False Positive Rate (FPR) on the horizontal axis.

The curve shows the trade-off between identifying positive cases and incorrectly flagging negative cases.

## 5.2 True Positive Rate

The True Positive Rate is the fraction of actual positive cases correctly identified by the model.

\[
TPR=\frac{TP}{TP+FN}
\]

TPR is also called **recall** or **sensitivity**.

A high TPR means the model identifies a large proportion of actual positive cases.

For example, if 100 examples are actually positive and the model correctly identifies 80:

\[
TPR=\frac{80}{100}=0.80
\]

The TPR is 80%.

## 5.3 False Positive Rate

The False Positive Rate is the fraction of actual negative cases incorrectly classified as positive.

\[
FPR=\frac{FP}{FP+TN}
\]

A low FPR means that the model incorrectly flags fewer negative examples.

For example, suppose 900 examples are actually negative, and the model incorrectly predicts 90 of them as positive.

\[
FPR=\frac{90}{900}=0.10
\]

The FPR is 10%.

## 5.4 How is a ROC curve created?

A model generates a probability for each example.

We select a threshold and convert those probabilities into predictions. We then calculate TPR and FPR for that threshold.

Next, we repeat the process using other thresholds.

Each threshold produces a point on the ROC curve.

For example:

| Threshold | TPR | FPR |
|---:|---:|---:|
| 0.90 | 0.20 | 0.02 |
| 0.70 | 0.50 | 0.08 |
| 0.50 | 0.75 | 0.18 |
| 0.30 | 0.90 | 0.40 |
| 0.10 | 0.98 | 0.75 |

These values are illustrative. Actual results depend on the dataset and model.

Lowering the threshold often increases both TPR and FPR because the model predicts more examples as positive.

## 5.5 Interpreting the ROC curve

A ROC curve plots:

\[
x=FPR,\qquad y=TPR
\]

The ideal point is the top-left corner:

\[
FPR=0,\qquad TPR=1
\]

This means the model identifies every positive example without incorrectly flagging any negative example.

### Random classifier

A random-ranking model produces a ROC curve that tends toward the diagonal line from `(0, 0)` to `(1, 1)`.

### Good classifier

A strong model's ROC curve tends to rise toward the top-left corner.

### Poor classifier

A model whose ranking is systematically reversed can perform below the diagonal. Reversing the score direction can reverse its ranking performance.

## 5.6 Creating a ROC curve in Python

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(
    y_val,
    y_pred_proba
)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label="Model"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random baseline"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
```

The `roc_curve()` function returns:

- `fpr`: False positive rates.
- `tpr`: True positive rates.
- `thresholds`: Thresholds used to produce the curve.

## 5.7 Why use probabilities rather than binary predictions?

A ROC curve is intended to show how performance changes across thresholds.

If we supply only binary predictions, we have already committed to one threshold. That provides much less information about the model's ranking ability.

Therefore, supply the model's predicted probabilities or continuous scores:

```python
y_pred_proba = model.predict_proba(X_val)[:, 1]
```

Then use these probabilities to calculate the ROC curve.

## 5.8 ROC curves and threshold selection

ROC curves help compare models and understand the trade-off between TPR and FPR.

For example, a fraud detection system may prioritize a high TPR because missing fraudulent transactions is costly. However, an extremely high TPR may come with an unacceptable FPR, causing many legitimate transactions to be blocked.

The best operating threshold depends on the consequences of these errors.

**Main lesson:** A ROC curve evaluates a model across thresholds rather than judging it at just one threshold.

---

# 6. ROC AUC

## 6.1 What does AUC mean?

AUC stands for **Area Under the Curve**.

In this module, AUC usually refers to the area under the ROC curve, also called ROC AUC.

The ROC curve provides a graphical representation of performance across thresholds. ROC AUC summarizes the ranking performance in one number.

The score generally ranges from 0 to 1.

| ROC AUC | Interpretation |
|---:|---|
| 0.50 | Random-ranking performance in expectation |
| 0.60 | Weak ranking ability |
| 0.70 | Some useful ranking ability |
| 0.80 | Good ranking ability |
| 0.90 | Strong ranking ability |
| 1.00 | Perfect ranking |

These descriptions are rough guidelines, not universal standards. Whether a particular AUC is useful depends on the problem and its costs.

A particular finite sample from a random model can produce an AUC slightly above or below 0.5.

## 6.2 Why is AUC useful?

Accuracy, precision, recall, and F1 usually evaluate predictions at a specific threshold.

ROC AUC evaluates how well the model ranks positive examples above negative examples across the range of possible thresholds.

For example, suppose a model assigns these scores:

| Example | Actual class | Predicted score |
|---|---:|---:|
| A | 1 | 0.90 |
| B | 0 | 0.20 |
| C | 1 | 0.80 |
| D | 0 | 0.30 |

The positive examples receive higher scores than the negative examples. This is good ranking performance.

AUC measures this type of separation without requiring us to select one classification threshold first.

## 6.3 Probabilistic interpretation of ROC AUC

ROC AUC can be interpreted as the probability that a randomly selected positive example receives a higher score than a randomly selected negative example, with ties handled by awarding half credit.

For example, an AUC of 0.85 means that, in a randomly selected positive-negative pair, the model ranks the positive example higher approximately 85% of the time under this interpretation.

It does not mean that 85% of all class predictions are correct.

That distinction is important:

- Accuracy measures the fraction of correct class predictions.
- ROC AUC measures the quality of the model's ranking.

## 6.4 Calculating ROC AUC with scikit-learn

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(
    y_val,
    y_pred_proba
)

print("ROC AUC:", auc)
```

The first argument contains the true binary labels. The second contains the model's continuous scores or positive-class probabilities.

## 6.5 Calculating AUC from a ROC curve

You can calculate the area from the FPR and TPR values:

```python
from sklearn.metrics import auc, roc_curve

fpr, tpr, thresholds = roc_curve(
    y_val,
    y_pred_proba
)

roc_auc = auc(fpr, tpr)

print("ROC AUC:", roc_auc)
```

The two approaches should produce the same area when applied to the same ROC curve and data.

The functions serve slightly different purposes:

- `roc_auc_score(y_true, y_score)` calculates ROC AUC directly from the true labels and scores.
- `auc(x, y)` calculates an area from supplied x- and y-coordinates.

## 6.6 AUC of a random model

A model that ranks positive and negative examples randomly has an expected ROC AUC of 0.5.

Its ROC curve follows the diagonal in expectation.

A random model has no useful ranking ability: it does not consistently assign higher scores to positive examples than to negative examples.

## 6.7 AUC of a perfect model

A perfect ranking model assigns a higher score to every positive example than to every negative example.

Its ROC curve reaches the top-left corner and then follows the top and right edges.

Its ROC AUC is 1.0.

## 6.8 Why can AUC be better than accuracy for model comparison?

Suppose two models produce the following results:

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy at threshold 0.5 | 0.80 | 0.82 |
| ROC AUC | 0.88 | 0.79 |

Model B has higher accuracy at the chosen threshold, while Model A has better ranking performance across thresholds.

If we intend to select a threshold later based on business requirements, ROC AUC may be a more useful metric for comparing these models.

However, ROC AUC is not always sufficient. In strongly imbalanced problems, precision-recall analysis can also be valuable, particularly when positive predictions are rare and the reliability of those predictions matters.

## 6.9 AUC for a single numerical feature

ROC AUC can also be used to evaluate an individual feature as a score.

For example, suppose we want to check whether customer tenure is associated with churn.

```python
from sklearn.metrics import roc_auc_score

score = roc_auc_score(
    y_train,
    df_train["tenure"]
)

print(score)
```

This evaluates how well the numerical feature ranks positive examples above negative examples.

If the score is below 0.5, the feature may be associated with the negative class in the opposite direction.

You can reverse the score direction:

```python
score_reversed = roc_auc_score(
    y_train,
    -df_train["tenure"]
)

print(score_reversed)
```

Negating the score reverses the ranking direction. AUC can be useful for identifying predictive features, but an individual feature's AUC does not necessarily predict how useful it will be in a multivariable model.

**Main lesson:** ROC AUC evaluates ranking quality and allows comparison across thresholds without choosing one threshold first.

---

# 7. Cross-Validation

## 7.1 What is cross-validation?

Cross-validation is a technique for evaluating a machine learning model across multiple training and validation splits.

A single validation split can give a performance estimate that depends on which examples happen to be in the validation set.

Cross-validation reduces dependence on one particular split by repeating the evaluation with different portions of the available training data.

It is commonly used to:

- Estimate model performance.
- Compare different models.
- Tune hyperparameters.
- Understand how much performance varies across data splits.

## 7.2 The basic idea of K-fold cross-validation

In K-fold cross-validation, the dataset is divided into `K` approximately equal-sized folds.

For example, with five-fold cross-validation:

1. Divide the available training data into five folds.
2. Train on folds 2, 3, 4, and 5; validate on fold 1.
3. Train on folds 1, 3, 4, and 5; validate on fold 2.
4. Continue until each fold has been used for validation once.
5. Collect all five evaluation scores.
6. Calculate the mean and standard deviation of the scores.

Each example is used for validation once and for training in the other folds.

## 7.3 Example of five-fold cross-validation

Suppose a dataset contains 1,000 examples.

With five-fold cross-validation, each fold contains approximately 200 examples.

| Run | Training examples | Validation examples |
|---|---:|---:|
| 1 | 800 | 200 |
| 2 | 800 | 200 |
| 3 | 800 | 200 |
| 4 | 800 | 200 |
| 5 | 800 | 200 |

Every run trains on four folds and validates on the remaining fold.

The precise number of examples per fold can differ by one when the total is not evenly divisible by the number of folds.

## 7.4 Why is cross-validation useful?

Imagine a model produces the following ROC AUC scores across five folds:

```text
0.82
0.85
0.84
0.83
0.86
```

The model performs reasonably consistently across the folds.

Now imagine another model produces:

```text
0.70
0.92
0.76
0.90
0.72
```

The second model has more variable results.

The mean describes average performance, while the standard deviation describes variation in the scores across folds.

Neither statistic tells the whole story, but together they provide more information than one score from one split.

## 7.5 Mean score

The mean is the average of the scores obtained across the folds.

\[
\text{Mean}=\frac{1}{K}\sum_{i=1}^{K}s_i
\]

Here:

- \(K\) is the number of folds.
- \(s_i\) is the score for fold \(i\).

For example, if the fold scores are 0.80, 0.82, 0.84, 0.86, and 0.88:

\[
\text{Mean}=\frac{0.80+0.82+0.84+0.86+0.88}{5}=0.84
\]

The mean ROC AUC is 0.84.

## 7.6 Standard deviation

Standard deviation measures how far the fold scores tend to vary from their mean.

A smaller standard deviation indicates that the fold scores are more similar to one another.

A larger standard deviation indicates that the scores vary more between folds.

For example:

- Model A: mean AUC = 0.84, standard deviation = 0.01.
- Model B: mean AUC = 0.85, standard deviation = 0.08.

Model B has a slightly higher mean, but its results vary considerably more across folds.

The lower standard deviation of Model A suggests greater consistency in these particular folds. It does not guarantee that Model A will always generalize better.

When calculating the standard deviation of the observed fold scores in NumPy, `np.std(scores)` uses the population standard deviation by default, dividing by the number of scores rather than one fewer than that number.

## 7.7 KFold in scikit-learn

```python
from sklearn.model_selection import KFold

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)
```

Arguments:

- `n_splits=5`: Use five folds.
- `shuffle=True`: Shuffle the examples before dividing them into folds.
- `random_state=1`: Make the shuffled split reproducible.

Setting a fixed random state allows you to obtain the same fold assignments when you rerun the code with the same data and environment.

## 7.8 Manual cross-validation

When working with a dataset that contains categorical variables, we must ensure that preprocessing is fitted separately within each fold.

The following example uses the Telco customer-churn dataset and `DictVectorizer`.

```python
import numpy as np

from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

scores = []

for fold, (train_idx, val_idx) in enumerate(
    kf.split(df_full_train),
    start=1
):
    train_fold = df_full_train.iloc[train_idx]
    val_fold = df_full_train.iloc[val_idx]

    y_train_fold = train_fold["churn"].values
    y_val_fold = val_fold["churn"].values

    X_train_fold_df = train_fold.drop(
        columns=["churn"]
    )

    X_val_fold_df = val_fold.drop(
        columns=["churn"]
    )

    dv_fold = DictVectorizer(sparse=True)

    X_train_fold = dv_fold.fit_transform(
        X_train_fold_df.to_dict(orient="records")
    )

    X_val_fold = dv_fold.transform(
        X_val_fold_df.to_dict(orient="records")
    )

    model_fold = LogisticRegression(
        solver="liblinear"
    )

    model_fold.fit(
        X_train_fold,
        y_train_fold
    )

    y_pred_fold = model_fold.predict_proba(
        X_val_fold
    )[:, 1]

    score = roc_auc_score(
        y_val_fold,
        y_pred_fold
    )

    scores.append(score)

    print(f"Fold {fold}: {score:.3f}")

print("Mean AUC:", np.mean(scores))
print("Standard deviation:", np.std(scores))
```

This code performs the following operations in each fold:

1. Selects the training and validation portions.
2. Separates the target from the features.
3. Fits a new vectorizer on the fold's training features.
4. Transforms the fold's validation features using that vectorizer.
5. Trains a fresh logistic regression model.
6. Calculates the validation ROC AUC.
7. Saves the score.

The vectorizer must not be fitted on the validation data. Fitting preprocessing on the training portion only is part of preventing information leakage.

## 7.9 Why should preprocessing happen inside each fold?

Suppose we encode categorical variables before cross-validation using the entire dataset.

The vectorizer may learn which categories exist in the validation fold before the model is trained on that fold.

Although unsupervised preprocessing such as one-hot encoding does not use the target labels, fitting preprocessing separately within each fold is the safer, standard practice. It also matters more for preprocessing steps that estimate statistics from the data.

The general rule is:

**Fit every learned preprocessing step on the training portion of a fold, then use it to transform that fold's validation portion.**

This applies to operations such as:

- Imputation using learned statistics.
- Feature scaling.
- Feature selection.
- Dimensionality reduction.
- Encoding categorical variables.

## 7.10 Cross-validation and model selection

Cross-validation can help compare models or hyperparameter values.

For example, we might compare logistic regression models with different values of `C`.

```python
C_values = [0.001, 0.01, 0.1, 1, 10]
```

For each value of `C`:

1. Run the same cross-validation procedure.
2. Record the ROC AUC for each fold.
3. Calculate the mean ROC AUC.
4. Calculate the standard deviation.
5. Compare the results.

The value of `C` with the highest mean ROC AUC may be selected, depending on the model-selection criteria. If two models have similar performance, variability, complexity, and business requirements can also matter.

## 7.11 KFold versus StratifiedKFold

`KFold` divides the data into folds without explicitly preserving the target-class proportions.

`StratifiedKFold` attempts to preserve the proportion of each class in every fold.

For imbalanced classification problems, stratified splitting can help ensure that each fold contains a representative number of positive and negative examples.

The official lesson demonstrates K-fold cross-validation. Follow the precise splitting procedure specified by the lesson or assignment when reproducing its results.

## 7.12 Cross-validation does not replace the final test set

Cross-validation is generally performed on the training portion of the available data.

The final test set should remain separate while we choose the model, tune its hyperparameters, and select the threshold.

After selecting the model and its settings, we can train the final model using the full training portion and evaluate it on the untouched test set.

Repeatedly changing the model based on test-set results makes the test set part of the model-selection process and can produce an overly optimistic estimate of performance.

**Main lesson:** Cross-validation provides a more robust estimate of performance across data splits and supports model selection, while a separate test set is reserved for final evaluation.

---

# 8. Complete Practical Implementation

This section brings the concepts together in a runnable workflow based on the Telco customer-churn example used in the course materials.

The target is `churn`:

- `0`: The customer stayed.
- `1`: The customer churned.

The example uses the dataset from the preceding classification lessons. It is separate from the 2026 lead-conversion homework dataset.

## 8.1 Install or import the required libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
```

## 8.2 Load and prepare the data

```python
data_url = (
    "https://raw.githubusercontent.com/"
    "alexeygrigorev/mlbookcamp-code/master/"
    "chapter-03-churn-prediction/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df = pd.read_csv(data_url)

df.columns = (
    df.columns.str.lower()
    .str.replace(" ", "_")
)

categorical_columns = list(
    df.dtypes[df.dtypes == "object"].index
)

for col in categorical_columns:
    df[col] = (
        df[col].str.lower()
        .str.replace(" ", "_")
    )

df["totalcharges"] = pd.to_numeric(
    df["totalcharges"],
    errors="coerce"
)

df["totalcharges"] = df["totalcharges"].fillna(0)

df["churn"] = (df["churn"] == "yes").astype(int)

# Customer ID is an identifier, not a useful predictive feature.
df = df.drop(columns=["customerid"])

print(df.head())
print(df["churn"].value_counts(normalize=True))
```

The cleaning steps make the column names and categorical values consistent, convert `totalcharges` into a numeric column, fill the missing values created during conversion, and encode the target as 0 or 1.

## 8.3 Split the data

```python
df_full_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=1
)

df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=1
)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train["churn"].values
y_val = df_val["churn"].values

X_train_df = df_train.drop(columns=["churn"])
X_val_df = df_val.drop(columns=["churn"])

print("Training examples:", len(df_train))
print("Validation examples:", len(df_val))
print("Test examples:", len(df_test))
```

The split produces:

- 60% training data.
- 20% validation data.
- 20% test data.

The training set is used to fit the model, the validation set is used for evaluation and model selection, and the test set is reserved for final evaluation.

## 8.4 Encode the features and train logistic regression

```python
dv = DictVectorizer(sparse=True)

X_train = dv.fit_transform(
    X_train_df.to_dict(orient="records")
)

X_val = dv.transform(
    X_val_df.to_dict(orient="records")
)

model = LogisticRegression(
    solver="liblinear",
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_val)[:, 1]

print(y_pred_proba[:10])
```

`DictVectorizer` converts each record into a numerical feature representation. `fit_transform()` learns the feature mapping from training data, while `transform()` applies the same mapping to validation data.

The model's `predict_proba()` method returns class probabilities. The expression `[:, 1]` selects the estimated probability of class 1, which represents churn.

## 8.5 Evaluate the dummy model

```python
dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train, y_train)

dummy_predictions = dummy_model.predict(X_val)

print(
    "Dummy accuracy:",
    accuracy_score(y_val, dummy_predictions)
)
```

This is the baseline. We will compare the trained model against it.

## 8.6 Evaluate the trained model at threshold 0.5

```python
threshold = 0.5

y_pred_binary = (
    y_pred_proba >= threshold
).astype(int)

accuracy = accuracy_score(y_val, y_pred_binary)

precision = precision_score(
    y_val,
    y_pred_binary,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred_binary,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred_binary,
    zero_division=0
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
```

These results describe the model's performance at a threshold of 0.5.

## 8.7 Inspect the confusion matrix

```python
cm = confusion_matrix(
    y_val,
    y_pred_binary,
    labels=[0, 1]
)

print(cm)

tn, fp, fn, tp = cm.ravel()

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)
```

Check that the four counts agree with the accuracy, precision, and recall calculated above.

## 8.8 Plot the ROC curve and calculate AUC

```python
fpr, tpr, thresholds = roc_curve(
    y_val,
    y_pred_proba
)

roc_auc = roc_auc_score(
    y_val,
    y_pred_proba
)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Model AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random baseline"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

print("ROC AUC:", roc_auc)
```

## 8.9 Find the best F1 threshold

```python
threshold_results = []

for threshold in np.linspace(0, 1, 101):
    predictions = (
        y_pred_proba >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_val,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_val,
            predictions,
            zero_division=0
        )
    })

df_thresholds = pd.DataFrame(threshold_results)

best_f1_row = df_thresholds.loc[
    df_thresholds["f1"].idxmax()
]

print(best_f1_row)
```

The selected threshold maximizes F1 among the 101 tested thresholds. Choose thresholds using validation data or cross-validation, not the final test set.

## 8.10 Run five-fold cross-validation

```python
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

cv_scores = []

for fold, (train_idx, val_idx) in enumerate(
    kf.split(df_full_train),
    start=1
):
    train_fold = df_full_train.iloc[train_idx]
    val_fold = df_full_train.iloc[val_idx]

    y_train_fold = train_fold["churn"].values
    y_val_fold = val_fold["churn"].values

    X_train_fold_df = train_fold.drop(
        columns=["churn"]
    )

    X_val_fold_df = val_fold.drop(
        columns=["churn"]
    )

    dv_fold = DictVectorizer(sparse=True)

    X_train_fold = dv_fold.fit_transform(
        X_train_fold_df.to_dict(orient="records")
    )

    X_val_fold = dv_fold.transform(
        X_val_fold_df.to_dict(orient="records")
    )

    model_fold = LogisticRegression(
        solver="liblinear",
        max_iter=1000
    )

    model_fold.fit(
        X_train_fold,
        y_train_fold
    )

    predictions_fold = model_fold.predict_proba(
        X_val_fold
    )[:, 1]

    score = roc_auc_score(
        y_val_fold,
        predictions_fold
    )

    cv_scores.append(score)

    print(f"Fold {fold}: {score:.3f}")

print("Mean ROC AUC:", np.mean(cv_scores))
print("Standard deviation:", np.std(cv_scores))
```

This estimates the model's performance across five different validation folds. The mean gives the average ROC AUC, while the standard deviation measures variation across the five fold scores.

## 8.11 Final test evaluation

After selecting your model and threshold using the training and validation process, fit a final model on the full training portion and evaluate the untouched test set once.

```python
y_full_train = df_full_train["churn"].values
y_test = df_test["churn"].values

X_full_train_df = df_full_train.drop(
    columns=["churn"]
)

X_test_df = df_test.drop(
    columns=["churn"]
)

dv_final = DictVectorizer(sparse=True)

X_full_train = dv_final.fit_transform(
    X_full_train_df.to_dict(orient="records")
)

X_test = dv_final.transform(
    X_test_df.to_dict(orient="records")
)

final_model = LogisticRegression(
    solver="liblinear",
    max_iter=1000
)

final_model.fit(X_full_train, y_full_train)

y_test_proba = final_model.predict_proba(X_test)[:, 1]

# Example: evaluate at the preselected threshold of 0.5.
# If you selected another threshold using validation data,
# use that threshold here instead.
test_threshold = 0.5

y_test_pred = (
    y_test_proba >= test_threshold
).astype(int)

print(
    "Test accuracy:",
    accuracy_score(y_test, y_test_pred)
)

print(
    "Test precision:",
    precision_score(y_test, y_test_pred, zero_division=0)
)

print(
    "Test recall:",
    recall_score(y_test, y_test_pred, zero_division=0)
)

print(
    "Test F1:",
    f1_score(y_test, y_test_pred, zero_division=0)
)

print(
    "Test ROC AUC:",
    roc_auc_score(y_test, y_test_proba)
)
```

The test results provide the final performance estimate for the chosen model. Avoid repeatedly modifying the model after looking at these results.

---

# 9. Important Functions and Methods

| Function or method | Purpose |
|---|---|
| `accuracy_score()` | Calculates the proportion of correct class predictions |
| `confusion_matrix()` | Counts true positives, true negatives, false positives, and false negatives |
| `precision_score()` | Calculates precision |
| `recall_score()` | Calculates recall |
| `f1_score()` | Calculates F1 score |
| `roc_curve()` | Calculates FPR and TPR at different thresholds |
| `roc_auc_score()` | Calculates the area under the ROC curve from true labels and scores |
| `auc()` | Calculates an area from supplied x- and y-coordinates |
| `KFold()` | Creates K-fold cross-validation splits |
| `cross_val_score()` | Evaluates an estimator using cross-validation |
| `DummyClassifier()` | Creates a simple baseline classifier |
| `predict_proba()` | Returns predicted class probabilities |
| `DictVectorizer()` | Converts dictionaries of feature values into a numerical feature matrix |
| `np.mean()` | Calculates the mean |
| `np.std()` | Calculates the population standard deviation by default |
| `train_test_split()` | Splits data into training and testing portions |

---

# 10. Common Mistakes

## Mistake 1: Evaluating only accuracy

High accuracy does not guarantee that the model identifies the positive class effectively, particularly when classes are imbalanced.

**Correction:** Compare accuracy with a baseline and inspect the confusion matrix, precision, recall, and ROC AUC.

## Mistake 2: Confusing precision with recall

Precision measures the reliability of positive predictions. Recall measures the fraction of actual positives found.

**Correction:** Remember that precision focuses on predicted positives, while recall focuses on actual positives.

## Mistake 3: Reversing false positives and false negatives

A false positive means predicting positive when the actual class is negative. A false negative means predicting negative when the actual class is positive.

**Correction:** Compare the actual class and predicted class separately.

## Mistake 4: Assuming the threshold must always be 0.5

The threshold changes which examples are predicted positive. A threshold of 0.5 is common, but it is not universally optimal.

**Correction:** Select a threshold based on validation results and the business costs of different mistakes.

## Mistake 5: Using binary predictions to calculate ROC AUC

Binary predictions contain only the result at one threshold.

**Correction:** Use predicted probabilities or continuous scores to evaluate ranking performance across thresholds.

## Mistake 6: Interpreting AUC as accuracy

An AUC of 0.85 does not mean that 85% of class predictions are correct.

**Correction:** Interpret ROC AUC as a ranking metric, not as the fraction of correct class predictions.

## Mistake 7: Fitting preprocessing on validation data

Fitting the vectorizer or other learned preprocessing using the validation portion can introduce information leakage.

**Correction:** Fit preprocessing on the training portion and use the fitted transformation on validation data.

## Mistake 8: Tuning a model repeatedly on the test set

Repeated test-set evaluation influences model selection and makes the test set less independent.

**Correction:** Use training data and cross-validation for model selection, validation data for threshold selection when appropriate, and the test set for the final evaluation.

## Mistake 9: Forgetting that cross-validation produces several scores

A single fold score is not the complete cross-validation result.

**Correction:** Examine the mean and standard deviation of the scores across all folds.

## Mistake 10: Assuming the highest F1 threshold is always best

F1 balances precision and recall, but it does not directly account for business costs or true negatives.

**Correction:** Select a metric and threshold that match the real problem.

# Final Summary

The central purpose of this module is to understand that classification models must be evaluated using metrics appropriate to the problem.

- **Accuracy** measures the proportion of correct predictions.
- **A dummy model** establishes a baseline.
- **A confusion matrix** reveals the different types of correct and incorrect predictions.
- **Precision** measures the reliability of positive predictions.
- **Recall** measures how many actual positive cases are identified.
- **F1 score** balances precision and recall.
- **ROC curves** show the trade-off between true positive rate and false positive rate across thresholds.
- **ROC AUC** summarizes ranking performance in a single number.
- **Cross-validation** evaluates a model across multiple training and validation splits.
- **The final test set** provides a final estimate of the selected model's performance.

The most important lesson is that there is no single evaluation metric that is always best. The appropriate metric depends on class balance, the types of errors the model makes, and the business problem being solved.

