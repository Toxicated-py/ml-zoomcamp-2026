Yes. We’ll finish **all of Module 3 in one go**, including the coding flow you should put into `03-classification/notebook.ipynb`.

The current course describes Module 3 as building a **customer-churn classification system with logistic regression**, covering categorical encoding, feature importance/selection, and model interpretation. [GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp?ref=datobra.com\&utm_source=chatgpt.com)

# Module 3 — Machine Learning for Classification

## 3.1 Churn prediction project

Our problem:

> Predict whether a customer will leave the telecom company.

Target:

```text
churn = 1 → customer leaves
churn = 0 → customer stays
```

Unlike regression:

```text
Regression:
car → $25,000

Classification:
customer → churn / no churn
```

More precisely, logistic regression gives us a probability:

```text
P(churn) = 0.82
```

Then we can decide:

```text
>= 0.5 → churn
<  0.5 → stay
```

---

# 3.2 Data preparation

Create:

```text
03-classification/notebook.ipynb
```

Imports:

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
```

Dataset:

```python
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(data)
```

Inspect:

```python
df.head()
```

Better:

```python
df.head().T
```

because there are many columns.

The official dataset contains **21 columns**, including services, contract information, charges, tenure, and the `Churn` target. [GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/03-classification/02-data-preparation.md)

---

## Clean column names

```python
df.columns = df.columns.str.lower().str.replace(' ', '_')
```

Find string columns:

```python
categorical_columns = list(
    df.dtypes[df.dtypes == 'object'].index
)
```

Clean their values:

```python
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
```

Example:

```text
Electronic check
↓
electronic_check
```

---

## `totalcharges` problem

Check types:

```python
df.dtypes
```

You'll notice:

```text
monthlycharges → number
totalcharges   → object
```

But `totalcharges` should be numeric.

Try:

```python
pd.to_numeric(df.totalcharges)
```

It fails because some values aren't proper numbers.

Fix:

```python
df.totalcharges = pd.to_numeric(
    df.totalcharges,
    errors='coerce'
)
```

`errors='coerce'` means:

```text
bad value → NaN
```

Check:

```python
df.totalcharges.isnull().sum()
```

There are **11 missing values** in this dataset after conversion. [GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/03-classification/02-data-preparation.md)

Fill them:

```python
df.totalcharges = df.totalcharges.fillna(0)
```

---

## Convert target to 0/1

Currently:

```text
yes
no
```

Convert:

```python
df.churn = (df.churn == 'yes').astype(int)
```

Now:

```text
yes → 1
no  → 0
```

This is our binary classification target. [GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/03-classification/02-data-preparation.md)

---

# 3.3 Validation framework

This time the course uses Scikit-Learn.

Import:

```python
from sklearn.model_selection import train_test_split
```

First separate 20% for test:

```python
df_full_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=1
)
```

Now split the remaining 80%.

We want:

```text
60% train
20% validation
```

Validation needs to be 25% of the remaining 80%:

```text
0.25 × 0.80 = 0.20
```

So:

```python
df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=1
)
```

Now approximately:

```text
Train      60%
Validation 20%
Test       20%
```

Check:

```python
len(df_train), len(df_val), len(df_test)
```

Reset indexes:

```python
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
```

Extract target:

```python
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values
```

Remove target:

```python
del df_train['churn']
del df_val['churn']
del df_test['churn']
```

Same idea as Module 2:

```text
X → features
y → target
```

And removing `churn` prevents leakage.

---

# 3.4 Exploratory Data Analysis — EDA

EDA means:

> Understand the dataset before building the model.

Work mostly with:

```python
df_full_train
```

Reset it:

```python
df_full_train = df_full_train.reset_index(drop=True)
```

Check missing values:

```python
df_full_train.isnull().sum()
```

---

## Target distribution

```python
df_full_train.churn.value_counts()
```

Better:

```python
df_full_train.churn.value_counts(normalize=True)
```

Because `churn` is `0/1`, its mean is also the churn rate:

```python
global_churn_rate = df_full_train.churn.mean()

global_churn_rate
```

Why?

Suppose:

```text
0
0
1
0
1
```

Mean:

```text
(0 + 0 + 1 + 0 + 1) / 5
= 2 / 5
= 0.4
```

So:

```text
40% churned
```

Very important idea:

```python
df.churn.mean()
```

means **churn rate**.

---

## Numerical vs categorical features

Numerical:

```python
numerical = [
    'tenure',
    'monthlycharges',
    'totalcharges'
]
```

Categorical:

```python
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod'
]
```

Notice:

```text
seniorcitizen
```

contains `0/1`, but conceptually it's categorical.

That's important:

> A column being numeric does not necessarily make it a numerical feature.

Compare Module 2:

```text
number_of_doors = 2 / 3 / 4
```

also looked numerical but represented categories.

---

## Inspect unique values

```python
df_full_train[categorical].nunique()
```

This tells us how many categories each feature contains.

---

# 3.5 Feature importance — churn rate and risk ratio

Now we ask:

> Which features are associated with churn?

Suppose global churn rate:

```python
global_churn_rate = df_full_train.churn.mean()
```

Now look at females:

```python
churn_female = df_full_train[
    df_full_train.gender == 'female'
].churn.mean()
```

And males:

```python
churn_male = df_full_train[
    df_full_train.gender == 'male'
].churn.mean()
```

Compare with global churn.

---

## Difference

```python
churn_female - global_churn_rate
```

Interpretation:

```text
positive → group churns more than average
negative → group churns less than average
```

---

# Risk ratio

More useful:

\[
Risk = \frac{group\ churn\ rate}{global\ churn\ rate}
\]

Example:

```python
churn_female / global_churn_rate
```

Interpretation:

```text
risk = 1
→ same risk as average

risk > 1
→ higher churn risk

risk < 1
→ lower churn risk
```

Example:

```text
Global = 0.25
Month-to-month customers = 0.40

risk = 0.40 / 0.25
     = 1.6
```

Meaning:

> That group churns at about 1.6× the global rate.

---

## Groupby

Instead of manually checking categories:

```python
df_group = df_full_train.groupby(
    'gender'
).churn.agg(['mean', 'count'])
```

Add:

```python
df_group['diff'] = (
    df_group['mean'] - global_churn_rate
)
```

And:

```python
df_group['risk'] = (
    df_group['mean'] / global_churn_rate
)
```

Then:

```python
df_group
```

---

## Do it for every categorical feature

```python
from IPython.display import display
```

Then:

```python
for c in categorical:

    print(c)

    df_group = df_full_train.groupby(c).churn.agg([
        'mean',
        'count'
    ])

    df_group['diff'] = (
        df_group['mean'] -
        global_churn_rate
    )

    df_group['risk'] = (
        df_group['mean'] /
        global_churn_rate
    )

    display(df_group)

    print()
```

This is a very useful EDA technique.

You'll notice features such as **contract type** are much more informative than something like gender.

---

# 3.6 Feature importance — Mutual Information

Churn rate/risk ratio is good for understanding individual categories.

But we need a single number for an entire categorical feature.

That's where **Mutual Information (MI)** comes in.

Import:

```python
from sklearn.metrics import mutual_info_score
```

Example:

```python
mutual_info_score(
    df_full_train.churn,
    df_full_train.contract
)
```

---

## Meaning

Mutual information asks:

> How much does knowing this feature tell me about churn?

Interpretation:

```text
MI = 0
→ feature and churn are independent

larger MI
→ stronger relationship
```

Create function:

```python
def mutual_info_churn_score(series):
    return mutual_info_score(
        series,
        df_full_train.churn
    )
```

Apply to every categorical feature:

```python
mi = df_full_train[categorical].apply(
    mutual_info_churn_score
)
```

Sort:

```python
mi.sort_values(ascending=False)
```

Now you have feature importance rankings.

Conceptually, you might see:

```text
contract             high
online_security      useful
tech_support         useful
internet_service     useful
...
gender               very low
```

Key:

> High mutual information = useful categorical feature for predicting churn.

---

# 3.7 Feature importance — Correlation

Mutual information was mainly for our categorical features.

For numerical features we use **correlation**.

```python
df_full_train[numerical].corrwith(
    df_full_train.churn
)
```

Typical interpretation:

```text
tenure            negative
monthlycharges    positive
totalcharges      negative
```

---

## What does positive correlation mean?

Example:

```text
monthlycharges → +0.2
```

As monthly charges increase:

```text
churn tends to increase
```

---

## Negative correlation

Example:

```text
tenure → -0.35
```

As tenure increases:

```text
churn tends to decrease
```

So long-time customers tend to leave less often.

---

## Correlation strength

Correlation ranges:

\[
-1 \le r \le 1
\]

Interpretation:

```text
r ≈ 1   strong positive
r ≈ -1  strong negative
r ≈ 0   weak/no linear relationship
```

For feature importance, magnitude matters:

```text
|0.7| → strong
|0.4| → moderate
|0.05| → weak
```

But sign tells the direction.

---

## Investigate tenure manually

For example:

```python
df_full_train[
    df_full_train.tenure <= 2
].churn.mean()
```

Compare with:

```python
df_full_train[
    df_full_train.tenure > 2
].churn.mean()
```

This makes the negative correlation intuitive:

```text
new customers → more likely to churn

long-term customers → less likely
```

---

# 3.8 One-Hot Encoding

Our model needs numbers.

But we have:

```text
contract = month-to-month
internetservice = fiber_optic
gender = female
```

We could manually create binary columns like Module 2.

But Scikit-Learn gives us:

```python
from sklearn.feature_extraction import DictVectorizer
```

---

## Convert dataframe to dictionaries

Select categorical + numerical:

```python
train_dicts = df_train[
    categorical + numerical
].to_dict(orient='records')
```

One customer becomes something like:

```python
{
    'gender': 'female',
    'contract': 'month-to-month',
    'tenure': 1,
    'monthlycharges': 70.7,
    ...
}
```

---

## DictVectorizer

```python
dv = DictVectorizer(sparse=False)
```

Fit and transform:

```python
X_train = dv.fit_transform(train_dicts)
```

See feature names:

```python
dv.get_feature_names_out()
```

You will get columns such as:

```text
contract=month-to-month
contract=one_year
contract=two_year

gender=female
gender=male

internetservice=dsl
internetservice=fiber_optic
internetservice=no
```

While numerical columns stay directly:

```text
tenure
monthlycharges
totalcharges
```

That's automatic **one-hot encoding**.

---

## Validation data

Important:

Train:

```python
X_train = dv.fit_transform(train_dicts)
```

Validation:

```python
val_dicts = df_val[
    categorical + numerical
].to_dict(orient='records')

X_val = dv.transform(val_dicts)
```

Notice:

```text
TRAIN → fit_transform()
VAL   → transform()
```

Why?

Because the encoder should **learn categories only from training data**.

Same principle as model fitting.

---

# 3.9 Logistic Regression

This is the main model.

Despite its name:

> Logistic regression is a **classification** algorithm.

It starts similar to linear regression.

Linear score:

\[
z = w_0 + w_1x_1 + w_2x_2 + \cdots + w_nx_n
\]

Or:

\[
z = w_0 + x^T w
\]

But `z` can be:

```text
-10
-2
0
4
100
```

We need a probability between:

```text
0 and 1
```

So logistic regression applies the **sigmoid function**.

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Implement it:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Try:

```python
sigmoid(-5)
sigmoid(0)
sigmoid(5)
```

Approximately:

```text
sigmoid(-5) ≈ 0.007
sigmoid(0)  = 0.5
sigmoid(5)  ≈ 0.993
```

So:

```text
large negative z
→ probability near 0

z = 0
→ probability 0.5

large positive z
→ probability near 1
```

---

## Full logistic regression

\[
P(y=1|x)
=
\sigma(w_0+x^Tw)
\]

For churn:

```text
P(churn = 1)
```

---

## Weight interpretation

Suppose:

```text
w = +0.8
```

Increasing that feature increases `z`, therefore increases churn probability.

If:

```text
w = -0.8
```

the feature lowers the score and generally lowers churn probability.

So:

```text
positive coefficient → pushes toward churn
negative coefficient → pushes away from churn
```

Caution:

Coefficient magnitudes can't always be directly compared when numerical variables have very different scales.

---

# 3.10 Train logistic regression with Scikit-Learn

Import:

```python
from sklearn.linear_model import LogisticRegression
```

Create:

```python
model = LogisticRegression()
```

Train:

```python
model.fit(X_train, y_train)
```

That's it.

Unlike Module 2, we don't implement the entire training algorithm ourselves.

---

## Model parameters

Bias:

```python
model.intercept_[0]
```

Weights:

```python
model.coef_[0]
```

---

## Predict probability

```python
model.predict_proba(X_val)
```

It returns two columns:

```text
P(class 0)   P(class 1)
P(stay)      P(churn)
```

Example:

```text
0.20   0.80
```

means:

```text
20% chance stay
80% chance churn
```

We want class `1`:

```python
y_pred = model.predict_proba(X_val)[:, 1]
```

Now `y_pred` contains churn probabilities.

---

## Turn probabilities into decisions

```python
churn_decision = y_pred >= 0.5
```

Example:

```text
0.21 → False
0.72 → True
0.49 → False
0.88 → True
```

`True` basically means:

```text
predict churn
```

---

## Basic accuracy

```python
(y_val == churn_decision).mean()
```

This gives roughly around **80%** for this project.

But don't get too attached to accuracy.

**Module 4** will show why accuracy alone can be misleading and introduce precision, recall, ROC/AUC, etc.

---

# 3.11 Model interpretation

Now we want to understand:

> Why does the model make its predictions?

Get names:

```python
feature_names = dv.get_feature_names_out()
```

Weights:

```python
weights = model.coef_[0]
```

Combine:

```python
dict(zip(feature_names, weights))
```

Better:

```python
for feature, weight in zip(
    feature_names,
    weights
):
    print(feature, weight)
```

---

## Example interpretation

Suppose:

```text
contract=month-to-month → +0.9
contract=two_year       → -0.9
```

Interpretation:

```text
month-to-month
→ increases churn score

two-year contract
→ decreases churn score
```

Makes sense:

Long-term contracts usually indicate more committed customers.

---

## Bias

Suppose:

```python
model.intercept_[0]
```

is negative.

Then before considering features:

```text
baseline score favors not churning
```

Features move that score up or down.

---

# Small model for easier interpretation

The course demonstrates interpretation more clearly using a few features.

For example:

```python
small = [
    'contract',
    'tenure',
    'monthlycharges'
]
```

Convert:

```python
dicts_train_small = df_train[
    small
].to_dict(orient='records')
```

Vectorize:

```python
dv_small = DictVectorizer(sparse=False)

X_train_small = dv_small.fit_transform(
    dicts_train_small
)
```

Train:

```python
model_small = LogisticRegression()

model_small.fit(
    X_train_small,
    y_train
)
```

Inspect:

```python
dv_small.get_feature_names_out()
```

and:

```python
model_small.coef_[0]
```

Now it becomes easier to understand:

```text
bias
+
contract weight
+
tenure × weight
+
monthlycharges × weight
→ sigmoid
→ churn probability
```

---

# Example manually

Imagine:

```text
bias = -2

month-to-month = +1.2
tenure = -0.04
monthlycharges = +0.03
```

Customer:

```text
month-to-month = 1
tenure = 2
monthlycharges = 80
```

Score:

\[
z=-2+1.2+(2\times-0.04)+(80\times0.03)
\]

Approximately:

```text
z = 1.52
```

Then:

```python
sigmoid(1.52)
```

Approximately:

```text
0.82
```

So:

```text
82% churn probability
```

That's exactly how logistic regression thinks.

---

# 3.12 Using the model

Now we have chosen the model.

As in Module 2:

```text
train + validation
```

should be combined.

We already have:

```python
df_full_train
```

Extract target:

```python
y_full_train = df_full_train.churn.values
```

Remove target:

```python
del df_full_train['churn']
```

---

## Prepare full training set

```python
dicts_full_train = df_full_train[
    categorical + numerical
].to_dict(orient='records')
```

Create a new vectorizer:

```python
dv = DictVectorizer(sparse=False)
```

Fit:

```python
X_full_train = dv.fit_transform(
    dicts_full_train
)
```

Train final model:

```python
model = LogisticRegression()

model.fit(
    X_full_train,
    y_full_train
)
```

---

## Test data

```python
dicts_test = df_test[
    categorical + numerical
].to_dict(orient='records')
```

Transform:

```python
X_test = dv.transform(dicts_test)
```

Predict:

```python
y_pred = model.predict_proba(
    X_test
)[:, 1]
```

Decision:

```python
churn_decision = y_pred >= 0.5
```

Check:

```python
(y_test == churn_decision).mean()
```

Again, test score should be reasonably close to validation score.

That's evidence that the model generalizes.

---

# Predict a completely new customer

This is the real-life use case.

Imagine a request arrives:

```python
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'fiber_optic',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 3,
    'monthlycharges': 95,
    'totalcharges': 285
}
```

Convert:

```python
X_customer = dv.transform([customer])
```

Predict:

```python
model.predict_proba(X_customer)
```

Get churn probability:

```python
churn_probability = model.predict_proba(
    X_customer
)[0, 1]
```

For example:

```text
0.78
```

Then:

```python
churn_probability >= 0.5
```

gives:

```text
True
```

Business could then decide:

```text
high churn risk
↓
offer discount / retention campaign
```

This is the beginning of turning an ML model into an actual product.

---

# 3.13 Module summary

This entire module is:

```text
Raw customer data
        ↓
clean data
        ↓
convert churn yes/no → 1/0
        ↓
train / validation / test
        ↓
EDA
        ↓
feature importance
 ├─ churn rate
 ├─ risk ratio
 ├─ mutual information
 └─ correlation
        ↓
one-hot encode categories
        ↓
DictVectorizer
        ↓
logistic regression
        ↓
churn probability
        ↓
threshold
        ↓
churn / stay
        ↓
interpret coefficients
        ↓
train final model
        ↓
predict new customers
```

The official course sequence includes churn prediction, preparation, validation, EDA, churn rate/risk ratio, mutual information, correlation, one-hot encoding, logistic regression, Scikit-Learn training, model interpretation, and using the model. [GitHub](https://github.com/abhigyan631/mlbookcamp-code?utm_source=chatgpt.com)

---

# 3.14 Explore more

The optional exploration part is where you experiment rather than follow a fixed answer.

Good experiments for this project:

```text
Remove weak features
Compare validation accuracy

Use only high-MI categorical features

Remove customerid entirely

Change logistic regression regularization

Try different thresholds:
0.3
0.4
0.5
0.6
0.7
```

But be careful:

**Choosing the best threshold properly is really Module 4 territory**, because classification evaluation is the next chapter.

---

# Things you must remember from Module 3

| Concept | Meaning |
|---|---|
| Classification | Predict a class |
| Binary classification | Two classes: 0/1 |
| Churn | Customer leaves |
| Churn rate | Mean of a 0/1 churn column |
| Risk ratio | Group churn / global churn |
| Mutual information | Importance for categorical variables |
| Correlation | Direction/strength for numerical variables |
| One-hot encoding | Category → binary columns |
| `DictVectorizer` | Automatically performs encoding |
| Logistic regression | Binary classification model |
| Sigmoid | Converts score into 0–1 |
| `predict_proba()` | Returns class probabilities |
| Threshold | Converts probability to class |
| Positive coefficient | Pushes probability upward |
| Negative coefficient | Pushes probability downward |

And the most important difference:

```text
LINEAR REGRESSION

X
↓
w0 + Xw
↓
number


LOGISTIC REGRESSION

X
↓
w0 + Xw
↓
sigmoid()
↓
probability between 0 and 1
```

The key Scikit-Learn workflow is:

```python
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

churn = y_pred >= 0.5
```