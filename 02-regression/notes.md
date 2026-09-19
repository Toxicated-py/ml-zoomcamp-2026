# Module 2 — Machine Learning for Regression

## 2.1 Car Price Prediction Project

This whole module is one practical project:

> **Predict a car's price from its characteristics.**

### Scenario

A user wants to sell a car but doesn't know what price to set.

They enter things like:

```text
make
model
year
engine
fuel type
transmission
...
```

The model predicts:

```text
price
```

So here:

```text
X = car characteristics
y = MSRP
```

`MSRP` = **Manufacturer Suggested Retail Price**.

### Module 2 plan

We will:

1. Load and clean the data
2. Explore it with **EDA**
3. Build **linear regression**
4. Implement linear regression ourselves
5. Measure error using **RMSE**
6. Create better features
7. Learn **regularization**
8. Train and use the final model

This is where the course becomes much more practical.

The main idea:

```text
car information
      ↓
linear regression model
      ↓
predicted price
```



## 2.2 Data preparation

Download the dataset:

```python
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
```

You can load it directly:

```python
import pandas as pd
import numpy as np

df = pd.read_csv(data)
df.head()
```

Clean column names:

```python
df.columns = df.columns.str.lower().str.replace(' ', '_')
```

Find string columns:

```python
strings = list(df.dtypes[df.dtypes == 'object'].index)
strings
```

Clean their values:

```python
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')
```

Main idea:

```text
Engine Fuel Type
↓
engine_fuel_type

Premium Unleaded
↓
premium_unleaded
```

Clean, consistent data is easier to work with.

---

# 2.3 Exploratory Data Analysis — EDA

Before training anything, understand the data.

Inspect every column:

```python
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()
```

Now inspect the target:

```python
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sns.histplot(df.msrp, bins=50)
```

You will see a **long-tail distribution**.

Most cars are relatively cheap, but a few are extremely expensive.

That can make regression harder.

### Log transformation

Use:

```python
price_logs = np.log1p(df.msrp)

sns.histplot(price_logs, bins=50)
```

`log1p(x)` means:

```text
log(x + 1)
```

Why `+1`?

Because:

```text
log(0)
```

doesn't exist.

To reverse it later:

```python
np.expm1(value)
```

So remember:

```text
np.log1p()  → transform
np.expm1()  → reverse
```

Check missing values:

```python
df.isnull().sum()
```

The dataset contains missing values such as in:

```text
engine_fuel_type
engine_hp
engine_cylinders
number_of_doors
market_category
```

---

# 2.4 Validation framework

We need:

```text
60% train
20% validation
20% test
```

Calculate sizes:

```python
n = len(df)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test
```

For this dataset:

```text
train = 7150
val   = 2382
test  = 2382
```

Don't split sequentially because the dataset has ordering.

Shuffle:

```python
np.random.seed(2)

idx = np.arange(n)
np.random.shuffle(idx)
```

Split:

```python
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]
```

Reset indexes:

```python
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
```

Prepare target:

```python
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)
```

Then remove price from features:

```python
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']
```

Very important.

Otherwise the model could use:

```text
price → predict price
```

which is **data leakage**.

---

# 2.5 Linear regression

For one car:

```text
prediction =
w0
+ feature1 × weight1
+ feature2 × weight2
+ ...
```

Mathematically:

$$
g(x_i)=w_0+\sum_j x_{ij}w_j
$$

Example:

```python
xi = [453, 11, 86]

w0 = 7.17
w = [0.01, 0.04, 0.002]
```

Manual implementation:

```python
def linear_regression(xi):
    pred = w0

    for j in range(len(xi)):
        pred = pred + w[j] * xi[j]

    return pred
```

Result:

```text
12.312
```

Remember: we're predicting **log price**.

Convert to dollars:

```python
np.expm1(12.312)
```

Approximately:

```text
$222,347
```

---

# 2.6 Vector form

The sum:

$$
x_1w_1+x_2w_2+\cdots
$$

is just a **dot product**.

So:

$$
g(x)=w_0+x^Tw
$$

In NumPy:

```python
xi = np.array([453, 11, 86])
w = np.array([0.01, 0.04, 0.002])

xi.dot(w)
```

We can make the bias part of the vector by adding a fictional feature:

```text
x0 = 1
```

Then:

```python
w_new = [w0] + list(w)
xi_new = [1] + list(xi)

np.array(xi_new).dot(np.array(w_new))
```

For the whole dataset:

$$
Xw = y_{pred}
$$

In code:

```python
y_pred = X.dot(w)
```

This is why the linear algebra from Module 1 mattered.

---

# 2.7 Training linear regression

Now the important question:

> Where do the weights come from?

We want:

$$
Xw \approx y
$$

The solution used by the course is the **normal equation**:

$$
w=(X^TX)^{-1}X^Ty
$$

Break it down:

```python
XTX = X.T.dot(X)
```

This is the **Gram matrix**.

Then:

```python
XTX_inv = np.linalg.inv(XTX)
```

Then:

```python
w = XTX_inv.dot(X.T).dot(y)
```

Complete function:

```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])

    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)

    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]
```

Here:

```text
w_full[0]  = bias w0
w_full[1:] = feature weights
```

This is basically what you accidentally implemented in **Homework 1 Q7**.

---

# 2.8 Baseline model

Start simple.

Use five numerical features:

```python
base = [
    'engine_hp',
    'engine_cylinders',
    'highway_mpg',
    'city_mpg',
    'popularity'
]
```

Create `X`:

```python
X_train = df_train[base].values
```

But there are missing values.

Check:

```python
df_train[base].isnull().sum()
```

For now the course fills them with zero:

```python
X_train = df_train[base].fillna(0).values
```

Train:

```python
w0, w = train_linear_regression(X_train, y_train)
```

Predict:

```python
y_pred = w0 + X_train.dot(w)
```

The initial model isn't very good.

That's okay.

This is why it's called a **baseline**.

---

# 2.9 RMSE

We need an objective number for model quality.

RMSE:

$$
RMSE =
\sqrt{
\frac{1}{m}
\sum
(y_i-\hat y_i)^2
}
$$

In plain English:

```text
prediction - actual
↓
square errors
↓
average
↓
square root
```

Implement it:

```python
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()

    return np.sqrt(mse)
```

Training RMSE for the baseline is roughly:

```text
0.755
```

Important:

```text
lower RMSE = better
```

---

# 2.10 Validate the model

Don't judge the model using the data it trained on.

Make one reusable preparation function:

```python
def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)

    X = df_num.values

    return X
```

Train:

```python
X_train = prepare_X(df_train)

w0, w = train_linear_regression(X_train, y_train)
```

Validate:

```python
X_val = prepare_X(df_val)

y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
```

Official result is roughly:

```text
0.762
```

Train ≈ validation.

That's good because there isn't a giant generalization gap.

---

# 2.11 Feature engineering

Now improve the model.

We have:

```text
year = 2010
```

But something more useful is:

```text
age = 2017 - 2010 = 7
```

The dataset was collected in **2017**.

Modify `prepare_X`:

```python
def prepare_X(df):
    df = df.copy()

    df['age'] = 2017 - df.year

    features = base + ['age']

    df_num = df[features]
    df_num = df_num.fillna(0)

    X = df_num.values

    return X
```

Important:

```python
df = df.copy()
```

prevents the function from changing the original dataframe.

Train again:

```python
X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
```

RMSE drops from:

```text
0.76
```

to roughly:

```text
0.517
```

That's a huge improvement.

This demonstrates why **feature engineering matters**.

---

# 2.12 Categorical variables

Features like:

```text
make
fuel type
transmission
vehicle style
```

aren't numbers.

We can't directly give:

```text
Toyota
BMW
Ford
```

to linear regression.

We convert categories into binary features.

Example:

```text
number_of_doors

2 → [1,0,0]
3 → [0,1,0]
4 → [0,0,1]
```

This is called:

**One-Hot Encoding**.

Manually:

```python
for v in [2, 3, 4]:
    df['num_doors_%s' % v] = (
        df.number_of_doors == v
    ).astype(int)
```

Then add those features.

Adding doors only improves RMSE slightly:

```text
0.5172
↓
0.5158
```

Not every feature is equally useful.

### Add manufacturer

Find most common makes:

```python
makes = list(df.make.value_counts().head().index)
```

Then binary encode:

```python
for v in makes:
    df['make_%s' % v] = (df.make == v).astype(int)
```

This improves things further.

Now define categorical variables:

```python
categorical_variables = [
    'make',
    'engine_fuel_type',
    'transmission_type',
    'driven_wheels',
    'market_category',
    'vehicle_size',
    'vehicle_style'
]
```

Take the top five values of each:

```python
categories = {}

for c in categorical_variables:
    categories[c] = list(
        df_train[c].value_counts().head().index
    )
```

Then add them inside `prepare_X`.

But something strange happens.

RMSE explodes to roughly:

```text
41
```

Weights become enormous.

Why?

---

# 2.13 Regularization

Remember:

$$
w=(X^TX)^{-1}X^Ty
$$

The problem is:

$$
(X^TX)^{-1}
$$

If columns in `X` are duplicated or nearly duplicates, the matrix becomes **singular or numerically unstable**.

That can create ridiculous weights like:

```text
3,400,000
-3,400,000
10^15
```

Solution:

Add a small number to the diagonal:

$$
X^TX+rI
$$

In NumPy:

```python
XTX = XTX + r * np.eye(XTX.shape[0])
```

This is **regularization**.

More specifically, this is related to **Ridge regression**.

New training function:

```python
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])

    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)

    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)

    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]
```

With:

```python
r = 0.01
```

RMSE becomes roughly:

```text
0.461
```

Compare:

```text
without regularization → 41
with regularization    → 0.46
```

Huge difference.

---

# Final `prepare_X`

Your feature preparation function should now conceptually look like:

```python
def prepare_X(df):
    df = df.copy()

    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v

        df[feature] = (
            df.number_of_doors == v
        ).astype(int)

        features.append(feature)

    for c, values in categories.items():
        for v in values:
            feature = '%s_%s' % (c, v)

            df[feature] = (
                df[c] == v
            ).astype(int)

            features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)

    X = df_num.values

    return X
```

This function is important.

It ensures train, validation, test and future cars all receive the **same transformations**.

---

# 2.14 Tuning the model

`r` is a **hyperparameter**.

Try different values:

```python
for r in [
    0.0,
    0.00001,
    0.0001,
    0.001,
    0.1,
    1,
    10
]:
    X_train = prepare_X(df_train)

    w0, w = train_linear_regression_reg(
        X_train,
        y_train,
        r=r
    )

    X_val = prepare_X(df_val)

    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val, y_pred)

    print(r, score)
```

The course chooses:

```python
r = 0.001
```

Validation RMSE:

```text
≈ 0.4608
```

Notice what validation is doing here:

```text
r=0.0001 → test
r=0.001  → test
r=0.1    → test
...
```

Actually these are tested on **validation**, not the real test dataset.

We choose the hyperparameter using validation.

Test stays untouched.

---

# 2.15 Final model

Once `r` is chosen, combine:

```text
train + validation
```

because we no longer need validation for tuning.

```python
df_full_train = pd.concat([
    df_train,
    df_val
])

df_full_train = df_full_train.reset_index(drop=True)
```

Prepare features:

```python
X_full_train = prepare_X(df_full_train)
```

Combine targets:

```python
y_full_train = np.concatenate([
    y_train,
    y_val
])
```

Train:

```python
w0, w = train_linear_regression_reg(
    X_full_train,
    y_full_train,
    r=0.001
)
```

Now finally evaluate on **test**:

```python
X_test = prepare_X(df_test)

y_pred = w0 + X_test.dot(w)

rmse(y_test, y_pred)
```

Official result:

```text
≈ 0.4601
```

Validation:

```text
0.4608
```

Test:

```text
0.4601
```

Very close.

That's what we want.

---

# Predict one actual car

Take a car:

```python
car = df_test.iloc[20].to_dict()
car
```

Turn the dictionary back into a one-row DataFrame:

```python
df_small = pd.DataFrame([car])
```

Prepare:

```python
X_small = prepare_X(df_small)
```

Predict:

```python
y_pred = w0 + X_small.dot(w)

y_pred = y_pred[0]
```

Remember this is log price.

Convert:

```python
np.expm1(y_pred)
```

Course example predicts roughly:

```text
$41,459
```

Actual price:

```python
np.expm1(y_test[20])
```

approximately:

```text
$35,000
```

Not perfect, but reasonable.

---

# 2.16 The whole module in one picture

```text
raw car data
      ↓
clean data
      ↓
EDA
      ↓
log-transform MSRP
      ↓
train / validation / test
      ↓
prepare_X()
      ↓
linear regression
      ↓
RMSE
      ↓
feature engineering
      ↓
categorical encoding
      ↓
regularization
      ↓
tune r using validation
      ↓
train + validation
      ↓
final test evaluation
      ↓
predict new car
```

The concepts you absolutely need to remember are:

| Concept             | Meaning                             |
| ------------------- | ----------------------------------- |
| Regression          | Predict a number                    |
| `log1p`             | Compress long-tailed target         |
| `expm1`             | Reverse `log1p`                     |
| Validation          | Tune/compare models                 |
| Test                | Final evaluation                    |
| Linear regression   | `w0 + Xw`                           |
| Normal equation     | `(XᵀX)⁻¹Xᵀy`                        |
| RMSE                | Prediction error; lower is better   |
| Feature engineering | Create useful new features          |
| One-hot encoding    | Turn categories into binary columns |
| Regularization      | Control unstable/huge weights       |
| Hyperparameter      | Setting chosen using validation     |

One especially important progression from the official project is:

```text
Baseline numerical features
RMSE ≈ 0.762

+ age
RMSE ≈ 0.517

+ categorical features without regularization
RMSE ≈ 41   ❌

+ regularization
RMSE ≈ 0.461 ✅

Final test
RMSE ≈ 0.460
```
