# Random Forest

## Why CART Trees Are Ideal for Bagging

Bagging works with any high-variance classifier, but CART trees are exceptionally well suited for three reasons:

1. **High variance** — A CART tree grown without a depth limit achieves essentially zero training error; it memorises the data completely
2. **No bias** — Because bias is already near zero, reducing variance directly reduces total error
3. **Fast** — CART trees are fast to train and fast to evaluate, so bagging 1,000 of them is computationally practical

---

## How Random Forest Works

Random Forest = **bagged CART trees** + **random feature subsampling at each split**
Random Forests introduce randomness in two stages: 
- sampling the data with replacement
- sampling a subset of features at each split.
The final output of a Random Forest averages the predictions of all the component trees.

At every split during tree construction, instead of searching all $d$ dimensions for the best split, you:

1. Sample $k$ dimensions **at random** (without replacement)
2. Find the best split **only within those $k$ dimensions**
3. Repeat independently for every split in every tree

**Example:** 100 dimensions, $k = 10$ → each split considers 10 randomly chosen features.

### Why subsample features?

Bagging already introduces diversity via bootstrapped data samples (some points included, some excluded). Feature subsampling adds a second source of diversity — trees may split on entirely different dimensions — which further reduces variance when averaged.

**Side effect:** each split is cheaper to evaluate ($k \ll d$ candidates), so training is faster.

---

## Hyperparameters

Random Forest has only **two** hyperparameters (m & k) and forests are extremely *insensitive* to both of these.

| Parameter | Meaning | How to set |
|---|---|---|
| $m$ | Number of trees to average | **Larger is always better** — set as large as compute allows. No trade-off. |
| $k$ | Features sampled per split | Use the rule of thumb: $k = \lfloor\sqrt{d}\rfloor$ |

$$k = \lfloor \sqrt{d} \rfloor$$

For $d = 100$: $k = \lfloor\sqrt{100}\rfloor = 10$

This rule of thumb has theoretical justification and works well in practice. With it fixed, the only remaining decision is how long to let training run.  A good choice for k is k = sqr root of d where d denotes the number of features.  You can set *m* as large as you can afford.

---

## Key Practical Advantages

1. **No feature scaling required** — splits are axis-aligned comparisons, so whether a feature is blood pressure (0–200) or binary gender (0/1) is irrelevant. No normalisation needed.
2. **Naturally non-linear** — handles non-linear decision boundaries without any model selection.
3. **Strong out-of-the-box baseline** — run with defaults on any new dataset; result is typically a reliable upper bound on what is achievable.

---

## Python Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Synthetic dataset: 500 samples, 100 features
X, y = make_classification(n_samples=500, n_features=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

d = X_train.shape[1]          # number of features = 100
k = int(np.sqrt(d))           # rule of thumb: k = sqrt(d) = 10

rf = RandomForestClassifier(
    n_estimators=500,          # m: more trees = better; set as large as affordable
    max_features=k,            # k: features sampled per split
    max_depth=None,            # no depth limit → zero training error per tree (high variance)
    bootstrap=True,            # bagging: each tree trained on a bootstrap sample
    random_state=42,
)

rf.fit(X_train, y_train)
print(f"Test accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")
print(f"d={d}, k={k}, trees={rf.n_estimators}")
```

> **`max_features=k`** is the sklearn parameter that controls random feature subsampling per split — this is the core modification that distinguishes Random Forest from plain bagged trees (`max_features=d`).
