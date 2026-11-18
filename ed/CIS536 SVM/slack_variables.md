# Support Vector Machines (SVMs): Quadratic Programming and Slack Variables

## Overview

Support Vector Machines (SVMs) are trained by solving an optimization problem. The optimization is formulated as a **quadratic program (QP)**, which allows us to use well-established mathematical solvers to find a unique and highly accurate solution.

---

# SVM as a Quadratic Programming Problem

For linearly separable data, the SVM objective is:

$$
\min_{w,b} \frac{1}{2}||w||^2
$$

Subject to:

$$
y_i(w^T x_i + b) \geq 1
$$

Where:

- $x_i$ = feature vector
- $y_i \in \{-1, +1\}$ = class label
- $w$ = weight vector
- $b$ = bias term

The objective function contains squared terms, making it a **quadratic optimization problem**, while the constraints are linear.

## Why This Is Useful

- Quadratic programming has been studied extensively.
- Reliable solvers already exist.
- The solution is deterministic.
- Given the same dataset, everyone obtains the same solution.

---

# The Limitation of Hard-Margin SVMs

A hard-margin SVM assumes the dataset is perfectly linearly separable.

Example:

```text
Positive samples (+)
Negative samples (-)

      +   +
   +        +

------------------

      -   -
   -      +   <- mislabeled point
```

The final positive point lies inside the negative region.

In this situation:

- No separating hyperplane exists.
- The optimization constraints cannot all be satisfied.
- The quadratic programming solver reports that the problem is infeasible.

---

# Slack Variables

To handle noisy or mislabeled data, SVM introduces **slack variables**:

$$
\xi_i
$$

The constraints become:

$$
y_i(w^T x_i+b) \geq 1 - \xi_i
$$

with:

$$
\xi_i \geq 0
$$

Each training example receives its own slack variable.

## Interpretation of Slack

### No Violation

$$
\xi_i = 0
$$

The point satisfies the original constraint.

### Inside the Margin

$$
0 < \xi_i < 1
$$

The point is correctly classified but lies within the margin.

### Misclassified

$$
\xi_i > 1
$$

The point crosses the decision boundary and is misclassified.

---

# Why Slack Must Be Penalized

If slack variables were unrestricted, the optimizer could choose extremely large values.

For example:

$$
\xi_i = 1000
$$

Then:

$$
1 - \xi_i = -999
$$

The constraint becomes:

$$
y_i(w^T x_i+b) \geq -999
$$

which is almost always true.

Therefore, slack variables must be kept small.

---

# Soft-Margin SVM Objective Function

The optimization problem becomes:

$$
\min_{w,b,\xi}
\left(
\frac{1}{2}||w||^2
+
C\sum_{i=1}^{n}\xi_i
\right)
$$

Subject to:

$$
y_i(w^T x_i+b) \geq 1-\xi_i
$$

$$
\xi_i \geq 0
$$

The objective now balances:

1. Large margin.
2. Few constraint violations.

---

# Understanding the Hyperparameter C

The parameter $C$ determines how strongly we penalize violations.

## Large C

$$
C \rightarrow \infty
$$

Characteristics:

- Strong penalty for mistakes.
- Behaves similarly to hard-margin SVM.
- Sensitive to outliers.

### Example

A single mislabeled point may force the hyperplane to move significantly.

---

## Moderate C

Characteristics:

- Balanced tradeoff.
- Allows a small number of violations.
- Often achieves the best generalization.

---

## Small C

$$
C \rightarrow 0
$$

Characteristics:

- Violations become inexpensive.
- Constraints become almost meaningless.
- Model underfits.

In the extreme case:

$$
w = 0
$$

which produces a useless classifier.

---

# Mathematical Example

Suppose four points have the following slack values:

| Point | Slack ($\xi$) |
|-------|-------|
| $x_1$ | 0.0 |
| $x_2$ | 0.0 |
| $x_3$ | 0.5 |
| $x_4$ | 1.2 |

Total slack:

$$
\sum \xi_i = 0 + 0 + 0.5 + 1.2 = 1.7
$$

If:

$$
C = 10
$$

Then the penalty term is:

$$
10 \times 1.7 = 17
$$

The optimizer minimizes:

$$
\frac{1}{2}||w||^2 + 17
$$

Larger violations increase the objective value and become less desirable.

---

# Python Example: Training a Linear SVM

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    class_sep=1.2,
    random_state=42
)

model = SVC(kernel="linear", C=1.0)
model.fit(X, y)

print("Support vectors:", len(model.support_vectors_))
```

---

# Python Example: Comparing Different Values of C

```python
from sklearn.svm import SVC

low_c = SVC(kernel="linear", C=0.01)
mid_c = SVC(kernel="linear", C=1)
high_c = SVC(kernel="linear", C=1000)

low_c.fit(X, y)
mid_c.fit(X, y)
high_c.fit(X, y)
```

Expected behavior:

### C = 0.01

```text
Large margin
Many violations allowed
Possible underfitting
```

### C = 1

```text
Balanced solution
Good generalization
```

### C = 1000

```text
Very strict classification
Sensitive to outliers
Possible overfitting
```

---

# Hyperparameter Tuning with Grid Search

Because the optimal value of $C$ depends on the dataset, it is usually determined through validation.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid = GridSearchCV(
    SVC(kernel="linear"),
    param_grid={
        "C": [0.01, 0.1, 1, 10, 100, 1000]
    },
    cv=5
)

grid.fit(X, y)

print("Best C:", grid.best_params_["C"])
```

A common strategy is:

1. Test values across orders of magnitude.
2. Identify the best-performing range.
3. Search more finely around that region.

---

# Key Takeaways

- SVM training is formulated as a quadratic programming problem.
- Hard-margin SVM requires perfect linear separability.
- Real-world datasets often contain noise and mislabeled samples.
- Slack variables ($\xi_i$) allow controlled violations of constraints.
- The soft-margin objective is:

$$
\min \frac{1}{2}||w||^2 + C\sum_i \xi_i
$$

- Large $C$ reduces violations but may overfit.
- Small $C$ allows more violations but may underfit.
- Grid search is commonly used to find the best value of $C$.
