# Support Vector Machines (SVMs): Hinge Loss and Unconstrained Optimization

## Overview

The traditional soft-margin SVM is formulated as a constrained quadratic optimization problem. While this formulation is elegant and can be solved using standard quadratic programming (QP) solvers, it is not always the most practical approach.

For very large datasets or specialized optimization settings, it is often preferable to transform the constrained optimization problem into an unconstrained loss minimization problem.

This transformation leads directly to the **hinge loss formulation** of SVMs.

---

# From Constrained to Unconstrained Optimization

The soft-margin SVM optimization problem is:

$$
\min_{w,b,\xi}
\left(
\frac{1}{2} ||w||^2 + C\sum_{i=1}^{n}\xi_i
\right)
$$

Subject to:

$$
y_i(w^T x_i + b) \geq 1 - \xi_i
$$

$$
\xi_i \geq 0
$$

Where:

- $w$ = model weights
- $b$ = bias term
- $y_i$ = class label (+1 or -1)
- $x_i$ = feature vector
- $\xi_i$ = slack variable
- $C$ = penalty hyperparameter

---

# Solving for the Slack Variable

Starting from the constraint:

$$
y_i(w^T x_i+b) \geq 1-\xi_i
$$

Rearrange to isolate $\xi_i$:

$$
\xi_i \geq 1-y_i(w^T x_i+b)
$$

Notice that the objective function minimizes the sum of all slack variables.

Therefore, the optimization procedure always chooses the smallest feasible value of $\xi_i$.

This means the optimum occurs when:

$$
\xi_i = 1-y_i(w^T x_i+b)
$$

whenever that quantity is positive.

However, we still have the constraint:

$$
\xi_i \geq 0
$$

which means slack can never become negative.

---

# Closed-Form Solution for Slack

Combining both constraints gives:

$$
\xi_i=
\max
\Big(
1-y_i(w^T x_i+b),
0
\Big)
$$

This equation completely determines the value of every slack variable.

Since we now know exactly what $\xi_i$ is, we no longer need to treat it as an optimization variable.

---

# Eliminating the Constraints

Substituting the slack variable expression into the objective function yields:

$$
\min_{w,b}
\left[
\frac{1}{2}||w||^2
+
C\sum_{i=1}^{n}
\max
\Big(
1-y_i(w^T x_i+b),
0
\Big)
\right]
$$

This is the unconstrained form of the soft-margin SVM.

The optimization variables are now only:

- $w$
- $b$

All constraints have been absorbed into the objective function.

---

# The Hinge Loss Function

The expression:

$$
L_{hinge}(x_i,y_i)
=
\max
\Big(
1-y_i(w^T x_i+b),
0
\Big)
$$

is called the **hinge loss**.

The full SVM objective becomes:

$$
\text{Objective}
=
\text{Regularization}
+
C\times\text{Loss}
$$

or:

$$
\frac{1}{2}||w||^2
+
C \sum_i L_{hinge}
$$

---

# Understanding Hinge Loss

Let:

$$
f(x)=w^Tx+b
$$

Then hinge loss is:

$$
\max(1-yf(x),0)
$$

## Case 1: Correctly Classified With Large Margin

Suppose:

$$
yf(x)=2
$$

Then:

$$
\max(1-2,0)
=
0
$$

Loss = 0

The point is correctly classified and safely outside the margin.

---

## Case 2: Inside the Margin

Suppose:

$$
yf(x)=0.7
$$

Then:

$$
\max(1-0.7,0)
=
0.3
$$

Loss = 0.3

The point is correctly classified but lies too close to the decision boundary.

---

## Case 3: Misclassified

Suppose:

$$
yf(x)=-1
$$

Then:

$$
\max(1-(-1),0)
=
2
$$

Loss = 2

The point is on the wrong side of the hyperplane.

---

# Visual Interpretation

```text
                Correct Side

Loss = 0
<------------------------------------------

            Margin Boundary
                  |
                  v
--------------+---------+------------------
             -1         +1

      Loss Increases As Points Move
      Toward Or Across The Boundary
```

Points outside the margin receive no penalty.

Points inside the margin or on the wrong side of the hyperplane receive increasing penalties.

---

# Connection to Linear Regression and Logistic Regression

The unconstrained SVM objective resembles many machine learning optimization problems.

General form:

$$
\text{Objective}
=
\text{Regularizer}
+
\text{Loss Function}
$$

Examples:

## Linear Regression

$$
\text{MSE Loss}
+
\text{Regularization}
$$

## Logistic Regression

$$
\text{Log Loss}
+
\text{Regularization}
$$

## Support Vector Machine

$$
\text{Hinge Loss}
+
\text{Regularization}
$$

The main difference is the choice of loss function.

---

# The Regularization Term

The first part of the objective is:

$$
\frac{1}{2}||w||^2
$$

This term is called the **regularizer**.

Its purpose is to:

- Prevent overly complex solutions.
- Control model complexity.
- Encourage large margins.
- Improve generalization.

In SVMs, minimizing $||w||^2$ is equivalent to maximizing the separation margin between classes.

---

# Mathematical Example

Assume:

$$
y_i(w^T x_i+b)=0.4
$$

Then the hinge loss is:

$$
\max(1-0.4,0)
=
0.6
$$

If:

$$
C=10
$$

Contribution to the objective:

$$
10 \times 0.6 = 6
$$

A point further inside the margin creates a larger penalty.

---

# Python Example: Computing Hinge Loss Manually

```python
import numpy as np

# prediction score
score = 0.4

# true label
y = 1

hinge_loss = max(1 - y * score, 0)

print(hinge_loss)
```

Output:

```text
0.6
```

---

# Python Example: Hinge Loss for Multiple Samples

```python
import numpy as np

scores = np.array([2.0, 0.7, -1.0])
labels = np.array([1, 1, 1])

losses = np.maximum(1 - labels * scores, 0)

print(losses)
```

Output:

```text
[0.0, 0.3, 2.0]
```

---

# Python Example: Training an SVM

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

model = SVC(
    kernel='linear',
    C=1.0
)

model.fit(X, y)

print(model.coef_)
print(model.intercept_)
```

Internally, the model is learning parameters that balance:

1. Low hinge loss.
2. Small weight magnitude.

---

# Key Takeaways

- Soft-margin SVM can be transformed from a constrained optimization problem into an unconstrained one.
- Slack variables can be written in closed form:

$$
\xi_i = \max(1-y_i(w^Tx_i+b),0)
$$

- Substituting this expression removes all constraints.
- The resulting loss function is called **hinge loss**.
- Hinge loss penalizes points inside the margin or on the wrong side of the hyperplane.
- The term $\frac{1}{2}||w||^2$ acts as a regularizer.
- The final SVM objective balances margin maximization and classification accuracy.
