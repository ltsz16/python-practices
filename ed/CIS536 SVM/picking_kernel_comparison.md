# Kernel Function Comparison Cheat Sheet

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              KERNEL FUNCTION COMPARISON                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Kernel | Formula | What It Measures | When to Use | Pros | Cons |
|----------|----------|----------|----------|----------|----------|
| **Linear Kernel** | `K(x,z) = xᵀz` | Standard dot-product similarity | Data is approximately linear, very high-dimensional data, text classification, debugging | Fastest, simplest, interpretable | Cannot capture nonlinear relationships |
| **Polynomial Kernel** | `K(x,z) = (xᵀz + c)^d` | Similarity including polynomial feature interactions | Interactions between features matter, known polynomial relationships | Captures feature interactions without explicit expansion | Can become computationally expensive for high degrees |
| **RBF (Gaussian) Kernel** | ![alt text](image-25.png)| Distance-based similarity | Unknown nonlinear relationships, general-purpose nonlinear classifier | Extremely flexible, universal approximator, most popular kernel | More parameters to tune, less interpretable |
| **Sigmoid Kernel** | `K(x,z) = tanh(axᵀz + c)` | Neural-network-like similarity | Rarely used today, historically connected to neural networks | Can model nonlinear boundaries | Not always PSD, typically outperformed by RBF |
| **Custom Composite Kernel** | `K = aK₁ + bK₂` | Combination of multiple similarity measures | Domain-specific applications | Combines strengths of multiple kernels | More complex to design and tune |

---

# Visual Intuition

## Linear Kernel

### Decision Boundary

```text
Class A      |      Class B

ooooooo      |      xxxxxxx
ooooooo      |      xxxxxxx
ooooooo      |      xxxxxxx
```

Learns:

```text
---------------
 Straight Line
---------------
```

### Implicit Feature Space

```text
Original Features
        ↓
No Expansion
        ↓
Linear Boundary
```

---

## Polynomial Kernel

### Decision Boundary

```text
oooooooo
ooooooo
oooo
  ooo
     xxx
       xxxx
         xxxxx
```

Learns:

```text
Curved Polynomial Boundary
```

### Implicit Features

Degree 2:

```text
(x₁,x₂)

→

(x₁,x₂,x₁²,x₂²,x₁x₂)
```

Degree 3:

```text
(x₁,x₂)

→

(x₁,x₂,x₁²,x₂²,x₁x₂,
 x₁³,x₂³,x₁²x₂,x₁x₂²)
```

Useful when feature interactions matter.

---

## RBF Kernel

### Decision Boundary

```text
       ooooo
     oo     oo
    oo  xxx  oo
     oo     oo
       ooooo
```

Learns:

```text
Arbitrarily Curved
Nonlinear Boundaries
```

### Similarity Function

```text
Distance Small
   ↓
Similarity ≈ 1

Distance Large
   ↓
Similarity ≈ 0
```

Think:

> Nearby points should behave similarly.

---

# Which Kernel Should I Use?

```text
Is the problem mostly linear?
           |
           +-- YES
           |      ↓
           |   Linear Kernel
           |
           NO
           |
Do you know feature interactions
are polynomial in nature?
           |
           +-- YES
           |      ↓
           |   Polynomial Kernel
           |
           NO
           |
Unknown nonlinear structure?
           |
           +-- YES
                  ↓
             RBF Kernel
```

---

# Exam Keywords → Kernel

| Exam Keywords | Likely Answer |
|---------------|---------------|
| "Standard dot product" | Linear Kernel |
| "Linear classifier" | Linear Kernel |
| "Text classification" | Linear Kernel |
| "Polynomial features" | Polynomial Kernel |
| "x², x³, interaction terms" | Polynomial Kernel |
| "Feature expansion" | Polynomial Kernel |
| "Most commonly used nonlinear kernel" | RBF Kernel |
| "Universal approximator" | RBF Kernel |
| "Distance-based similarity" | RBF Kernel |
| "Highly nonlinear boundary" | RBF Kernel |
| "Unknown relationship" | RBF Kernel |

---

# Quick Comparison

```text
                 LINEAR       POLYNOMIAL       RBF
─────────────────────────────────────────────────────────
Linear Data        ★★★★★         ★★★★          ★★★
Nonlinear Data      ★             ★★★          ★★★★★
Speed              ★★★★★          ★★★          ★★★
Interpretability   ★★★★★          ★★★          ★
Flexibility         ★             ★★★★         ★★★★★
Ease of Tuning     ★★★★★          ★★★          ★★
```

---

# Practical Rule of Thumb

```text
Need speed?
    → Linear Kernel

Know interactions are polynomial?
    → Polynomial Kernel

Don't know the true relationship?
    → RBF Kernel

Taking an ML exam and unsure?
    → RBF Kernel is usually the safest nonlinear answer.
```

## One-Line Summary

```text
Linear Kernel      = Straight-line boundaries
Polynomial Kernel = Polynomial feature interactions
RBF Kernel        = Flexible nonlinear boundaries based on distance
```

## Python Examples
# Python Examples for Common SVM Kernels (scikit-learn)

Common imports used by all examples:

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Sample dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

# 1. Linear Kernel

## Mathematical Definition

\[
K(x,z) = x^T z
\]

Uses the standard dot product.

## When to Use

- Data is approximately linearly separable
- Text classification (high-dimensional sparse features)
- Fast training and prediction
- Need interpretability

## Python Example

```python
from sklearn.svm import SVC

model = SVC(kernel='linear')

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
```

---

## Accessing Feature Weights

A nice property of linear SVMs:

```python
print(model.coef_)
```

This gives the learned weights for each feature.

---

# 2. Polynomial Kernel

## Mathematical Definition

\[
K(x,z) = (x^T z + c)^d
\]

Where:

- \(d\) = degree
- \(c\) = coefficient

---

## When to Use

- Polynomial relationships exist
- Feature interactions matter
- Want implicit feature expansion

Examples:

```text
x²
x³
xy
x²y
```

---

## Degree 2 Polynomial Kernel

```python
model = SVC(
    kernel='poly',
    degree=2
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))
```

---

## Degree 3 Polynomial Kernel

```python
model = SVC(
    kernel='poly',
    degree=3
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))
```

---

## Custom Polynomial Parameters

```python
model = SVC(
    kernel='poly',
    degree=4,
    coef0=1,
    gamma='scale'
)

model.fit(X_train, y_train)
```

---

# 3. RBF (Gaussian) Kernel

## Mathematical Definition

\[
K(x,z)
=
\exp
\left(
-\frac{\|x-z\|^2}{2\sigma^2}
\right)
\]

Measures similarity based on distance.

---

## When to Use

- Unknown nonlinear relationships
- General-purpose kernel
- Most common SVM kernel
- Good default choice

---

## Basic RBF SVM

```python
model = SVC(
    kernel='rbf'
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))
```

---

## RBF with Hyperparameter Tuning

```python
model = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))
```

---

## Understanding Parameters

### C

Controls regularization.

```text
Small C
    More regularization

Large C
    Less regularization
```

---

### Gamma

Controls the influence range.

```text
Small Gamma
    Smooth decision boundary

Large Gamma
    Complex decision boundary
```

---

# 4. Sigmoid Kernel

## Mathematical Definition

\[
K(x,z) = \tanh(ax^Tz + c)
\]

Historically related to neural networks.

---

## When to Use

Rare today.

Usually RBF performs better.

---

## Python Example

```python
model = SVC(
    kernel='sigmoid'
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))
```

---

# 5. Custom Kernel

Sometimes you can define your own kernel.

---

## Example Custom Kernel

\[
K(x,z)
=
(x^Tz)^2
\]

---

### Python Function

```python
import numpy as np

def custom_kernel(X, Y):
    return (np.dot(X, Y.T)) ** 2
```

---

### Using It

```python
model = SVC(
    kernel=custom_kernel
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))
```

---

# Comparing Kernels on the Same Dataset

```python
kernels = [
    "linear",
    "poly",
    "rbf",
    "sigmoid"
]

for kernel in kernels:

    model = SVC(kernel=kernel)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)

    print(f"{kernel}: {score:.3f}")
```

Example output:

```text
linear: 0.840
poly:   0.860
rbf:    0.910
sigmoid:0.810
```

---

# Choosing a Kernel

```python
if data_is_linear:
    kernel = "linear"

elif polynomial_relationships_exist:
    kernel = "poly"

else:
    kernel = "rbf"
```

---

# Regression Versions (SVR)

The same kernels can be used with Support Vector Regression.

## Linear SVR

```python
from sklearn.svm import SVR

model = SVR(kernel='linear')

model.fit(X_train, y_train)
```

---

## Polynomial SVR

```python
model = SVR(
    kernel='poly',
    degree=3
)

model.fit(X_train, y_train)
```

---

## RBF SVR

```python
model = SVR(
    kernel='rbf'
)

model.fit(X_train, y_train)
```

---


# Kernel Function Implementations in Pure Python (NumPy) Non-scikit examples

These examples implement kernel functions directly using NumPy.

```python
import numpy as np
```

---

# Linear Kernel

## Mathematical Definition

\[
K(x, z) = x^T z
\]

The standard dot product.

## When to Use

- Data is approximately linear
- High-dimensional sparse data (text classification)
- Fastest kernel
- Most interpretable

## Implementation

```python
# single pair of vectors
def linear_kernel(x, z):
    return np.dot(x, z)
```
if x,z are matricies make sure to do:
```np.dot(x,z.T)```

## Example

```python
x = np.array([1, 2, 3])
z = np.array([4, 5, 6])

print(linear_kernel(x, z))
```

Output:

```text
32
```

---

# Polynomial Kernel

## Mathematical Definition

\[
K(x, z) = (x^T z + c)^d
\]

where:

- d = polynomial degree
- c = constant offset

## When to Use

- Polynomial relationships exist
- Feature interactions matter
- Need terms such as:
  - x²
  - x³
  - xy
  - x²y

## Implementation

```python
def polynomial_kernel(
    x,
    z,
    degree=2,
    c=1
):
    return (np.dot(x, z) + c) ** degree
    # Use .T if matrix
```

## Example

```python
x = np.array([1, 2])
z = np.array([3, 4])

print(
    polynomial_kernel(
        x,
        z,
        degree=2
    )
)
```

Output:

```text
144
```

---

# RBF (Gaussian) Kernel

## Mathematical Definition

\[
K(x,z)
=
\exp\left(
-\frac{||x-z||^2}
{2\sigma^2}
\right)
\]

## When to Use

- Unknown nonlinear relationships
- General-purpose nonlinear classification
- Most common kernel in practice
- Excellent default choice when unsure

## Implementation

```python
def rbf_kernel(
    x,
    z,
    sigma=1.0
):
    squared_distance = (
        np.linalg.norm(x - z) ** 2
    )

    return np.exp(
        -squared_distance /
        (2 * sigma**2)
    )
```

## Example

```python
x = np.array([1, 2])
z = np.array([1.5, 2.5])

print(
    rbf_kernel(
        x,
        z,
        sigma=1.0
    )
)
```

Output:

```text
0.778800783...
```

Interpretation:

```text
Close points  -> Similarity near 1

Far points    -> Similarity near 0
```

```
# If matrix
# square distance between every pair of rows ||x_subi - z_subi||^2
sq_distance = np.sum((X[:, None, :] - Z[None, :, :])**2, axis=2)
# This can also be written as a loop (inefficient) as:
sq_distance = np.zeros((x.shape[0], z.shape{0}))
for i in range(x.shape[0]):
    for j in range(z.shape[0]):
        sq_distance[i,j] = np.sum((x[i,:] - z[j,:])**2)
    ## end loop
answer = np.exp(
        -squared_distance /
        (2 * sigma**2)
```

---

# Sigmoid Kernel

## Mathematical Definition

\[
K(x,z)
=
\tanh(a(x^T z)+c)
\]

## When to Use

- Historically related to neural networks
- Rarely used today
- Usually outperformed by RBF kernels

## Implementation

```python
def sigmoid_kernel(
    x,
    z,
    a=0.1,
    c=0
):
    return np.tanh(
        a * np.dot(x, z) + c
    )
```

## Example

```python
x = np.array([1, 2])
z = np.array([3, 4])

print(
    sigmoid_kernel(
        x,
        z
    )
)
```

---

# Building a Kernel Matrix

For kernel methods we often compute:

\[
K_{ij}
=
K(x_i, x_j)
\]

for all training examples.

## Generic Implementation

```python
def kernel_matrix(
    X,
    kernel_function
):
    n = len(X)

    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_function(
                X[i],
                X[j]
            )

    return K
```

---

## Example Dataset

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
```

---

## Linear Kernel Matrix

```python
K = kernel_matrix(
    X,
    linear_kernel
)

print(K)
```

Output:

```text
[[ 5. 11. 17.]
 [11. 25. 39.]
 [17. 39. 61.]]
```

---

# Explicit Polynomial Feature Expansion

A degree-2 polynomial expansion can look like:

\[
(x_1,x_2)
\rightarrow
(x_1,x_2,x_1^2,x_2^2,x_1x_2)
\]

## Implementation

```python
def quadratic_feature_map(x):

    x1 = x[0]
    x2 = x[1]

    return np.array([
        x1,
        x2,
        x1**2,
        x2**2,
        x1 * x2
    ])
```

## Example

```python
x = np.array([2, 3])

print(
    quadratic_feature_map(x)
)
```

Output:

```text
[2 3 4 9 6]
```

---

# Minimal Kernelized Classifier

Kernel classifiers often make predictions using:

\[
f(x)
=
\sum_i
\alpha_i K(x_i,x)
+
b
\]

where:

- \(x_i\) are support vectors
- \(\alpha_i\) are learned coefficients
- \(b\) is the bias

---

## Example Parameters

```python
alphas = np.array([
    0.5,
    -0.8
])

support_vectors = np.array([
    [1, 2],
    [3, 4]
])

bias = 0.2
```

---

## Prediction Function

```python
def predict(x):

    total = bias

    for alpha, sv in zip(
        alphas,
        support_vectors
    ):

        total += (
            alpha *
            rbf_kernel(
                sv,
                x,
                sigma=1.0
            )
        )

    return np.sign(total)
```

---

## Example Prediction

```python
x = np.array([2, 2])

print(
    predict(x)
)
```

Output:

```text
1
```

Meaning:

```text
Positive Class
```

---

# Quick Reference

## Linear Kernel

```python
linear_kernel(x, z)
```

Formula:

```text
xᵀz
```

Use when:

```text
Data is approximately linear
```

---

## Polynomial Kernel

```python
polynomial_kernel(
    x,
    z,
    degree=3
)
```

Formula:

```text
(xᵀz + c)^d
```

Use when:

```text
Feature interactions matter
```

---

## RBF Kernel

```python
rbf_kernel(
    x,
    z,
    sigma=1.0
)
```

Formula:

```text
exp(-||x-z||²/(2σ²))
```

Use when:

```text
You need a flexible nonlinear model
```

---

## Sigmoid Kernel

```python
sigmoid_kernel(
    x,
    z,
    a=0.1
)
```

Formula:

```text
tanh(a(xᵀz)+c)
```

Use when:

```text
Experimenting with neural-network-like similarity functions
```

---

# Exam Cheat Sheet

```text
Linear Kernel
    Fastest
    Linear decision boundary

Polynomial Kernel
    Polynomial feature interactions
    x², x³, xy terms

RBF Kernel
    Most popular nonlinear kernel
    Universal approximator
    Best default nonlinear choice

Sigmoid Kernel
    Historically interesting
    Rarely used in practice
```

# One-Line Summary

```text
Linear      -> Linear boundary
Polynomial  -> Polynomial feature interactions
RBF         -> Flexible nonlinear boundary
Sigmoid     -> Neural-network-like similarity
```