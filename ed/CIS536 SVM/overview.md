# Support Vector Machines: Comprehensive Course Overview

## Course Introduction

This course covers **Support Vector Machines (SVMs)** and **Kernel Methods**, which are a powerful class of algorithms for both linear and non-linear classification problems. SVMs combine mathematical elegance with exceptional practical performance and are widely used in machine learning applications.

**Key Learning Outcomes:**
- Find maximum margin classifiers using linear SVMs
- Understand empirical risk minimization (ERM) framework with loss functions and regularizers
- Apply the kernel trick to train non-linear classifiers
- Implement and optimize SVM algorithms

---

## Part 1: Linear SVMs and Maximum Margin Classifiers

### 1.1 The Maximum Margin Principle

**Core Concept:** The "best" hyperplane for classification is not just one that separates the data, but the one that **maximizes the margin**—the distance between the hyperplane and the closest data points on either side.

**Why Maximum Margin?**
- A Perceptron simply finds *any* hyperplane that separates the classes
- An SVM finds the *optimal* hyperplane positioned in the middle
- This provides maximum "wiggle room" for test points that differ slightly from training points
- Better generalization to unseen data

**Mathematical Formulation:**

For a binary classification problem with data points $(x_i, y_i)$ where $y_i \in \{-1, +1\}$:

The hyperplane is defined by: $w^T x + b = 0$

The **margin** $\gamma$ is:
$$\gamma(w, b) = \frac{1}{\|w\|_2} \min_{x_i} |w^T x_i + b|$$

**SVM Objective:** Maximize the margin subject to correct classification:
$$\max_{w,b} \frac{1}{\|w\|} \quad \text{subject to} \quad y_i(w^T x_i + b) \geq 1 \text{ for all } i$$

**Python Example:**
```python
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Generate linearly separable data
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + [2, 2],
    np.random.randn(50, 2) + [-2, -2]
])
y = np.array([1]*50 + [-1]*50)

# Train linear SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# The support vectors are the points closest to the decision boundary
support_vectors = svm.support_vectors_
print(f"Number of support vectors: {len(support_vectors)}")
print(f"Hyperplane normal vector w: {svm.coef_[0]}")
print(f"Intercept b: {svm.intercept_[0]}")

# Visualize decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', label='Class -1')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
            s=100, linewidth=1.5, facecolors='none', edgecolors='green', 
            label='Support Vectors')
plt.legend()
plt.title('Maximum Margin Classifier')
plt.show()
```

### 1.2 Support Vectors

**Definition:** Support vectors are the training points that **lie exactly on the margin boundary** (or violate it). They are the "critical" points that define the decision boundary.

**Key Properties:**
- If you move a support vector, the decision boundary changes
- Non-support vectors can be removed without affecting the solution
- In practice, often only a small fraction of training data are support vectors
- This makes SVMs memory-efficient for large datasets

**Mathematical Insight:** Points satisfying $y_i(w^T x_i + b) = 1$ are support vectors.

### 1.3 Simplified SVM Optimization

Due to the **scale invariance** of hyperplanes, we can simplify the optimization problem.

If $(w, b)$ defines a hyperplane, so does $(\tau w, \tau b)$ for any $\tau > 0$ (same decision boundary).

We exploit this by enforcing: $\min_i |w^T x_i + b| = 1$

**Simplified SVM Problem:**
$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \geq 1 \text{ for all } i$$

This is a **quadratic programming (QP)** problem:
- Quadratic objective: $\frac{1}{2}w^T w$
- Linear constraints: $y_i(w^T x_i + b) \geq 1$

**Python Example:**
```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=100, n_features=2, 
                          n_informative=2, n_redundant=0, 
                          random_state=42)
y = 2*y - 1  # Convert to -1, +1 labels

# SVM solves the QP problem automatically
svm = SVC(kernel='linear')
svm.fit(X, y)

# Access the learned parameters
print(f"Norm of w: {np.linalg.norm(svm.coef_[0])}")
print(f"Margin (2/||w||): {2 / np.linalg.norm(svm.coef_[0])}")
```

---

## Part 2: Handling Non-Linearly Separable Data

### 2.1 Slack Variables and Soft-Margin SVM

**Problem:** Real-world data is often not linearly separable. Points may be misclassified or lie on the wrong side of the hyperplane.

**Solution:** Introduce **slack variables** $\xi_i \geq 0$ to allow constraint violations.

**Soft-Margin SVM Problem:**
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i$$
$$\text{subject to: } y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Interpretation:**
- $\xi_i = 0$: Point $x_i$ is correctly classified with margin $\geq 1$
- $0 < \xi_i < 1$: Point is correctly classified but violates the margin
- $\xi_i > 1$: Point is misclassified
- $C$ is the **regularization parameter**: trades off margin size vs. training error
  - Large $C$: penalize misclassification heavily (may overfit)
  - Small $C$: allow more misclassification (more regularization)

**Python Example:**
```python
from sklearn.svm import SVC
import numpy as np

# Generate non-linearly separable data
np.random.seed(42)
X = np.vstack([
    np.random.randn(30, 2) + [1, 1],
    np.random.randn(30, 2) + [-1, -1],
    np.random.randn(5, 2) + [1, -1]  # Noise/outliers
])
y = np.array([1]*30 + [-1]*30 + [1]*5)

# Try different C values
for C_val in [0.1, 1.0, 10.0]:
    svm = SVC(kernel='linear', C=C_val)
    svm.fit(X, y)
    
    # Count slack violations
    margins = y * (svm.decision_function(X))
    violations = np.sum(margins < 1)
    
    print(f"C={C_val}: {len(svm.support_vectors_)} support vectors, "
          f"{violations} margin violations")

# Visualize the effect
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, C_val in enumerate([0.1, 1.0, 10.0]):
    svm = SVC(kernel='linear', C=C_val)
    svm.fit(X, y)
    
    ax = axes[idx]
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1')
    ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', label='Class -1')
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    ax.set_title(f'C = {C_val}')
    ax.legend()

plt.tight_layout()
plt.show()
```

### 2.2 Unconstrained Formulation (Hinge Loss)

**Key Insight:** We can rewrite the constrained soft-margin SVM as an unconstrained optimization using the **hinge loss**.

For any fixed $(w, b)$, the optimal slack variable is:
$$\xi_i = \max(0, 1 - y_i(w^T x_i + b))$$

This leads to:
$$\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \max(0, 1 - y_i(w^T x_i + b))$$

The term $\max(0, 1 - y_i(w^T x_i + b))$ is called the **hinge loss** $L_{\text{hinge}}$.

**Advantages:**
- No constraints—easier to optimize
- Can use standard gradient descent
- Generalizes to other loss functions

**Python Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize hinge loss vs. other loss functions
margin = np.linspace(-2, 3, 100)

# Hinge loss
hinge_loss = np.maximum(0, 1 - margin)

# Squared hinge loss
squared_hinge = np.maximum(0, 1 - margin)**2

# Logistic loss
logistic_loss = np.log(1 + np.exp(-margin))

# 0-1 loss (training error)
zero_one_loss = (margin < 0).astype(float)

plt.figure(figsize=(10, 6))
plt.plot(margin, hinge_loss, label='Hinge Loss', linewidth=2)
plt.plot(margin, squared_hinge, label='Squared Hinge Loss', linewidth=2)
plt.plot(margin, logistic_loss, label='Logistic Loss', linewidth=2)
plt.plot(margin, zero_one_loss, label='0-1 Loss (Training Error)', linewidth=2)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Margin: $y_i(w^T x_i + b)$')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Loss Functions for Classification')
plt.ylim([0, 3])
plt.show()

# Compute gradients of hinge loss
def hinge_loss_func(margin):
    return np.maximum(0, 1 - margin)

# Gradient (note: not differentiable at margin=1)
def hinge_gradient(margin):
    return np.where(margin < 1, -1, 0)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(margin, hinge_loss, linewidth=2)
ax1.set_title('Hinge Loss')
ax1.set_xlabel('Margin')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(margin[1:-1], hinge_gradient(margin[1:-1]), linewidth=2)
ax2.set_title('Hinge Loss Gradient')
ax2.set_xlabel('Margin')
ax2.set_ylabel('Gradient')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Part 3: Empirical Risk Minimization Framework

### 3.1 General ERM Framework

**Concept:** Most machine learning algorithms can be unified under **Empirical Risk Minimization (ERM)**.

$$\min_{w} \frac{1}{n}\sum_{i=1}^n L(y_i, w^T x_i + b) + \lambda R(w)$$

Where:
- $L(y_i, \hat{y}_i)$: **Loss function** measuring prediction error
- $R(w)$: **Regularizer** penalizing model complexity
- $\lambda$: Regularization strength (trades off fitting vs. simplicity)

**Key Insight:** SVMs, logistic regression, ridge regression, and LASSO all fit this framework with different choices of loss and regularizer.

### 3.2 Loss Functions

**For Classification:**

| Loss | Formula | When to Use | Properties |
|------|---------|------------|------------|
| **Hinge Loss** | $\max(0, 1-y_i\hat{y}_i)$ | Linear SVM | Piecewise linear, not differentiable at margin |
| **Squared Hinge** | $[\max(0, 1-y_i\hat{y}_i)]^2$ | Smooth SVM | Differentiable, more aggressive on misclassifications |
| **Logistic Loss** | $\log(1+e^{-y_i\hat{y}_i})$ | Logistic Regression | Smooth, probabilistic interpretation, linear asymptotically |
| **0-1 Loss** | $\mathbb{1}[y_i\hat{y}_i < 0]$ | Theory | Training error, non-differentiable, not used for optimization |

**For Regression:**

| Loss | Formula | When to Use | Properties |
|------|---------|------------|------------|
| **Squared Loss** | $(y_i - \hat{y}_i)^2$ | Linear/Ridge Regression | Smooth, sensitive to outliers |
| **Absolute Loss** | $\|y_i - \hat{y}_i\|$ | Robust Regression | Robust to outliers, not differentiable at 0 |
| **Huber Loss** | Hybrid of squared and absolute | Noisy data | Combines advantages: smooth + robust |

**Python Example:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVC, SVR

# Classification: Compare loss functions
np.random.seed(42)
X_class = np.random.randn(100, 2)
y_class = np.sign(X_class[:, 0] + X_class[:, 1] + 0.5 * np.random.randn(100))

# Train different classifiers
svm_hinge = SVC(kernel='linear', C=1.0)
svm_squared = SVC(kernel='linear', C=1.0)  # With squared=True in practice
log_reg = LogisticRegression()

svm_hinge.fit(X_class, y_class)
log_reg.fit(X_class, y_class)

# Regression: Compare loss functions
X_reg = np.linspace(-5, 5, 100).reshape(-1, 1)
y_reg = 2*X_reg[:, 0] + np.random.randn(100) * 2
y_reg[0:5] += 20  # Add outliers

ridge = Ridge(alpha=1.0)
linear = LinearRegression()
svr = SVR(kernel='linear', C=1.0, epsilon=0.1)

ridge.fit(X_reg, y_reg)
linear.fit(X_reg, y_reg)
svr.fit(X_reg, y_reg)

# Compare predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Classification
ax = axes[0]
x_min, x_max = X_class[:, 0].min() - 1, X_class[:, 0].max() + 1
y_min, y_max = X_class[:, 1].min() - 1, X_class[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z_svm = svm_hinge.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_log = log_reg.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax.contour(xx, yy, Z_svm, levels=[0], colors='red', linewidths=2, label='SVM')
ax.contour(xx, yy, Z_log, levels=[0], colors='blue', linewidths=2, label='Logistic')
ax.scatter(X_class[y_class==1, 0], X_class[y_class==1, 1], c='red', alpha=0.5)
ax.scatter(X_class[y_class==-1, 0], X_class[y_class==-1, 1], c='blue', alpha=0.5)
ax.set_title('Classification: Hinge vs. Logistic Loss')
ax.legend()

# Regression
ax = axes[1]
X_sorted = X_reg[np.argsort(X_reg[:, 0]).flatten()]
y_sorted = y_reg[np.argsort(X_reg[:, 0]).flatten()]

y_pred_ridge = ridge.predict(X_sorted)
y_pred_linear = linear.predict(X_sorted)
y_pred_svr = svr.predict(X_sorted)

ax.scatter(X_reg, y_reg, alpha=0.5, label='Data')
ax.plot(X_sorted, y_pred_ridge, 'r-', linewidth=2, label='Ridge (Squared Loss)')
ax.plot(X_sorted, y_pred_linear, 'b-', linewidth=2, label='Linear (Squared Loss)')
ax.plot(X_sorted, y_pred_svr, 'g-', linewidth=2, label='SVR (Epsilon Loss)')
ax.set_title('Regression: Different Loss Functions')
ax.legend()

plt.tight_layout()
plt.show()
```

### 3.3 Regularizers

**Purpose:** Prevent overfitting by penalizing model complexity.

| Regularizer | Formula | Name | Effect |
|---|---------|------|--------|
| **L2 Regularization** | $\frac{\lambda}{2}\|w\|_2^2$ | Ridge Regression / Max Margin | Shrinks all weights, prefers smaller norms |
| **L1 Regularization** | $\lambda\|w\|_1$ | LASSO / Sparse SVM | Induces sparsity, sets many weights to exactly 0 |
| **Elastic Net** | $\lambda_1\|w\|_1 + \lambda_2\|w\|^2$ | Combines L1 and L2 | Best of both worlds for feature selection |

**Python Example:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression

# Generate data with many features
X, y = make_regression(n_samples=100, n_features=50, n_informative=10, 
                      random_state=42)

# Train with different regularizers
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

ridge.fit(X, y)
lasso.fit(X, y)
elastic.fit(X, y)

# Visualize regularization effects
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Ridge (L2): All weights shrink but remain non-zero
ax = axes[0]
ax.stem(ridge.coef_)
ax.set_title('Ridge (L2): All weights shrink proportionally')
ax.set_ylabel('Coefficient Value')
ax.set_xlabel('Feature Index')

# LASSO (L1): Many weights become exactly zero
ax = axes[1]
ax.stem(lasso.coef_)
ax.set_title('LASSO (L1): Sparse solution')
ax.set_ylabel('Coefficient Value')
ax.set_xlabel('Feature Index')

# Elastic Net: Compromise
ax = axes[2]
ax.stem(elastic.coef_)
ax.set_title('Elastic Net: L1 + L2 combination')
ax.set_ylabel('Coefficient Value')
ax.set_xlabel('Feature Index')

plt.tight_layout()
plt.show()

# Compare sparsity
print(f"Ridge non-zero weights: {np.sum(ridge.coef_ != 0)} / 50")
print(f"LASSO non-zero weights: {np.sum(lasso.coef_ != 0)} / 50")
print(f"ElasticNet non-zero weights: {np.sum(elastic.coef_ != 0)} / 50")
```

### 3.4 Common Algorithm Combinations

| Algorithm | Loss | Regularizer | Framework |
|-----------|------|-------------|-----------|
| **Linear SVM** | Hinge | L2 | max-margin |
| **Ridge Regression** | Squared | L2 | — |
| **LASSO** | Squared | L1 | — |
| **Elastic Net** | Squared | L1 + L2 | — |
| **Logistic Regression** | Logistic | L1 or L2 | — |

**Python Example - Unified Framework:**
```python
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVC

# Same data
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.sign(X[:, 0] + X[:, 1])

# All can be viewed as ERM with different loss + regularizer choices
models = {
    'Ridge (Squared Loss + L2)': Ridge(alpha=0.1),
    'Logistic Regression (Logistic Loss + L2)': LogisticRegression(C=10),
    'SVM (Hinge Loss + L2)': SVC(kernel='linear', C=10)
}

for name, model in models.items():
    model.fit(X, y)
    accuracy = model.score(X, y)
    print(f"{name}: Accuracy = {accuracy:.3f}")
```

---

## Part 4: Kernel Trick and Non-Linear SVMs

### 4.1 Feature Expansion and the Curse of Dimensionality

**Problem:** Non-linearly separable data requires higher-dimensional feature representations.

**Example - Concentric Circles:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate concentric circles (not linearly separable)
np.random.seed(42)
n_samples = 100

# Inner circle
inner = np.random.randn(n_samples // 2, 2) * 0.5
# Outer circle
outer = np.random.randn(n_samples // 2, 2) * 1.5 + 2

X = np.vstack([inner, outer])
y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])

# Can't separate with linear classifier
from sklearn.svm import SVC

svm_linear = SVC(kernel='linear')
svm_linear.fit(X, y)
print(f"Linear SVM accuracy: {svm_linear.score(X, y):.3f}")

# Need non-linear mapping
# Map to 3D: (x1, x2) -> (x1, x2, x1^2 + x2^2)
# This makes the classes separable by a plane
X_expanded = np.hstack([X, (X**2).sum(axis=1).reshape(-1, 1)])

svm_3d = SVC(kernel='linear')
svm_3d.fit(X_expanded, y)
print(f"3D Linear SVM accuracy: {svm_3d.score(X_expanded, y):.3f}")

# Visualize
fig = plt.figure(figsize=(12, 4))

# Original 2D
ax = fig.add_subplot(121)
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1')
ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', label='Class -1')
ax.set_title('Original Data: Not Linearly Separable')
ax.legend()

# Expanded 3D (project onto 3D)
ax = fig.add_subplot(122, projection='3d')
ax.scatter(X_expanded[y==1, 0], X_expanded[y==1, 1], X_expanded[y==1, 2], 
          c='red', label='Class 1')
ax.scatter(X_expanded[y==-1, 0], X_expanded[y==-1, 1], X_expanded[y==-1, 2], 
          c='blue', label='Class -1')
ax.set_title('Expanded 3D: Now Linearly Separable')
ax.legend()

plt.tight_layout()
plt.show()
```

**Challenge:** Feature expansion explodes dimensionality ($d$ features → $2^d$ in extreme case).

### 4.2 The Kernel Trick

**Key Insight:** We only need **inner products** between data points, not the actual high-dimensional coordinates.

**Example - Polynomial Features:**

Original features: $(x_1, x_2)$  
Expanded features: $(1, x_1, x_2, x_1x_2, x_1^2, x_2^2)$ (degree 2)

Rather than computing $\phi(x_i)^T \phi(x_j)$ explicitly, we use a **kernel function**:
$$K(x_i, x_j) = (1 + x_i^T x_j)^2$$

This computes the same inner product in the expanded space but requires only $O(d)$ operations instead of $O(d^2)$.

**General Dual SVM with Kernels:**
$$\min_{\alpha} \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i, x_j) - \sum_i \alpha_i$$
$$\text{subject to: } 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

**Prediction:**
$$h(x) = \text{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)$$

**Python Example:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles

# Generate non-linearly separable data
np.random.seed(42)
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3)
X_moons, y_moons = make_moons(n_samples=200, noise=0.1)

y_circles = 2*y_circles - 1  # Convert to -1, 1
y_moons = 2*y_moons - 1

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

datasets = [
    (X_circles, y_circles, 'Circles'),
    (X_moons, y_moons, 'Moons')
]

for row, (X, y, title) in enumerate(datasets):
    # Linear kernel
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X, y)
    
    # Polynomial kernel
    svm_poly = SVC(kernel='poly', degree=3)
    svm_poly.fit(X, y)
    
    # RBF kernel
    svm_rbf = SVC(kernel='rbf', gamma='scale')
    svm_rbf.fit(X, y)
    
    svms = [svm_linear, svm_poly, svm_rbf]
    names = ['Linear', 'Polynomial (d=3)', 'RBF']
    
    for col, (svm, name) in enumerate(zip(svms, names)):
        ax = axes[row, col]
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6)
        ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', alpha=0.6)
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                  s=100, linewidths=1.5, facecolors='none', edgecolors='green')
        
        ax.set_title(f'{title}: {name}\nAccuracy: {svm.score(X, y):.3f}')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
```

### 4.3 Common Kernels

| Kernel | Formula | Use Case | Parameters |
|--------|---------|----------|------------|
| **Linear** | $K(x, z) = x^T z$ | Already separable, interpretability needed | — |
| **Polynomial** | $K(x, z) = (1 + x^T z)^p$ | Moderate complexity | $p$ (degree) |
| **RBF (Radial Basis Function)** | $K(x, z) = \exp(-\gamma\|x-z\|^2)$ | General purpose, complex boundaries | $\gamma$ (gamma) |
| **Sigmoid** | $K(x, z) = \tanh(\kappa x^T z + \theta)$ | Neural network-like | $\kappa, \theta$ |

**Mathematical Properties of Valid Kernels:**

A kernel $K(x, z)$ is valid if the corresponding Gram matrix $K_{ij} = K(x_i, x_j)$ is **positive semidefinite**.

**Python Example - Kernel Selection:**
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

# Generate classification dataset
X, y = make_classification(n_samples=200, n_features=20, 
                          n_informative=15, n_redundant=5, random_state=42)
y = 2*y - 1

# Grid search over kernel types and parameters
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],  # Only for RBF and poly
}

# Note: Grid search will handle parameter compatibility
svm = SVC()
grid = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
grid.fit(X, y)

print(f"Best kernel: {grid.best_params_['kernel']}")
print(f"Best C: {grid.best_params_['C']}")
print(f"Best score: {grid.best_score_:.3f}")
```

### 4.4 Kernelizing Any ERM Algorithm

**Beautiful Result:** Not just SVMs, but ANY ERM algorithm can be kernelized!

**Key Observation:** In gradient descent with ERM:
$$w = \sum_{i=1}^n \alpha_i x_i$$

For any loss and regularizer where this holds, we can replace:
- Inner products $x^T z$ with kernel $K(x, z)$
- We only need to track coefficients $\alpha_i$, never explicitly compute $w$

**Examples:**
- Kernel Ridge Regression
- Kernel Logistic Regression
- Kernel SVM (most common)

**Python Example - Kernelized Ridge Regression:**
```python
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generate non-linear regression dataset
np.random.seed(42)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.randn(100) * 0.2

# Train with different kernels
kr_linear = KernelRidge(kernel='linear', alpha=0.01)
kr_poly = KernelRidge(kernel='poly', degree=3, alpha=0.01)
kr_rbf = KernelRidge(kernel='rbf', gamma=1, alpha=0.01)

kr_linear.fit(X, y)
kr_poly.fit(X, y)
kr_rbf.fit(X, y)

# Make predictions
X_test = np.linspace(-5, 5, 300).reshape(-1, 1)
y_pred_linear = kr_linear.predict(X_test)
y_pred_poly = kr_poly.predict(X_test)
y_pred_rbf = kr_rbf.predict(X_test)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (kr, name, y_pred) in zip(axes, [
    (kr_linear, 'Linear Kernel', y_pred_linear),
    (kr_poly, 'Polynomial Kernel', y_pred_poly),
    (kr_rbf, 'RBF Kernel', y_pred_rbf)
]):
    ax.scatter(X, y, alpha=0.5, label='Data')
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label='Prediction')
    ax.set_title(f'{name}\nScore: {kr.score(X, y):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Part 5: Solving the Dual Problem

### 5.1 Primal vs. Dual Formulation

**Primal Problem** (what we write):
$$\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_i \max(0, 1 - y_i(w^T x_i + b))$$

**Dual Problem** (what solvers actually solve):
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$
$$\text{subject to: } 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

**Why Solve the Dual?**
- Only depends on inner products $x_i^T x_j$ → enables kernel trick
- Often more efficient for small number of support vectors
- Provides useful insights (support vector coefficients $\alpha_i$)

**Relationship to Primal:**
- Once we solve for $\alpha$, we recover: $w = \sum_i \alpha_i y_i x_i$
- $b$ is recovered from support vectors: $b = y_i - w^T x_i$ for any support vector

**Python Example:**
```python
import numpy as np
from sklearn.svm import SVC

# Generate data
np.random.seed(42)
X = np.random.randn(50, 2)
y = np.sign(X[:, 0] + X[:, 1])

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Access dual solution
print(f"Support vector coefficients (alpha):\n{svm.dual_coef_}")
print(f"\nRecovered w = sum(alpha_i * y_i * x_i):\n{svm.coef_}")
print(f"Intercept b:\n{svm.intercept_}")

# Verify manually
w_manual = svm.dual_coef_[0].dot(X[svm.support_indices_])
print(f"\nManually computed w:\n{w_manual}")
print(f"Matches svm.coef_[0]: {np.allclose(w_manual, svm.coef_[0])}")
```

---

## Part 6: High-Dimensional Feature Expansion

### 6.1 Feature Expansion Strategies

For $d$-dimensional input, polynomial features of degree $p$:
- Linear features: $d$ features
- Quadratic features: $O(d^2)$ features
- General degree $p$: $O(d^p)$ features

**Example:**
- $d = 100$ features
- Degree 2: $\approx 5,000$ features
- Degree 3: $\approx 166,000$ features

**This is why the kernel trick is essential!**

**Python Example:**
```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Generate 3 features
X = np.array([[1, 2, 3], [4, 5, 6]])

# Polynomial features of different degrees
for degree in [1, 2, 3]:
    pf = PolynomialFeatures(degree=degree)
    X_poly = pf.fit_transform(X)
    print(f"Degree {degree}: {X.shape[1]} → {X_poly.shape[1]} features")
    print(f"Feature names: {pf.get_feature_names_out()}\n")
```

---

## Part 7: Practical Implementation

### 7.1 Complete SVM Example

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM with RBF kernel
svm = SVC(kernel='rbf', gamma='scale', C=1.0)
svm.fit(X_train_scaled, y_train)

# Evaluate
y_pred = svm.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize decision boundaries (for 2D subset)
X_2d = X_train_scaled[:, :2]
X_2d_test = X_test_scaled[:, :2]

svm_2d = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_2d.fit(X_2d, y_train)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.8)
plt.scatter(X_2d[y_train==0, 0], X_2d[y_train==0, 1], c='red', label='Setosa')
plt.scatter(X_2d[y_train==1, 0], X_2d[y_train==1, 1], c='green', label='Versicolor')
plt.scatter(X_2d[y_train==2, 0], X_2d[y_train==2, 1], c='blue', label='Virginica')
plt.scatter(svm_2d.support_vectors_[:, 0], svm_2d.support_vectors_[:, 1],
           s=100, linewidths=1.5, facecolors='none', edgecolors='black',
           label='Support Vectors')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title(f'Iris SVM (2D projection)\nSupport vectors: {len(svm_2d.support_vectors_)}')
plt.legend()
plt.show()
```

### 7.2 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=300, n_features=20, 
                          n_informative=15, random_state=42)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
}

# Grid search with cross-validation
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Access best model
best_svm = grid_search.best_estimator_
print(f"Training accuracy: {best_svm.score(X, y):.3f}")
```

---

## Key Takeaways

1. **Maximum Margin Principle:** SVMs find the hyperplane with largest margin for better generalization
2. **Soft Margins:** Slack variables allow some misclassification for robustness
3. **Empirical Risk Minimization:** Unified framework with loss + regularizer choices
4. **Kernel Trick:** Transform non-linearly separable problems without explicit feature expansion
5. **Dual Problem:** Enables kernel methods and is often more efficient to solve
6. **Hyperparameter Tuning:** C, kernel type, and kernel parameters must be optimized for your data

---

## Summary Table: SVM Concepts

| Concept | Definition | Key Equation | When Used |
|---------|-----------|--------------|-----------|
| Margin | Distance from hyperplane to nearest point | $\gamma = \frac{1}{\|w\|}$ | Maximum margin principle |
| Hinge Loss | Penalty for margin violation | $\max(0, 1-y_i\hat{y}_i)$ | Soft-margin SVM |
| Slack Variable | Allows constraint violation | $y_i(w^Tx_i + b) \geq 1-\xi_i$ | Non-linearly separable data |
| Support Vector | Point on or violating margin | $y_i(w^Tx_i + b) = 1 \pm \xi_i$ | Define decision boundary |
| Kernel Function | Maps to high-dimensional space implicitly | $K(x,z) = \phi(x)^T\phi(z)$ | Non-linear classification |
| RBF Kernel | Radial basis function kernel | $K(x,z) = \exp(-\gamma\|x-z\|^2)$ | General-purpose non-linear |

---

# Extended Derivations & Python Lab

The sections above cover concepts and applied recipes. This appendix adds (1) the full derivations from the PDF that are worth doing on paper at least once, and (2) runnable NumPy/scikit code you can drop into a REPL to *see* every idea work.

## A. Extended Derivations from the PDF

### A.1 Distance from a point to a hyperplane (full derivation)

Let the hyperplane be $H = \{ \mathbf{x} : \mathbf{w}^\top \mathbf{x} + b = 0 \}$. Given a query point $\mathbf{x}$, let $\mathbf{x}_p$ be its orthogonal projection onto $H$ and let $\mathbf{d} = \mathbf{x} - \mathbf{x}_p$.

1. Because $\mathbf{x}_p \in H$: $\mathbf{w}^\top \mathbf{x}_p + b = 0$.
2. Because $\mathbf{d} \perp H$ and $\mathbf{w} \perp H$: $\mathbf{d} = \alpha \mathbf{w}$ for some scalar $\alpha$.
3. Substitute $\mathbf{x}_p = \mathbf{x} - \alpha \mathbf{w}$: $\mathbf{w}^\top(\mathbf{x} - \alpha \mathbf{w}) + b = 0 \Rightarrow \alpha = \dfrac{\mathbf{w}^\top \mathbf{x} + b}{\mathbf{w}^\top \mathbf{w}}$.
4. Therefore

$$\text{dist}(\mathbf{x}, H) \;=\; \|\mathbf{d}\|_2 \;=\; |\alpha|\,\|\mathbf{w}\|_2 \;=\; \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|_2}.$$

Taking the minimum over the training set gives the margin $\gamma(\mathbf{w}, b) = \tfrac{1}{\|\mathbf{w}\|_2} \min_i |\mathbf{w}^\top \mathbf{x}_i + b|$.

### A.2 Why the complex and simple SVM formulations are equivalent

Complex form:

$$\min_{\mathbf{w}, b} \mathbf{w}^\top \mathbf{w} \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 0 \ \forall i, \quad \min_i |\mathbf{w}^\top \mathbf{x}_i + b| = 1.$$

Simple form:

$$\min_{\mathbf{w}, b} \mathbf{w}^\top \mathbf{w} \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \ \forall i.$$

**Complex $\Rightarrow$ simple.** From $\min_i |\mathbf{w}^\top \mathbf{x}_i + b| = 1$ we get $|\mathbf{w}^\top \mathbf{x}_i + b| \geq 1$ for all $i$. Multiplying by $y_i \in \{-1,+1\}$ (whose sign matches the prediction by the first constraint) gives $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$.

**Simple $\Rightarrow$ complex.** $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 > 0$ satisfies the first complex constraint. It also implies $|\mathbf{w}^\top \mathbf{x}_i + b| \geq 1$. Suppose the minimum were strictly $> 1$; then we could rescale $(\mathbf{w}, b) \to (\mathbf{w}/\tau, b/\tau)$ with $\tau > 1$, still satisfy all constraints, and *decrease* $\mathbf{w}^\top \mathbf{w}$ — contradicting optimality. Hence at the optimum $\min_i |\mathbf{w}^\top \mathbf{x}_i + b| = 1$. The training points achieving this equality are exactly the **support vectors**.

### A.3 Slack variables collapse into hinge loss

From the soft-margin constraints $\xi_i \geq \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$, and because the objective penalizes $\xi_i$ with $C > 0$, the optimizer will always pick the smallest feasible slack:

$$\xi_i^\star = \max\bigl(0,\; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\bigr).$$

Substituting back gives the **unconstrained** soft-margin objective:

$$\min_{\mathbf{w}, b}\ \underbrace{\mathbf{w}^\top \mathbf{w}}_{\text{L2 regularizer}} + C \sum_{i=1}^n \underbrace{\max\bigl(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\bigr)}_{\text{hinge loss}}.$$

This is the SVM cast as **empirical risk minimization**: hinge loss + L2 regularizer.

### A.4 Gradient of the (squared-)hinge SVM loss

With $\delta_i = \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$,

$$\mathcal{L}(\mathbf{w}, b) = \mathbf{w}^\top \mathbf{w} + C \sum_{i=1}^n \delta_i^2$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2 \mathbf{w} - 2C \sum_{i=1}^n \delta_i \, y_i \mathbf{x}_i, \qquad \frac{\partial \mathcal{L}}{\partial b} = -2C \sum_{i=1}^n \delta_i \, y_i.$$

For plain (un-squared) hinge, replace $2\delta_i$ with the subgradient $\mathbf{1}[\delta_i > 0]$.

### A.5 The kernel identity $\phi(\mathbf{x})^\top \phi(\mathbf{z}) = \prod_{k=1}^d (1 + x_k z_k)$

For the PDF's "all subsets" feature map $\phi: \mathbb{R}^d \to \mathbb{R}^{2^d}$ whose coordinates enumerate every product $\prod_{k \in S} x_k$ over $S \subseteq \{1, \dots, d\}$:

$$\phi(\mathbf{x})^\top \phi(\mathbf{z}) = \sum_{S \subseteq [d]} \prod_{k \in S} x_k z_k = \prod_{k=1}^{d} (1 + x_k z_k),$$

by distributivity — every term in the expanded product picks either $1$ or $x_k z_k$ from each factor, generating exactly one summand per subset $S$. This turns a $2^d$-dimensional inner product into $d$ multiplications: **the kernel trick in one line**.

### A.6 The RBF is positive semidefinite (compositional proof)

Following the "8 rules" from the PDF:

1. $K_1(\mathbf{x}, \mathbf{z}) = \tfrac{2 \mathbf{x}^\top \mathbf{z}}{\sigma^2}$ — valid (Rule 2, scalar multiple of linear).
2. $K_2 = e^{K_1}$ — valid (Rule 7, exponential of a kernel).
3. $K_3(\mathbf{x}, \mathbf{z}) = f(\mathbf{x}) K_2(\mathbf{x}, \mathbf{z}) f(\mathbf{z})$ with $f(\mathbf{x}) = e^{-\|\mathbf{x}\|^2/\sigma^2}$ — valid (Rule 6).
4. Expanding: $K_3(\mathbf{x}, \mathbf{z}) = \exp\!\left(-\tfrac{\|\mathbf{x}\|^2 - 2\mathbf{x}^\top \mathbf{z} + \|\mathbf{z}\|^2}{\sigma^2}\right) = \exp\!\left(-\tfrac{\|\mathbf{x} - \mathbf{z}\|^2}{\sigma^2}\right)$. ✓

### A.7 Representer theorem sketch (why *every* ERM kernelizes)

For any loss $L(\mathbf{w}^\top \mathbf{x}, y)$ that depends on $\mathbf{w}$ only through $\mathbf{w}^\top \mathbf{x}$, the gradient with respect to $\mathbf{w}$ is

$$\nabla_{\mathbf{w}} L = L'(\mathbf{w}^\top \mathbf{x}_i, y_i)\, \mathbf{x}_i,$$

a *scalar times a data point*. Starting from $\mathbf{w}_0 = \mathbf{0}$, each gradient step keeps $\mathbf{w}$ inside $\text{span}(\mathbf{x}_1, \dots, \mathbf{x}_n)$. Hence at every iterate

$$\mathbf{w} = \sum_{i=1}^n \alpha_i \mathbf{x}_i \quad \Rightarrow \quad \mathbf{w}^\top \mathbf{x} = \sum_i \alpha_i \mathbf{x}_i^\top \mathbf{x} \;\longrightarrow\; \sum_i \alpha_i K(\mathbf{x}_i, \mathbf{x}).$$

This is why **kernel ridge regression, kernel logistic regression, kernel SVM** — every ERM — can all be trained without ever materializing $\mathbf{w}$.

---

## B. Python Lab

Everything below is self-contained NumPy / scikit-learn. Copy any block into a notebook.

### B.1 From-scratch soft-margin SVM via gradient descent on squared hinge

Trains the unconstrained ERM objective from §A.3–A.4 directly with vanilla gradient descent.

```python
import numpy as np

class LinearSVMGD:
    """Soft-margin linear SVM trained by gradient descent on squared hinge + L2."""
    def __init__(self, C=1.0, lr=1e-3, n_iter=2000):
        self.C, self.lr, self.n_iter = C, lr, n_iter

    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d); b = 0.0
        for _ in range(self.n_iter):
            margin = y * (X @ w + b)          # y_i (w.x_i + b)
            delta  = np.maximum(0.0, 1.0 - margin)
            grad_w = 2*w - 2*self.C * (delta * y) @ X
            grad_b =      -2*self.C * (delta * y).sum()
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.w_, self.b_ = w, b
        return self

    def decision_function(self, X): return X @ self.w_ + self.b_
    def predict(self, X):           return np.sign(self.decision_function(X))

# Sanity check vs sklearn
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

X, y = make_classification(n_samples=300, n_features=5, random_state=0)
y = 2*y - 1

mine  = LinearSVMGD(C=1.0, lr=1e-3, n_iter=3000).fit(X, y)
skl   = LinearSVC(C=1.0, loss='squared_hinge', max_iter=5000).fit(X, y)

print(f"my train acc  : {(mine.predict(X) == y).mean():.3f}")
print(f"sklearn train : {skl.score(X, y):.3f}")
print(f"cosine(w, sklearn.w) = "
      f"{mine.w_ @ skl.coef_[0] / (np.linalg.norm(mine.w_)*np.linalg.norm(skl.coef_[0])):.3f}")
```

### B.2 Plot every classification loss on one axis (reproduces the Loss Function Cheat Sheet)

```python
import numpy as np, matplotlib.pyplot as plt

m = np.linspace(-3, 3, 400)                      # margin = y * (w.x + b)
losses = {
    '0-1':           (m < 0).astype(float),
    'hinge':         np.maximum(0, 1 - m),
    'squared hinge': np.maximum(0, 1 - m)**2,
    'logistic':      np.log1p(np.exp(-m)),
    'exponential':   np.exp(-m),
}
for name, L in losses.items():
    plt.plot(m, L, label=name)
plt.axvline(1, ls='--', c='gray'); plt.axvline(0, ls=':', c='gray')
plt.xlabel(r'margin $y_i(\mathbf{w}^\top \mathbf{x}_i + b)$'); plt.ylabel('loss')
plt.ylim(0, 4); plt.legend(); plt.title('Classification losses')
```

Key takeaways to point at while looking at the plot:

- **Hinge** — kink at margin $= 1$; grows linearly on the wrong side. Robust to mislabeled points.
- **Squared hinge** — smooth but grows *quadratically* on the wrong side; more aggressive.
- **Logistic** — smooth everywhere; linear on the wrong side; gives calibrated probabilities.
- **Exponential** — extreme growth on the wrong side (AdaBoost's engine); very sensitive to noise.

### B.3 The $C$ knob — bias/variance of soft-margin SVM

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=80, centers=2, cluster_std=1.8, random_state=1)
y = 2*y - 1

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, C in zip(axes, [0.01, 0.1, 1.0, 100.0]):
    clf = SVC(kernel='linear', C=C).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--','-','--'], colors='k')
    ax.scatter(*X.T, c=y, cmap='bwr', edgecolors='k')
    ax.scatter(*clf.support_vectors_.T, s=140, facecolors='none', edgecolors='lime', lw=2)
    ax.set_title(f'C = {C}  |  {len(clf.support_vectors_)} SVs')
```

Small $C$ → wide margin, many SVs, high bias. Large $C$ → narrow margin, tries to nail every point, high variance.

### B.4 Sanity-checking the kernel identity from §A.5

```python
import numpy as np
from itertools import combinations

def all_subsets_phi(x):
    d = len(x)
    feats = [1.0]
    for k in range(1, d+1):
        for S in combinations(range(d), k):
            feats.append(np.prod(x[list(S)]))
    return np.array(feats)

def K_all_subsets(x, z):
    return np.prod(1.0 + x*z)

rng = np.random.default_rng(0)
for d in [1, 2, 3, 6]:
    x, z = rng.normal(size=d), rng.normal(size=d)
    lhs = all_subsets_phi(x) @ all_subsets_phi(z)   # 2^d dim inner product
    rhs = K_all_subsets(x, z)                       # d-term product
    print(f"d={d}  explicit={lhs:.6f}  kernel={rhs:.6f}  diff={abs(lhs-rhs):.2e}")
```

At $d=20$ the explicit $\phi$ has $2^{20} \approx 10^6$ entries. At $d=1000$ it has more coordinates than atoms in the visible universe — but the kernel is still $1000$ multiplications.

### B.5 Kernel SVM zoo — visualize linear vs polynomial vs RBF

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles

datasets = [('moons',   make_moons(n_samples=200, noise=0.2, random_state=0)),
            ('circles', make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=0))]
kernels  = [dict(kernel='linear'),
            dict(kernel='poly', degree=3, coef0=1),
            dict(kernel='rbf', gamma=1.0)]

fig, axes = plt.subplots(len(datasets), len(kernels), figsize=(13, 8))
for i, (name, (X, y)) in enumerate(datasets):
    y = 2*y - 1
    for j, kws in enumerate(kernels):
        clf = SVC(C=10, **kws).fit(X, y)
        xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 200),
                             np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 200))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax = axes[i, j]
        ax.contourf(xx, yy, Z, levels=20, cmap='bwr', alpha=0.3)
        ax.contour (xx, yy, Z, levels=[0], colors='k')
        ax.scatter(*X.T, c=y, cmap='bwr', edgecolors='k', s=25)
        ax.set_title(f"{name} / {kws['kernel']}  acc={clf.score(X,y):.2f}")
```

### B.6 Building an RBF kernel SVM from scratch (kernel matrix + QP dual)

The full dual QP:

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j), \quad 0 \le \alpha_i \le C, \ \sum_i \alpha_i y_i = 0.$$

Solved here with `cvxopt` (equivalent to sklearn under the hood):

```python
# pip install cvxopt
import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def rbf_gram(X, Z, gamma):
    sq = (X**2).sum(1)[:, None] + (Z**2).sum(1)[None, :] - 2 * X @ Z.T
    return np.exp(-gamma * sq)

class KernelSVM:
    def __init__(self, C=1.0, gamma=1.0):
        self.C, self.gamma = C, gamma

    def fit(self, X, y):
        n = len(y)
        K = rbf_gram(X, X, self.gamma)
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), self.C * np.ones(n)]))
        A = matrix(y.astype(float), (1, n))
        b = matrix(0.0)
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x']).ravel()

        sv = alpha > 1e-5
        self.alpha_, self.sv_X_, self.sv_y_ = alpha[sv], X[sv], y[sv]
        # bias from any margin SV (0 < alpha < C)
        margin_sv = (alpha > 1e-5) & (alpha < self.C - 1e-5)
        k = rbf_gram(X[margin_sv], self.sv_X_, self.gamma)
        self.b_ = np.mean(y[margin_sv] - (self.alpha_ * self.sv_y_) @ k.T)
        return self

    def decision_function(self, X):
        k = rbf_gram(X, self.sv_X_, self.gamma)
        return k @ (self.alpha_ * self.sv_y_) + self.b_

    def predict(self, X): return np.sign(self.decision_function(X))

# Verify against sklearn on the circles dataset
from sklearn.datasets import make_circles
from sklearn.svm import SVC
X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=0)
y = 2*y - 1

mine = KernelSVM(C=1.0, gamma=1.0).fit(X, y)
skl  = SVC(C=1.0, gamma=1.0, kernel='rbf').fit(X, y)
print(f"scratch acc = {(mine.predict(X) == y).mean():.3f}")
print(f"sklearn acc = {skl.score(X, y):.3f}")
```

### B.7 Platt scaling — turn SVM scores into probabilities

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=0)
y = 2*y - 1
X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.4, random_state=0)

svm = SVC(kernel='rbf', C=1.0).fit(X_tr, y_tr)
scores = svm.decision_function(X_cal).reshape(-1, 1)
platt  = LogisticRegression().fit(scores, y_cal)

def predict_proba(x):
    s = svm.decision_function(x).reshape(-1, 1)
    return platt.predict_proba(s)[:, 1]

print("first 5 calibrated P(y=+1) =", predict_proba(X_cal[:5]).round(3))
```

### B.8 One-vs-all and one-vs-one multi-class from scratch

```python
import numpy as np
from itertools import combinations
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)   # 3 classes
classes = np.unique(y)

# --- One-vs-All ---
ova = {c: SVC(kernel='linear', probability=True).fit(X, (y == c).astype(int))
       for c in classes}
def ova_predict(X):
    scores = np.stack([ova[c].decision_function(X) for c in classes], axis=1)
    return classes[np.argmax(scores, axis=1)]

# --- One-vs-One ---
ovo = {(a, b): SVC(kernel='linear').fit(X[(y==a)|(y==b)],
                                        (y[(y==a)|(y==b)] == a).astype(int))
       for a, b in combinations(classes, 2)}
def ovo_predict(X):
    votes = np.zeros((len(X), len(classes)), dtype=int)
    for (a, b), clf in ovo.items():
        pred = clf.predict(X)
        votes[pred == 1, list(classes).index(a)] += 1
        votes[pred == 0, list(classes).index(b)] += 1
    return classes[np.argmax(votes, axis=1)]

print(f"OvA train acc: {(ova_predict(X) == y).mean():.3f}")
print(f"OvO train acc: {(ovo_predict(X) == y).mean():.3f}")
```

### B.9 The RBF $\gamma$ knob — visualize influence radius

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.25, random_state=0)
y = 2*y - 1

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, gamma in zip(axes, [0.1, 1.0, 10.0, 100.0]):
    clf = SVC(kernel='rbf', C=10, gamma=gamma).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap='bwr', alpha=0.3)
    ax.contour (xx, yy, Z, levels=[0], colors='k')
    ax.scatter(*X.T, c=y, cmap='bwr', edgecolors='k', s=20)
    ax.set_title(f'γ = {gamma}')
```

- Tiny $\gamma$ → each training point influences the whole plane → almost-linear boundary.
- Huge $\gamma$ → each point only influences its immediate neighborhood → boundary shatters into little bubbles (classic overfitting).

### B.10 ERM zoo — same objective, different (loss, regularizer) picks

```python
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGDClassifier)
from sklearn.svm import LinearSVC

erm = {
    'OLS         (squared,  none)': LinearRegression(),
    'Ridge       (squared,  L2)':   Ridge(alpha=1.0),
    'Lasso       (squared,  L1)':   Lasso(alpha=0.1),
    'ElasticNet  (squared,  L1+L2)':ElasticNet(alpha=0.1, l1_ratio=0.5),
    'LogReg L2   (logistic, L2)':   LogisticRegression(penalty='l2', C=1.0),
    'LogReg L1   (logistic, L1)':   LogisticRegression(penalty='l1',  C=1.0, solver='liblinear'),
    'LinearSVC   (hinge,    L2)':   LinearSVC(C=1.0),
    'SGD-hinge   (hinge,    L1)':   SGDClassifier(loss='hinge', penalty='l1'),
}
for name in erm: print(name)
```

Every row above is the same optimization problem $\min \tfrac{1}{n}\sum L + \lambda R$ with different plug-ins for $L$ and $R$.

---

## C. Common pitfalls (memorize these)

1. **Always standardize features before an RBF SVM.** RBF distances are dominated by the largest-scale feature. `StandardScaler().fit_transform(X)` is not optional.
2. **`C` and `gamma` are entangled — always grid-search them together**, ideally on a log scale (`{10^-2, ..., 10^2}` × `{10^-3, ..., 10^1}`), never one at a time.
3. **Never trust `.decision_function` as a probability.** Use `CalibratedClassifierCV` or Platt scaling (§B.7) to get calibrated $P(y \mid \mathbf{x})$.
4. **Squared hinge amplifies outliers.** If your data has label noise, prefer plain hinge or logistic loss — squared hinge will chase the mislabeled point.
5. **Kernel SVM is $O(n^2)$ memory and roughly $O(n^2)$–$O(n^3)$ compute.** Above ~50k rows, switch to a linear SVM on explicit random-feature maps (`sklearn.kernel_approximation.RBFSampler`) or to a tree ensemble.
6. **Dual $\alpha_i$ tells you which points are support vectors:** $\alpha_i = 0$ ⇒ irrelevant, $0 < \alpha_i < C$ ⇒ on the margin, $\alpha_i = C$ ⇒ violates the margin. Great for debugging.
7. **L1 vs L2 regularizer:** L2 shrinks *all* weights smoothly (max-margin flavor); L1 zeros many weights (sparse, interpretable — genes, tokens, etc.). Elastic-net when you want both sparsity *and* uniqueness.
8. **Cross-reference:** hinge + L2 = SVM. Logistic + L2 = logistic regression. Squared + L2 = ridge. Squared + L1 = lasso. Same skeleton, different bones.

Now that you have finished this module, you can find the maximum margin classifier even for not linearly separable data sets using kernel support vector machines. You were introduced to the kernel trick and different kernels. The kernel trick is extremely useful, as it extends your existing tool chest of linear classifiers from previous modules to a whole set of extremely powerful non-linear classifiers. You learned how to train ERM algorithms in infinite dimensional feature spaces using the kernel trick. The final part of the project allowed you to practice applying these techniques. You were finally able to complete your project, by writing functions to compute the loss and gradient.