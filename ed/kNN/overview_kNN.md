# CIS531: Problem-Solving With Machine Learning
**Cornell University — Prof. Kilian Weinberger**

---

## Module 1: Framing a Supervised ML Problem

### 1.1 Core Setup

Supervised learning learns a function $h: \mathbb{R}^d \to \mathcal{C}$

- **Features** $\mathbf{x} \in \mathbb{R}^d$: known input (d-dimensional vector)
- **Label** $y \in \mathcal{C}$: unknown output to predict
- **Training data**: $D = \{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\} \subseteq \mathbb{R}^d \times \mathcal{C}$ — $n$ labeled examples
- **Goal**: learn $h$ so it generalizes to unseen inputs

**Formal notation (Cornell CIS531 conventions)**

| Symbol | Meaning |
|--------|---------|
| $\mathbb{R}^d$ | $d$-dimensional feature space |
| $\mathbf{x}_i \in \mathbb{R}^d$ | feature vector of sample $i$, for $i \in \{1,\ldots,n\}$ |
| $[\mathbf{x}_i]_\alpha$ | the $\alpha$-th feature dimension of $\mathbf{x}_i$, for $\alpha \in \{1,\ldots,d\}$ |
| $\mathcal{C}$ | the label space |
| $y_i \in \mathcal{C}$ | the label of sample $i$ |
| $h : \mathbb{R}^d \to \mathcal{C}$ | the learned hypothesis / prediction function |
| $\mathcal{H}$ | the hypothesis class — the set of all $h$ the algorithm could learn |

During training we search $\mathcal{H}$ for a specific $h$ such that $h(\mathbf{x}_i) \approx y_i$ for all $(\mathbf{x}_i, y_i) \in D$, in the hope it also holds for $(\mathbf{x}, y) \notin D$.

### 1.2 Label Types

| Type | $\mathcal{C}$ | Example |
|------|----------------|---------|
| Binary classification | $\{-1, +1\}$ | Spam vs. not spam |
| Multiclass classification | $\{1, 2, \ldots, K\}$ | Digit recognition (0–9) |
| Regression | $\mathbb{R}$ | Predict house price |

### 1.3 Loss Functions

Measure how wrong a prediction $h(\mathbf{x}_i)$ is compared to true label $y_i$:

| Loss | Formula | Used For |
|------|---------|----------|
| Zero-one | $\ell_{0/1} = \mathbf{1}[h(\mathbf{x}_i) \neq y_i]$ | Classification |
| Squared | $\ell_{sq} = (h(\mathbf{x}_i) - y_i)^2$ | Regression |
| Absolute | $\ell_{abs} = \vert h(\mathbf{x}_i) - y_i \vert$ | Regression (robust) |

**Training loss** (empirical risk):
$$\mathcal{L}(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(\mathbf{x}_i), y_i)$$

> Convention: lowercase $\ell$ is the loss on a **single** example; uppercase $\mathcal{L}$ (or $e_{TR}(h)$) is the **average** loss over a data set.

**Why the choice of loss matters (regression):**
- Squared loss $(h(\mathbf{x}_i) - y_i)^2$ grows *quadratically* — one big miss dominates the total. Great when labels are clean; dangerous when they are noisy.
- Absolute loss $|h(\mathbf{x}_i) - y_i|$ grows *linearly* — outliers cannot swing the objective as hard, so it is more robust to label noise.
- Zero–one loss is non-differentiable and non-continuous, so it is used to *evaluate* classifiers but rarely to *optimize* them directly.

**Python: compute each loss on a small example**

```python
import numpy as np

y_true = np.array([120_000, 95_000, 210_000])   # actual house prices
y_pred = np.array([130_000, 90_000, 150_000])   # model predictions

abs_loss = np.mean(np.abs(y_pred - y_true))     #   25,000
sq_loss  = np.mean((y_pred - y_true) ** 2)      # 1.2e9 — dominated by the $60k miss

# Zero–one loss (classification) — fraction misclassified
y_cls_true = np.array([1, -1,  1,  1, -1])
y_cls_pred = np.array([1,  1,  1, -1, -1])
zero_one   = np.mean(y_cls_pred != y_cls_true)  # 0.4  → 40% error
```

Notice how the single 60k miss inflates squared loss by orders of magnitude — the classic reason to prefer absolute loss on noisy data.

### 1.4 Data Splitting

Never evaluate on training data — it leads to overfitting.

| Split | Fraction | Purpose |
|-------|----------|---------|
| Training set | 80% | Fit the model |
| Validation set | 10% | Tune hyperparameters |
| Test set | 10% | Final, one-time evaluation |

> **Rule**: Use the test set exactly once. Multiple evaluations on test data inflate performance estimates.

**Formal training / testing / generalization objective**

Let $\mathcal{H}$ be the hypothesis class. The learner picks

$$h^{*} = \underset{h \in \mathcal{H}}{\arg\min}\; e_{TR}(h), \qquad e_{TR}(h) = \frac{1}{|D_{TR}|} \sum_{(\mathbf{x},y) \in D_{TR}} \ell(h(\mathbf{x}), y)$$

We then estimate deployment error by

$$e_{TE}(h^{*}) = \frac{1}{|D_{TE}|} \sum_{(\mathbf{x},y) \in D_{TE}} \ell(h^{*}(\mathbf{x}), y)$$

which is an unbiased estimator of the true **generalization loss**

$$\epsilon(h^{*}) = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{P}} \bigl[\ell(h^{*}(\mathbf{x}), y)\bigr]$$

*only if* $h^{*}$ was chosen independently of $D_{TE}$. Peeking at the test set even once and tweaking based on the result destroys that guarantee — that is why we introduce a validation set as a proxy.

**When a random split is wrong**

| Data has… | Correct split |
|-----------|---------------|
| Temporal drift (e.g. spam) | Train on the past, test on the future — never the reverse |
| Grouped samples (e.g. multiple visits per patient) | Split by group so no patient appears in both train and test |
| Class imbalance | Stratify so class proportions match across splits |

### 1.5 No Free Lunch Theorem

No single ML algorithm is universally best. Every algorithm has inductive biases — assumptions baked in about what "good" generalization looks like. Performance depends on whether those biases match the true data distribution.

---

## Module 2: k-Nearest Neighbors (k-NN)

### 2.1 Core Assumption

**Similar inputs have similar labels.**

Given a test point $\mathbf{x}$, find the $k$ training points closest to it and take a majority vote (classification) or average (regression).

### 2.2 Algorithm

```
kNN(D, x, k):
  for each (x_i, y_i) in D:
    compute distance d(x, x_i)
  sort by distance
  take top-k nearest neighbors
  return majority_vote(labels of top-k)
```

**Distance metric** (Euclidean):
$$d(\mathbf{x}, \mathbf{z}) = \|\mathbf{x} - \mathbf{z}\|_2 = \sqrt{\sum_{j=1}^{d}(x_j - z_j)^2}$$

**Common alternative distance metrics**

| Metric | Formula | When to prefer |
|--------|---------|----------------|
| L1 / Manhattan (taxicab) | $\sum_j \lvert x_j - z_j \rvert$ | Grid-like data, robust to outlier dimensions |
| L2 / Euclidean | $\sqrt{\sum_j (x_j - z_j)^2}$ | Default; isotropic feature space |
| Minkowski ($L_p$) | $\bigl(\sum_j \lvert x_j - z_j \rvert^p\bigr)^{1/p}$ | Tunable between L1 ($p{=}1$) and L2 ($p{=}2$) |
| Mahalanobis | $\sqrt{(\mathbf{x}-\mathbf{z})^\top \Sigma^{-1}(\mathbf{x}-\mathbf{z})}$ | Correlated features with different scales |

**Worked example — 1-NN by hand (L2)**

Training points:

| point | $(x_1, x_2)$ | label |
|-------|--------------|-------|
| $\mathbf{x}_1$ | $(1, 2)$ | +1 |
| $\mathbf{x}_2$ | $(1, 4)$ | +1 |
| $\mathbf{x}_3$ | $(5, 4)$ | +1 |
| $\mathbf{x}_4$ | $(3, 1)$ | −1 |
| $\mathbf{x}_5$ | $(3, 2)$ | −1 |

Test point $\mathbf{z} = (5, 1)$. Squared distances (no need to `sqrt` — order is preserved):

$$D^2(\mathbf{z}, \mathbf{x}_1) = 16 + 1 = 17,\; D^2(\mathbf{z}, \mathbf{x}_4) = 4,\; D^2(\mathbf{z}, \mathbf{x}_5) = 4 + 1 = 5$$

The nearest neighbor is $\mathbf{x}_4 = (3,1)$ with label −1, so 1-NN predicts −1.

**Minimal pure-Python k-NN (no NumPy)**

```python
from collections import Counter
from math import sqrt

def knn_predict(train, x_query, k=3):
    """train: list of (features_tuple, label). x_query: features_tuple."""
    def d(a, b):
        return sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    ranked = sorted(train, key=lambda pair: d(pair[0], x_query))
    top_k = [label for _, label in ranked[:k]]
    return Counter(top_k).most_common(1)[0][0]

train = [((1,2), +1), ((1,4), +1), ((5,4), +1),
         ((3,1), -1), ((3,2), -1)]
print(knn_predict(train, (5, 1), k=1))   # -1
print(knn_predict(train, (2, 3), k=3))   # +1
```

This is $O(nd)$ per query — fine for teaching, terrible for millions of points (see §2.4).

### 2.3 Hyperparameter: k

| k value | Decision boundary | Risk |
|---------|------------------|------|
| Low (k=1) | Complex, jagged | Overfits to noise |
| High (k=n) | Smooth, possibly underfits | Misses local structure |

**Best k**: tuned on the validation set.

### 2.4 Enhancing k-NN

**Tie-breaking**: If votes are tied, fall back to $k-2$ neighbors and repeat until tie resolves.

**Data Augmentation**: Apply label-preserving transformations to training data to increase effective $n$:
- Image: rotate, flip, crop, add noise
- Text: synonym replacement, back-translation

**k-d Trees** (speedup):
- Naive search: $O(nd)$ per query
- k-d tree: $O(n)$ amortized per query (after $O(nd \log n)$ build)
- Partition feature space into axis-aligned hyperplanes
- Prune branches that can't contain closer points

**Why k-d tree query is $O(n)$ (not $O(nd)$)** — assume each split cleanly halves the remaining candidates. Comparisons per level form the geometric series

$$n + \tfrac{n}{2} + \tfrac{n}{4} + \tfrac{n}{8} + \cdots \;<\; 2n$$

since $1 + \tfrac{1}{2} + \tfrac{1}{4} + \cdots = 2$. The $d$ factor of naive search disappears because we compare one coordinate at a time and prune whole subtrees. This works beautifully in low dimensions but degrades as $d$ grows (see §2.5).

**Trade-off**: naive k-NN is O(1) to "train" (just store the data) but O(nd) to query. k-d trees flip that — O($nd \log n$) to build, O(n) to query. Choose based on whether you have more training or more testing data.

**Python — use scipy's k-d tree for real workloads**

```python
import numpy as np
from scipy.spatial import cKDTree

rng = np.random.default_rng(0)
X_train = rng.standard_normal((10_000, 8))
y_train = rng.integers(0, 3, size=10_000)
X_test  = rng.standard_normal((500, 8))

tree = cKDTree(X_train)                    # O(nd log n) build
_, idx = tree.query(X_test, k=5)           # (500, 5) neighbor indices

# 5-NN majority vote
preds = np.array([np.bincount(y_train[row]).argmax() for row in idx])
```

### 2.5 Curse of Dimensionality

For k-NN to work, we need the $k$ neighbors to actually be *close*.

**Derivation.** If $n$ points are drawn uniformly from the unit hypercube $[0,1]^d$, a sub-cube of edge length $\ell$ has volume $\ell^d$ and therefore contains roughly $n \cdot \ell^d$ points. To capture $k$ neighbors we need

$$n \cdot \ell^d \;\approx\; k \quad\Longrightarrow\quad \ell \approx \left(\frac{k}{n}\right)^{1/d}$$

As $d$ grows the exponent $1/d$ shrinks toward $0$, so $\ell \to 1$ — the "nearest" neighbors have to span almost the entire feature space to reach $k$ of them, which means they are no longer near in any meaningful sense.

| $d$ | $\ell$ (for $k/n = 0.1$) |
|-----|--------------------------|
| 1 | 0.10 |
| 2 | 0.32 |
| 10 | 0.79 |
| 100 | 0.98 |

As $d \to \infty$, $\ell \to 1$ — neighbors fill the entire space and are no longer "near".

> **Implication**: k-NN degrades in truly high-dimensional spaces. Dimensionality reduction (PCA, embeddings) is often needed before applying k-NN.

**Escape hatch — intrinsic dimensionality.** Real data is rarely uniformly distributed across all $d$ dimensions. Features are usually correlated (blood pressure ↔ weight; adjacent pixels; word co-occurrences), so the data actually lives on a lower-dimensional manifold embedded in $\mathbb{R}^d$. When that intrinsic dimension is small, k-NN can still work in nominally high-$d$ spaces — which is exactly why deep-learning-derived face embeddings + 1-NN power production face recognition at Facebook, Google, etc.

**Python — watch the curse take hold**

```python
import numpy as np

rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000]:
    X = rng.random((1000, d))                         # uniform on [0,1]^d
    z = rng.random(d)
    dists = np.linalg.norm(X - z, axis=1)
    print(f"d={d:4d}   min={dists.min():.3f}   max={dists.max():.3f}   "
          f"ratio={dists.max()/dists.min():.3f}")
# d=   2   ratio ≈ 100+   (near/far very distinguishable)
# d=1000   ratio ≈  1.1   (everything is roughly the same distance away)
```

---

## Module 3: NumPy & Vectorization

### 3.1 Why Vectorize?

Python loops are slow. NumPy operations run on optimized C/Fortran code.

```python
# Python loop (slow ~4.2 seconds)
result = 0
for i in range(n):
    result += x[i] * y[i]

# NumPy (fast ~1ms — ~4000x speedup)
result = np.dot(x, y)
```

### 3.2 Key NumPy Functions

| Function | Description |
|----------|-------------|
| `np.dot(A, B)` | Matrix/vector dot product |
| `A.T` or `np.transpose(A)` | Transpose |
| `np.sum(A, axis=0)` | Sum along rows (column sums) |
| `np.sum(A, axis=1)` | Sum along columns (row sums) |
| `np.argmin(A, axis=1)` | Index of min per row |
| `np.sqrt(A)` | Element-wise square root |
| `np.linalg.norm(x)` | Euclidean norm |
| `A[np.newaxis, :]` | Add dimension (broadcasting) |
| `np.tile(A, (r, c))` | Repeat array r×c times |

### 3.3 Broadcasting

NumPy automatically expands dimensions for element-wise operations:

```python
# Subtract row vector from every row of matrix
X - mu  # X: (n, d), mu: (d,) → broadcasts to (n, d)
```

---

## Module 4: Facial Recognition with k-NN

### 4.1 Problem Setup

- Training: $n$ face images → matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ (each row = flattened image)
- Test: $m$ face images → matrix $\mathbf{Z} \in \mathbb{R}^{m \times d}$
- Goal: for each test image, find the nearest training image and return its label

### 4.2 Euclidean Distance Matrix

Computing all pairwise distances naively is expensive. Use the identity:

$$\|\mathbf{x} - \mathbf{z}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{z}\|^2 - 2\mathbf{x}^\top\mathbf{z}$$

**Derivation** (natural extension of the 1-D identity $(a-b)^2 = a^2 + b^2 - 2ab$):

$$\begin{aligned}
D^2(\mathbf{x}, \mathbf{z}) &= (\mathbf{x} - \mathbf{z})^\top (\mathbf{x} - \mathbf{z}) \\
&= \mathbf{x}^\top\mathbf{x} - \mathbf{x}^\top\mathbf{z} - \mathbf{z}^\top\mathbf{x} + \mathbf{z}^\top\mathbf{z} \\
&= \mathbf{x}^\top\mathbf{x} + \mathbf{z}^\top\mathbf{z} - 2\,\mathbf{x}^\top\mathbf{z}
\end{aligned}$$

The payoff: we can compute *all* pairwise distances with three matrix operations — one matmul ($\mathbf{G}$) and two vector norms ($\mathbf{S}, \mathbf{R}$).

In matrix form for all pairs $(i, j)$:

$$\mathbf{D} \odot \mathbf{D} = \mathbf{S} + \mathbf{R} - 2\mathbf{G}$$

Where:
- $\mathbf{G} = \mathbf{X}\mathbf{Z}^\top \in \mathbb{R}^{n \times m}$ (Gram matrix)
- $\mathbf{S}_{ij} = \|\mathbf{x}_i\|^2$ (row norms of $\mathbf{X}$, broadcast over columns)
- $\mathbf{R}_{ij} = \|\mathbf{z}_j\|^2$ (row norms of $\mathbf{Z}$, broadcast over rows)

> **Careful — $\mathbf{S} \ne \mathbf{X}\mathbf{X}^\top$.** The Gram $\mathbf{X}\mathbf{X}^\top$ has shape $n \times n$, but $\mathbf{S}$ must have shape $n \times m$: it is $\|\mathbf{x}_i\|^2$ *repeated across $m$ columns*. Likewise $\mathbf{R}$ is $\|\mathbf{z}_j\|^2$ repeated across $n$ rows.

**Worked numerical example** (matching Cornell's slide with $n{=}2$, $m{=}3$, $d{=}2$):

$$\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},\qquad \mathbf{Z} = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

1. Gram: $\mathbf{G} = \mathbf{X}\mathbf{Z}^\top = \begin{bmatrix} 9 & 12 & 15 \\ 19 & 26 & 33 \end{bmatrix}$
2. Row norms of $\mathbf{X}$: $\{5, 25\}$ → broadcast:
   $\mathbf{S} = \begin{bmatrix} 5 & 5 & 5 \\ 25 & 25 & 25 \end{bmatrix}$
3. Row norms of $\mathbf{Z}$: $\{17, 29, 45\}$ → broadcast:
   $\mathbf{R} = \begin{bmatrix} 17 & 29 & 45 \\ 17 & 29 & 45 \end{bmatrix}$
4. Squared distances $\mathbf{S} + \mathbf{R} - 2\mathbf{G} = \begin{bmatrix} 4 & 10 & 20 \\ 4 & 2 & 4 \end{bmatrix}$

Sanity check: $D^2(\mathbf{x}_1, \mathbf{z}_2) = (1-2)^2 + (2-5)^2 = 1 + 9 = 10$. ✅

### 4.3 NumPy Implementation

```python
import numpy as np

def l2_distances(X, Z):
    """Compute pairwise Euclidean distances between rows of X and Z."""
    G = X @ Z.T                                    # (n, m) Gram matrix
    S = np.sum(X**2, axis=1, keepdims=True)        # (n, 1)
    R = np.sum(Z**2, axis=1, keepdims=True).T      # (1, m)
    D_sq = S + R - 2 * G                           # (n, m)
    D_sq = np.maximum(D_sq, 0)                     # numerical safety
    return np.sqrt(D_sq)

def knn_predict(X_train, y_train, X_test, k=1):
    """k-NN classifier using Euclidean distance."""
    D = l2_distances(X_train, X_test)              # (n_train, n_test)
    # For each test point, get k nearest training indices
    nn_indices = np.argsort(D, axis=0)[:k, :]     # (k, n_test)
    nn_labels = y_train[nn_indices]                # (k, n_test)
    # Majority vote
    predictions = np.array([
        np.bincount(col).argmax() for col in nn_labels.T
    ])
    return predictions
```

### 4.4 Facial Recognition Pipeline

1. **Load images** — flatten each to a vector (e.g., 64×64 image → 4096-d vector)
2. **Compute distance matrix** — all train-test pairs via vectorized formula
3. **Find nearest neighbor** — `np.argmin(D, axis=0)` for 1-NN
4. **Return label** — identity of the closest training face

### 4.5 Why k-NN is a natural fit for facial recognition

From Laurens van der Maaten (Facebook AI Research):

- A face recognition system may need to distinguish **hundreds of millions of identities** (every user on a platform).
- Each identity typically has only a **handful of photos** — often just one enrollment image.
- Most classifiers assume the opposite: *few* classes, *many* examples per class. Nearest-neighbor flips comfortably to "many classes, few examples per class".
- Rather than compare raw pixels (millions of dimensions, curse of dimensionality applies), production systems learn a compact **face embedding** via a deep CNN and then run 1-NN in that low-dimensional embedding space. The classic distinguishing template features (eye-to-eye distance ÷ nose length, etc.) are learned automatically instead of hand-engineered.

**End-to-end recipe (embedding + 1-NN)**

```python
# Pseudocode — assume `embed(img)` is a pretrained face encoder → R^128
X_train = np.stack([embed(img) for img in enrollment_images])  # (n, 128)
y_train = np.array(enrollment_identities)

def identify(new_face_img, threshold=0.6):
    z = embed(new_face_img)                     # (128,)
    dists = np.linalg.norm(X_train - z, axis=1) # (n,)
    i = np.argmin(dists)
    if dists[i] > threshold:
        return "unknown"                        # reject if no one is close enough
    return y_train[i]
```

The `threshold` is what turns k-NN identification into an **open-set** recognizer — the ability to say "none of the above" when the query face doesn't belong to anyone enrolled.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Supervised learning | Learn $h: \mathbf{x} \to y$ from labeled examples |
| Loss functions | Measure prediction error; choice depends on task type |
| Data splits | Train/validation/test — use test set once only |
| No Free Lunch | No universally best algorithm |
| k-NN | Majority vote of k nearest neighbors |
| Hyperparameter k | Tune on validation set; low k overfits, high k underfits |
| Curse of dimensionality | k-NN breaks in high-d; use dim reduction |
| Vectorization | NumPy is ~4000x faster than Python loops |
| Distance matrix | $\mathbf{D}^2 = \mathbf{S} + \mathbf{R} - 2\mathbf{G}$ |
 
 
3. Examples
•	Stock Prediction 
o	x: market stats (e.g., Pepsi stock, weather)
o	y: stock goes up or down
•	Medical Diagnosis 
o	x: patient data (age, blood pressure, symptoms)
o	y: ill or not
•	Face Recognition 
o	x: image pixels
o	y: identity of person
•	Spam Detection 
o	x: words in email
o	y: spam or not
 
4. Types of Outputs (y)
•	Binary Classification 
o	Two classes (e.g., +1 / −1, yes/no, up/down)
•	Multiclass Classification 
o	Multiple categories (e.g., 1–150 identities, e.g. facial recognition)
•	Regression 
o	Continuous values (e.g., house price).  Like multiclass but where in the former the classes themselves are binary regression has a sense of closeness.  For example, assume you are trying to predict the height (label) of a person based on other data (features), such as their gender, weight, age, etc. If you cast this as a multiclass classification problem, you might define each class as a rounded incremental value of 1cm. If we try to classify someone who is actually 183cm tall and return a label of 182cm, we are just as wrong as if we had predicted 140cm. Most people would consider a predicted height of 182cm to be much closer to the truth (183cm) than 140cm is, but since each height is its own class and there is typically no assumption that one class is more similar to another in classification problems, we’ve simply failed to predict the height correctly. For this example, we should instead choose regression. In regression, we typically assume that, because the label can be any real value, you will almost never hit it exactly but hope to get very close.  An example problem where multiclass classification would be the best choice is facial recognition. Bill Gates, Steve Jobs, and Linus Torvalds would each be their own class. We see here that each class is separate and distinct; selecting the wrong class is simply an incorrect label. There is no notion of similarity between classes since you are either right or wrong. For image classification problems, multiclass classification is the appropriate choice.
 
5. Types of Features (x)
•	Numerical stats (e.g., patient data)
•	Text (e.g., email words)
•	Images (e.g., pixels)
 
6. Key Takeaways
•	Supervised learning uses labeled data
•	Goal: generalize from past data to future predictions
•	Works across many domains (finance, healthcare, vision, etc.)
•	Choosing the right algorithm is a key responsibility for data scientists
📘 Concise Notes: The Machine Learning Setup
 
1. Core Idea
•	Supervised machine learning builds programs that learn from past data
•	Goal: make decisions or predictions from data (instead of manually coding rules)
 
2. Key Concepts
•	Features (x): Known inputs (e.g., age, symptoms)
•	Label (y): Target output to predict (e.g., sick or healthy)
•	Must clearly separate what you know (x) from what you want to infer (y)
 
3. Workflow
•	Training Stage 
o	Use historical data (features + labels)
o	Learn a function (model) from data
•	Testing Stage 
o	Apply learned function to new, unseen data
o	Generate predictions
 
4. Example
•	Medical diagnosis: 
o	x: patient data (temperature, age, symptoms)
o	y: diagnosis (sick/healthy)
o	Train on past patient records → predict for new patients
 
5. Hypothesis (Model)
•	h (hypothesis): Learned function mapping x → y
•	Hypothesis Class (H): Set of all possible functions the algorithm can choose from
•	Final model = best hypothesis found during training
 
6. Algorithms & Choice
•	Different algorithms = different hypothesis classes
•	Each algorithm has its own parameters and behavior
•	Data scientist’s job: choose the best algorithm for the problem
 
✅ Summary:
Machine learning replaces manual rules with models learned from data by distinguishing features and labels, training on past examples, and selecting the best hypothesis to make future predictions.
📘 Concise Notes: Features of a Problem
 
1. Core Idea
•	Features = relevant characteristics used to describe data
•	They are represented as a feature vector (x)
•	Feature design strongly impacts model performance
 
2. Key Insight
•	How data is encoded into vectors (representation + content) directly affects learning outcomes
•	Domain expertise helps choose meaningful features
 
3. Types of Feature Vectors
🔹 Heterogeneous Features
•	Mixed types with different units and scales 
o	Example: age, blood pressure, height
•	Challenge: values vary widely in range
•	Important: handle scaling properly to avoid bias toward large-value features
 
🔹 Bag-of-Words (Text Features)
•	Converts text into fixed-size vectors
•	Steps: 
o	Choose a set of important words (vocabulary) (e.g. d=100,000 most common words in English lang.)
o	Count occurrences of each word in a document
•	Ignores word order
•	Results in sparse vectors (mostly zeros)
 
🔹 Pixel Features (Images)
•	Images represented as numerical pixel values
•	Example: 
o	RGB images → 3 values per pixel (0–255)
•	Flatten image → long vector (very high dimensional)
•	Advanced models (e.g., CNNs) preserve 2D structure
 
4. Key Challenges
•	Feature scaling (especially with heterogeneous data)
•	High dimensionality (e.g., images, text)
•	Sparsity in text data
 
5. Key Takeaways
•	Features define how data is understood by the model
•	Better feature representation → better predictions
•	Choosing and transforming features is a critical step in ML
 
✅ Summary:
Features are the input representation of data, and their design—including type, scale, and encoding—plays a major role in the success of machine learning models.

📘 Concise Notes: Loss Functions & Learning from Mistakes
 
1. Core Idea
•	Supervised learning improves by learning from mistakes
•	The model adjusts its function h to perform better on training data
 
2. Loss Function
•	A loss function measures how well the model performs
•	It gives a clear objective to optimize
•	The model tries to minimize loss
✅ Key properties:
•	Lower loss = better performance
•	Loss = 0 ⇒ perfect predictions
 
3. Purpose
•	Quantifies how many errors or how big the errors are
•	Guides the learning process by telling the model what to improve
 
4. Examples of Loss Functions
🔹 Zero-One Loss (Classification)
•	Used for classification problems (right/wrong, no in-betweens)
•	Counts incorrect predictions: 
o	Correct → 0 loss
o	Incorrect → 1 loss
•	Output: error rate 
o	Example: 10% loss = 10% misclassified (e.g. how many emails mis-identified in spam tagger)
 
🔹 Absolute Loss (Regression)
•	Used for predicting continuous values (how far off are you)
•	Measures error as: absolute value of predicted value – actual value
•	Because the suffered loss grows linearly with mispredictions, it is more suitable for noisy data (when some mispredictions are unavoidable and shouldn't dominate the loss).
•	Example: 
o	Actual = $120,000, Predicted house sale = $130,000
o	Loss = $10,000
o	
🔹 Squared Loss (Regression)
•	The squaring has two effects:
o	The loss suffered is always non-negative.
o	The loss suffered grows quadratically with the absolute mispredicted amount.
•	Squared loss can be problematic when your data has noisy labels, which could incur a large loss and distract the classifier (it punishes big mistakes more)
 
5. Overall Loss
•	Computed as the average loss across all data points
 
6. Key Takeaways
•	Loss functions define what “good performance” means
•	They guide the model to improve predictions
•	Different tasks require different loss functions
 
✅ Summary:
Supervised learning uses loss functions to measure errors and guide the model to improve, aiming to minimize mistakes and achieve the best possible predictions.

📘 Concise Notes: Generalization & Data Splitting in Supervised Learning
 
1. Core Idea
•	Goal is not just to perform well on training data, but to generalize to future (unseen) data
•	Models should make accurate predictions on new inputs
 
2. Key Problem
•	Powerful models may memorize training data instead of learning patterns
•	This is called overfitting
•	Memorization ≠ true learning
 
3. Train–Test Split
•	Split dataset into: 
o	training data (80%)
o	validation data (10%)
o	test data (10%)
•	Test data must remain unseen during training
•	All three sets should be drawn from the same distribution. A good rule of thumb is to make sure that the data sets simulate the real-life settings in which the algorithm will operate.
✅ Purpose:
•	Simulates real-world deployment on new data
 
4. Proper Data Splitting
🔹 Random Split (Typical Case)
•	Shuffle data randomly before splitting
•	Ensures both sets have similar distributions
 
🔹 Special Cases
•	Time-based (temporal component) data 
o	Train on past, test on future (make sure that you never predict the past from the future and always the future from the past)
o	Example: spam (attacks evolving over time, so train on email from Mon-Thurs and test on data from Fri-Sun).
•	Grouped data (e.g., patients) 
o	Train and test on different groups
o	Ensures model works on new individuals
 
5. Second Problem: Test Data Leakage
•	Looking at test data multiple times = cheating
•	Repeated tuning based on test results biases performance
•	Leads to overly optimistic results
 
6. Solution: Validation Set
•	Further split training data into: 
o	Train set → train model
o	Validation set → tune & improve model
•	Process: 
1.	Train on training set
2.	Evaluate and tweak using validation set
3.	Repeat until satisfied
4.	Retrain on train + validation
5.	Evaluate once on test set
 
7. Key Takeaways
•	Always evaluate on unseen data
•	Avoid overfitting and test data leakage
•	Use validation for tuning, test for final unbiased evaluation
 
✅ Summary:
To ensure reliable performance, supervised learning must generalize beyond training data by using proper data splits (train/validation/test) and avoiding overfitting or misuse of test data.

📘 Concise Recap: No Free Lunch & Assumptions in Machine Learning
 
1. Key Idea
•	You cannot make reliable predictions if the world behaves randomly
•	Without structure or patterns, machine learning cannot work
 
2. Example Insight
•	Given data points, you might predict a reasonable value (e.g., ~2.2)
•	But actual value could be anything (e.g., 4.5) if no pattern exists
•	Shows that predictions rely on assumptions
 
3. No Free Lunch Theorem
•	No single algorithm works best for all problems
•	Every ML algorithm makes assumptions about the data
 
4. Role of Assumptions
•	Example assumption: 
o	“Similar inputs → similar outputs” (smoothness)
•	Only when such assumptions hold can models generalize and predict
 
5. Data Scientist’s Role
•	Understand which assumptions each algorithm makes
•	Ensure those assumptions match the real data
•	Choose the right algorithm accordingly
 
✅ Summary:
Machine learning depends on underlying patterns in data. Without assumptions about structure, prediction is impossible—so selecting the right model means matching its assumptions to the problem.
📘 Concise Notes: k-Nearest Neighbors (k-NN)
 
1. Core Idea
•	Predicts a data point’s label based on the labels of its k closest neighbors
•	Assumes: similar inputs → similar outputs
 
2. Similarity Measure
•	Typically measured using Euclidean distance (the straight-line distance between two points in space)
•	Closer points = more similar
 
3. Classification Rule
•	Choose k nearest neighbors to a test point
•	Assign the label by majority vote among those neighbors
 
4. How It Works
1.	Pick a value for k (e.g., 3 or 5)
2.	Compute distances from test point to all training points
3.	Select the k closest points
4.	Assign the most common label among them
 
5. Example
•	With k = 3: 
o	2 neighbors = positive (blue)
o	1 neighbor = negative (red)
✅ Prediction: positive (majority)
 
6. Real-World Example (Loan Prediction)
•	Features: age, credit score, income, etc.
•	Labels: default or not default
•	Process: 
o	Find k closest past applicants
o	Predict based on their outcomes (majority vote)
 
7. Key Takeaways
•	Simple, intuitive algorithm
•	No explicit training step (stores data)
•	Performance depends on: 
o	Choice of k
o	Distance metric
•	Works best when similarity assumption holds
 
✅ Summary:
k-NN classifies a data point by looking at the most similar examples and using majority vote, relying on the assumption that nearby points share similar labels.
 
📘 Concise Notes: Choosing k & Distance in k-NN
 
1. Key Parameter: k
•	k = number of neighbors used for prediction
•	Typical values: odd numbers (e.g., 1–15)
•	Controls model behavior
 
2. Effect of k on Model
🔹 Small k (e.g., k = 1)
•	Very complex / jagged decision boundary
•	Sensitive to noise and mislabeled data
•	Can create small “islands” around outliers
•	Risk: overfitting
 
🔹 Larger k (e.g., k = 3, 9)
•	Smoother decision boundary
•	Noise gets averaged out (outliers “outvoted”)
•	More stable predictions
•	Risk: may lose important details (underfitting)
 
✅ Key Insight:
•	k controls complexity 
o	Small k → complex, sensitive
o	Large k → simple, robust
 
3. Noise Handling
•	Mislabelled points can distort predictions
•	Increasing k reduces their influence through majority voting
 
4. Distance Metric
•	Defines how “similarity” is measured
Common choices:
•	Euclidean distance (most common)
•	Manhattan distance
•	Mahalanobis distance
•	Minkowski distance
✅ Important:
•	Distance metric must reflect meaningful similarity for the data
 
5. Advanced Idea
•	Instead of using raw features, models can learn better representations
•	Example: deep learning (e.g., CNNs) + k-NN for tasks like face recognition
 
6. Key Takeaways
•	Choosing k balances complexity vs. noise sensitivity
•	Distance metric choice is critical for meaningful comparisons
•	Good performance depends on both k and similarity definition
 
✅ Summary:
In k-NN, the choice of k controls how sensitive the model is to noise, while the distance metric defines similarity—both are crucial for accurate predictions.
📘 Concise Notes: Choosing k in k-NN
 
1. Hyperparameter
•	k (number of neighbors) is the only parameter in k-NN
•	Can be tuned to improve model accuracy
•	Often chosen as an odd number (avoids ties)
 
2. Effect of Small k (e.g., k = 1)
•	Highly sensitive to data variations
•	Produces complex / jagged decision boundaries
•	Captures noise and mislabeled points → creates “islands”
•	Risk: overfitting
 
3. Effect of Larger k (e.g., k = 3, 9)
•	Averages over more neighbors
•	Removes noise influence (outliers get “outvoted”)
•	Produces smoother, simpler boundaries
•	More stable predictions
 
4. Trade-off
•	Low k 
o	Captures detail
o	Sensitive to noise
•	High k 
o	Robust to noise
o	May oversimplify and miss patterns (underfitting)
 
5. Key Takeaways
•	Best k depends on the dataset
•	Increasing k: 
o	↓ complexity
o	↑ smoothness
o	↓ noise sensitivity
•	Too large k can misclassify important structures
 
✅ Summary:
k controls the balance between sensitivity and smoothness—small k risks overfitting, large k risks oversmoothing, so choosing the right k is crucial.
     

📘 Concise Notes: Improving k-Nearest Neighbors (k-NN)
 
1. Handling Ties
•	Usually use odd numbers for k
•	Ties can occur when neighbors have equal votes across classes
•	Simple fix: 
o	Reduce k (e.g., from 3 → 1) until tie is resolved
•	k = 1 never produces ties
 
2. Data Augmentation
•	Increase training data by applying label-preserving transformations
•	Examples (for images): 
o	Translate, rotate, flip, crop, warp (label remains the same, e..g. a cat, even as the image changes)
•	Helps: 
o	Expand dataset size
o	Improve model performance
•	Especially useful for k-NN (works better with more data)
•	More data → better k-NN performance
•	Augmentation makes dataset denser and more informative
 
3. Choosing distance function
•	how you determine the distance between two points is critical
•	Euclidean distance — also known as L2 distance — with k-NN is common
o	Grid-like or city-block data
o	may be suboptimal in some settings where features follow particular structures (e.g., are normalized)
•	L1 (taxicab)
o	can only travel along vertical and horizontal axes, much like a taxicab driving through a city grid
•	Minkowski distance
•	Mahalanobis distance 
o	General cases
•	Manhattan distance
o	Correlated features / different scales
 
4. Computational Challenge
•	k-NN becomes slow with large datasets
•	Reason: 
o	Must compute distance to every training point
 
5. Speed Optimization
•	Use specialized data structures: 
o	k-d trees
o	Ball trees
•	a k-d tree structure that splits the data space into  boxes. If you are trying to find the nearest neighbors for the yellow test point in the lower box, rather than computing the distance from the test point to all of the training points ( computations) — even to the points in the upper box that are clearly far — you simply compute the distance to the box containing those points in the upper box and ignore them if they are further away than the box is.
•	Naive nearest neighbors just involves storing the training points in memory, whereas building k-d trees also involves finding the axis split values. Hence, k-d trees are a little more computationally expensive during training but are cheaper during testing, whereas naive nearest neighbors is cheap during training but expensive during testing. Depending on whether you have more training or testing data, you will have to decide which implementation of nearest neighbors to use.
✅ How they help:
•	Partition space into regions (boxes/balls)
•	Skip distance calculations for far-away regions
•	Avoid unnecessary computations
 
6. Key Takeaways
•	Use smaller k to resolve ties
•	Augment data to boost performance
•	Optimize speed with efficient data structures
 
✅ Summary:
k-NN can be improved by handling ties, increasing data through augmentation, and speeding up computations with smart data structures, making it more accurate and efficient.

📘 Concise Notes: k-NN & the Curse of Dimensionality
 
1. Key Assumption of k-NN
•	Assumes: similar inputs → similar labels
•	Relies on finding nearby (similar) points
 
2. What is High-Dimensional Data?
•	Data represented as feature vectors with many attributes
o	Each feature = one dimension 
o	More features → more dimensions
•	Example: 
o	1 feature → 1D
o	100 features → 100D space
✅ Example
📊 Medical Data
Suppose you describe a patient using:
•	Age
•	Height
•	Weight
•	Blood pressure
👉 This is 4-dimensional data
Now imagine adding:
•	Cholesterol
•	Heart rate
•	Genetic markers
•	Full medical history
•	Thousands of test results
👉 This quickly becomes hundreds or thousands of dimensions → high-dimensional data
🖼️ Another Example: Images
•	A 6-megapixel image has 6 million pixels
•	Each pixel has 3 values (RGB)
👉 Total features = 18 million dimensions
✅ Even though it’s just an image, mathematically it’s very high-dimensional

 
3. Curse of Dimensionality
•	In high dimensions: 
o	Data points become very spread out
o	No points are truly “close” anymore
•	Result: 
o	Similarity becomes meaningless
o	k-NN fails to find useful neighbors
 
4. Why This Happens
•	As dimensions increase: 
o	Matching similarity across all features becomes unlikely
o	Example: 
	Matching height alone → easy
	Matching height + age + weight + history → very unlikely
 
5. Impact on k-NN
•	k-NN becomes ineffective in truly high-dimensional data
•	Cannot rely on nearest neighbors because no one is similar
 
6. When k-NN Still Works ✅ If data is not truly high-dimensional:
•	Correlated features
o	Example: weight ↔ blood pressure
o	Reduces effective dimensionality
•	Intrinsic low-dimensional structure
o	Only a few features actually matter
•	Example: Images (faces)
o	Raw pixels = millions of dimensions
o	But only a few features (e.g., shape, hair, age) matter (e.g. report to policy a theft they ask a few questions about person and can really narrow down with that)
 
7. Key Insight
•	High-dimensional data may still be effectively low-dimensional
•	k-NN works if meaningful similarity still exists
 
8. Data Scientist’s Role
•	Check if: 
o	Data is truly high-dimensional
o	Or has underlying structure/correlation
•	If not → consider different algorithms
 
✅ Summary:
k-NN struggles in high-dimensional spaces due to the curse of dimensionality, but can still work when data has underlying structure or correlated features that preserve meaningful similarity.
The animation below shows what happens to data points with two dimensions after we add a third dimension. This dataset is randomly sampled, but a similar effect can be observed on real-world, correlated data.
 
📘 Concise Notes: Using NumPy for k-NN Efficiency
•	Used for fast linear algebra operations
•	NumPy can be thousands of times faster (~4000×)
•	Example: NumPy: 1 day → Python loops: ~11 years
•	Avoids loops for large computations
•	Uses vectorized operations (matrix math)
•	NumPy is essential for scalable ML implementations
•	much of the NumPy library is written in C
o	allows for more efficient lower-level operations than Python offers natively
•	Vectorization
o	the process of rewriting a loop so that instead of processing elements of an array one at a time, it processes several elements of the array simultaneously.
o	In the case of vectorization, NumPy takes advantage of "single instruction, multiple data" parallelism, which is a type of parallel processing that calls for the processor to run the same operation simultaneously for all entries in the array.
•	The main object in NumPy is an n-dimensional array, an object with elements of the same data type.
o	To create an array, you need to pass a list of objects into the function np.array()
•	Attributes of NumPy Arrays
o	Three important ones: shape (.shape); dimension (.ndim); total number of elements (.size)
•	Vectors in NumPy
o	You can use arrays to represent a vector, which is essentially a 1-D array. 
o	There are three ways to represent a vector using NumPy (NumPy, column, and row).
	You can use .reshape() to transform a vector into a column or row vector
	n [5]: v1 = np.array([3,4,5])
	
	In [6]: v1
	Out[6]: array([3, 4, 5])
	
	In [7]: v1.shape
	Out[7]: (3,)
	
	In [8]: v2 = v1.reshape(3,1)
	
	In [9]: v3 = v1.reshape(1,3)
o	The three representations of a vector (NumPy, column, and row) are often not compatible for most operations. Some operations might “work” but will not return the expected output.
o	You can transform any vector (matrix) into a NumPy vector with .flatten().
o	There are also a few functions in NumPy for creating common matrices:
	np.eye() creates an identity matrix
	np.zeros() creates a matrix of all zeros
	np.arange() creates a vector of equally spaced values
	Create a vector of equally spaced values: np.arange(10)
	Create a vector of equally spaced values: np.arange(1, 10, 2)
	The syntax for the np.arrange() function is np.arange([start, ]stop, [step, ]). For details, review the documentation for np.arrange()
	To perform matrix multiplication, use np.dot() or the @ symbol. For example, you could multiply matrices X and Y by writing either np.dot(X, Y) or X @ Y
	np.dot() must be used (instead of @) when multiplying a matrix by a scalar.
	Recall that the function np.dot() expects matrices to have compatible dimensions in that the first matrix must have the same number of columns as the rows of the second matrix
	When multiplying two matrices, the number of columns in the first (e.g., 3 columns in X) must match the number of rows in the second (e.g., 3 rows in Y).  If not must transpose one of them.
	To transpose a matrix, use .transpose(). An alternative and more concise syntax is .T
	The function np.outer() is similar to np.dot() but np.outer() transposes the first matrix before applying np.dot(). In other words, np.outer(X, Y) is equivalent to np.dot(X.T, Y)
Other Matrix Operations
unction	Example
Largest element in X	np.amax(X)
Largest elements along the first axis (viz. run down all columns)	np.amax(X, axis=0)
Smallest element in X	np.amin(X)
Smallest elements along the second axis
Run across all rows	np.amin(X, axis=1)
Index of smallest element in X	np.argmin(X)
Indices of largest elements along first axis	np.argmax(X, axis=0)
Sum of all elements in X	np.sum(X)
Sum of elements along first axis	np.sum(X, axis=0)
Sort matrix (in ascending order) along last axis (axis=-1)	np.sort(D)
Sort matrix along first axis	np.sort(D, axis=0)
Indices for sorting matrix (in ascending order) along last axis (axis=-1)	np.argsort(D)
Indices for sorting matrix along first axis	np.argsort(D, axis=0)

The np.vstack() and np.hstack() functions can be used to add columns or rows to existing matrices.
•	The function np.vstack() performs vertical stacking that concatenates vectors along the first axis
•	The function np.hstack() performs horizontal stacking that concatenates vectors along the second axis.
np.exp(), np.sqrt(), and np.square()

Slicing
1.	Index a single element, such as x1[3].
2.	Slice a portion of the array, such as x1[1:3].
3.	Slice the last two numbers of the array using x1[-2:].

Calc mode
def mode3(a):
    return max(a, key=a.count)

from scipy.stats import mode as modescipy
def modeC(M):
    return modescipy(M)[0]

Add vector v to each column and vector w to each row
•	option: output = M + np.array([v] * 8).T + np.array([w] * 3)
•	option: output = M + np.outer(v, np.ones(8)) + np.outer(np.ones(3), w)
•	option: output = (M.T + v).T + w
•	

change all values of X (here -1) in a matrix (A) to 0
A[A < 0] = 0

 how a vector can be repeated along rows or columns to form a matrix. (here matrix M1(3,8) each column is matrix v and M2(10,3)  each row is matrix v)
Option: 
M1 = np.tile(np.array([v]).T, (1, 8)) 
M2 = np.tile(v, (10, 1))

Option:
M1 = np.repeat(v.reshape((3, 1)), 8, axis=1) 
M2 = np.repeat(v.reshape((1, 3)), 10, axis=0)


Calculate L2 distance and kNN
S contains the X row norms
R contains the Z row norms
G contains all pairwise dot products

n, d1 = X.shape
m, d2 = Z.shape
G = X @ Z.T

# stretch sideways
S = np.repeat(
            np.sum(X**2, axis=1).reshape(-1,1),
            m,
            axis=1)
# stretch downward
R = np.repeat(
        np.sum(Z**2, axis=1).reshape(1,-1),
        n,
        axis=0
    )

D2 = (S + R - 2*G)
    d = np.sqrt(np.maximum(D2,0))
    I = np.argsort(d, axis=0)[:k]
    D = np.sort(d, axis=0)[:k]

Verify Accuracy
    truth = truth.flatten()
    predictions = preds.flatten()
    accur = np.mean(truth == predictions)
def accuracy(truth, preds):
    """
    Analyzes the accuracy of a prediction against the ground truth

    Input:
        truth: n-dimensional vector of true class labels
        preds: n-dimensional vector of predictions

    Output:
        accur: Percent of predictions that are correct (a scalar)
    """
    accur = None
    
    # YOUR CODE HERE
    truth = truth.flatten()
    predictions = preds.flatten()
    accur = np.mean(truth == predictions)
    # END OF YOUR CODE
  
    return accur

def knnclassifier(xTr, yTr, xTe, k):
    """
    Function that returns predictions from k-NN classifier

    Input:
        xTr: nxd input matrix with n row-vectors of dimensionality d
        yTr: n-dimensional vector of labels
        xTe: mxd input matrix with m row-vectors of dimensionality d
        k: Number of nearest neighbors to be found
    Output:
        preds: Predicted labels; i.e., preds[i] is the predicted label of xTe[i, :]
    """
    preds = None
    yTr = yTr.flatten()
    from scipy.stats import mode as calc_mode
    I, D = findknn(xTr, xTe, k)
    neighbors = yTr[I]
    preds = np.zeros(xTe.shape[0])
    for i in range(xTe.shape[0]):
        preds[i] = calc_mode(neighbors[:,i], keepdims=False).mode

    return preds

You can compute the actual classification error on the test set by calling
yPreds = knnclassifier(xTr, yTr, xTe, 3)
accuracy(yTe, yPreds)
MASTER
# Load the face data
xTr, yTr, xTe, yTe = loaddata("faces.mat")

# Make predictions and calculate accuracy
k = 1                                    # Use 1-NN classification
t0 = time.time()                         # Record start time
preds = knnclassifier(xTr, yTr, xTe, k)  # Make predictions
result = accuracy(yTe, preds)            # Compute accuracy
t1 = time.time()                         # Record stop time

print(f"Using {k}-NN classification to perform facial recognition...")
print(
    f"You obtained {100 * result:.2f}% classification",
    f"acccuracy in {t1 - t0:.4f} seconds.",
)

FDR
The transcript explains how facial recognition works at a high level:
•	The system extracts a small set of measurable facial features (a “template”) that uniquely describe a person’s face.
•	These features are based on distinguishing properties, such as ratios between facial measurements (e.g., distance between the eyes compared to nose length).
•	Each person’s face produces a different combination of these values.
Once features are collected for many individuals:
•	The system stores these feature templates in a database.
•	When a new face image is received, it extracts the same types of features.
•	It then compares this new set of features to those in the database.
Finally:
•	The system identifies the person by finding the closest match—essentially using a “nearest neighbor” approach to determine which stored face is most similar.
•	Facial recognition deals with an unusual machine learning setup:
o	Very large number of classes (e.g., millions of people).
o	Very few samples per class (only a few photos per person).
•	This is the opposite of many typical ML problems, where you have:
o	Few classes
o	Many examples per class.
•	Because of this structure, methods like nearest neighbor classification work well:
o	They don’t require a lot of training data per individual.
o	They can directly compare a new face’s features to stored examples.
Bottom line:
Facial recognition is a “many classes, few samples” problem, which makes nearest neighbor approaches particularly effective compared to other ML methods.
Provide your feedback on BizChat
What are the challenges of few-shot learning?How does this affect model scalability?What are alternative methods for facial recognition?
