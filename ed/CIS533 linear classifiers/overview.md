# Learning With Linear Classifiers

> Structured study notes for Cornell **CIS533: Learning With Linear Classifiers** (Prof. Kilian Weinberger). The raw bullet-note dump is preserved below as **Appendix: Raw Study Notes**.

## Overview

A **linear classifier** predicts the class of an input by checking which side of a hyperplane the point lies on. Everything in this course boils down to the same recipe:

$$h(\mathbf{x}) = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)$$

The four algorithms we study only differ in **how** they choose $\mathbf{w}$ (and $b$):

| Algorithm | Problem | Chooses $\mathbf{w}$ by\u2026 | Loss |
|-----------|---------|--------------------------|------|
| Perceptron | Binary classification | Fixing mistakes one at a time | Mistake-driven (no explicit loss) |
| Linear Regression (OLS) | Regression | Gaussian-noise MLE $\Rightarrow$ closed form | Squared loss |
| Logistic Regression | Binary classification (with probabilities) | Sigmoid MLE $\Rightarrow$ gradient descent | Logistic (cross-entropy) |
| Gradient Descent | *Optimizer* for the losses above | Step opposite the gradient | \u2014 |

---

## Module 1: Linear Classifiers & the Hyperplane

### 1.1 Setup

- Data: $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \{+1, -1\}$.
- A **hyperplane** is the set $\{\mathbf{x} : \mathbf{w}^\top \mathbf{x} + b = 0\}$.
  - $\mathbf{w}$ is **orthogonal** to the hyperplane \u2014 it sets the *direction*.
  - $b$ is the **bias / offset** \u2014 how far from the origin the hyperplane sits.
- **Prediction:** $\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)$.

### 1.2 Two assumptions

1. Labels are **binary** ($+1$ or $-1$).
2. The two classes are **linearly separable** \u2014 some hyperplane cleanly divides them.

> **When is assumption 2 realistic?** In *low* dimensions it is restrictive. In *high* dimensions (text with millions of tokens, image pixels, etc.) it is almost always achievable \u2014 which is why linear classifiers dominate spam filtering, document categorization, and other high-dimensional tasks.

### 1.3 The bias trick \u2014 absorb $b$ into $\mathbf{w}$

Rather than carry a separate offset, append a constant $1$ to every feature vector and append $b$ to the weight vector:

$$\widetilde{\mathbf{w}} = \begin{bmatrix} \mathbf{w} \\ b \end{bmatrix}, \qquad \widetilde{\mathbf{x}} = \begin{bmatrix} \mathbf{x} \\ 1 \end{bmatrix} \qquad\Longrightarrow\qquad \widetilde{\mathbf{x}}^\top \widetilde{\mathbf{w}} = \mathbf{w}^\top \mathbf{x} + b$$

Now every hyperplane passes through the origin in the augmented space, and we can write the entire course in terms of $\mathbf{w}^\top \mathbf{x}$ alone.

```python
import numpy as np

def add_bias(X):
    """Append a column of 1s so w can absorb the bias."""
    return np.hstack([X, np.ones((X.shape[0], 1))])

X = np.array([[1.0, 3.0], [-1.0, 4.0]])
print(add_bias(X))
# [[ 1.  3.  1.]
#  [-1.  4.  1.]]
```

---

## Module 2: The Perceptron

Invented at Cornell by Frank Rosenblatt in **1957** \u2014 arguably the first modern ML algorithm and the direct ancestor of neural networks.

### 2.1 Idea

Start with $\mathbf{w} = \mathbf{0}$. Loop over the data. If a point is misclassified ($y_i (\mathbf{w}^\top \mathbf{x}_i) \le 0$), **rotate the hyperplane toward it**:

$$\mathbf{w} \leftarrow \mathbf{w} + y_i \mathbf{x}_i$$

Repeat until no point is misclassified.

### 2.2 Pseudocode

```
procedure Perceptron(D = {(x_i, y_i)}_{i=1..n})
    w \u2190 0
    repeat
        m \u2190 0                              # misclassification counter
        for (x_i, y_i) in D:
            if y_i (w\u1d40 x_i) \u2264 0:            # misclassified
                w \u2190 w + y_i x_i             # update
                m \u2190 m + 1
    until m == 0                            # (or iter == max)
    return w
```

### 2.3 Clean NumPy implementation

```python
import numpy as np

def perceptron(X, y, max_iter=1000):
    """
    X : (n, d) features (bias already absorbed via add_bias)
    y : (n,)  labels in {+1, -1}
    """
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(max_iter):
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi) <= 0:
                w += yi * xi
                mistakes += 1
        if mistakes == 0:
            return w, True             # converged
    return w, False                    # hit iteration cap

# Toy example \u2014 the exercise from the PDF
X = np.array([[ 1.0, 3.0],
              [-1.0, 4.0]])
y = np.array([+1, -1])
w, ok = perceptron(np.hstack([X, np.ones((2, 1))]), y)
print("w=", w[:-1], "b=", w[-1], "converged=", ok)
```

### 2.4 Worked update by hand (from the PDF)

Positive point $\mathbf{x}_1 = (1,3)$, $y_1 = +1$; negative point $\mathbf{x}_2 = (-1,4)$, $y_2 = -1$; $\mathbf{w}_0 = \mathbf{0}$.

| iter | $\mathbf{w}_t$ | example | $y_i(\mathbf{w}_t^\top \mathbf{x}_i)$ | mis? | $\mathbf{w}_{t+1}$ |
|------|----------------|---------|----------------------------------------|------|--------------------|
| 1 | $(0,0)$ | $\mathbf{x}_1,+1$ | $0$ | \u2705 | $(0,0)+(1,3) = (1,3)$ |
| 2 | $(1,3)$ | $\mathbf{x}_2,-1$ | $-11$ | \u2705 | $(1,3)-(-1,4) = (2,-1)$ |
| 3 | $(2,-1)$ | $\mathbf{x}_1,+1$ | $-1$ | \u2705 | $(3,2)$ |
| 4 | $(3,2)$ | $\mathbf{x}_2,-1$ | $-5$ | \u2705 | $(4,-2)$ |
| 5 | $(4,-2)$ | $\mathbf{x}_1,+1$ | $-2$ | \u2705 | $(5,1)$ |
| 6 | $(5,1)$ | $\mathbf{x}_2,-1$ | $1$ | \u274c | \u2014 |
| 7 | $(5,1)$ | $\mathbf{x}_1,+1$ | $8$ | \u274c | \u2014 |

Five updates, then convergence. The final $\mathbf{w}^{*}$ can be reconstructed as $\mathbf{w}^{*} = \sum_{(\mathbf{x}_i, y_i)\in M} y_i \mathbf{x}_i$ where $M$ is the multiset of mistakes \u2014 useful when you have a log of *which* examples caused updates but lost the final weights.

### 2.5 Convergence guarantee (informal proof)

Assume every $\|\mathbf{x}_i\| \le 1$ and a true separator $\mathbf{w}^{*}$ with $\|\mathbf{w}^{*}\| = 1$ exists. Define the **margin**

$$\gamma = \min_{(\mathbf{x}_i, y_i)\in D} \lvert \mathbf{x}_i^\top \mathbf{w}^{*} \rvert > 0$$

After $M$ updates:

1. **Alignment grows linearly:** $\mathbf{w}^\top \mathbf{w}^{*} \ge M \gamma$
   *(each update adds at least $\gamma$ because the mistaken point is at least $\gamma$ deep on the correct side of $\mathbf{w}^{*}$.)*
2. **Norm grows at most root-linearly:** $\mathbf{w}^\top \mathbf{w} \le M$
   *(each update adds $2 y_i \mathbf{w}^\top \mathbf{x}_i < 0$ plus $y_i^2 \|\mathbf{x}_i\|^2 \le 1$.)*

Combining via $\mathbf{w}^\top \mathbf{w}^{*} = \|\mathbf{w}\|\,\|\mathbf{w}^{*}\|\cos\theta \le \|\mathbf{w}\| \le \sqrt{M}$:

$$M\gamma \le \sqrt{M} \quad\Longrightarrow\quad \boxed{\,M \le \tfrac{1}{\gamma^2}\,}$$

**Intuition:** wider margin $\Rightarrow$ faster convergence. Tiny margin $\Rightarrow$ many updates. If no separator exists ($\gamma = 0$), the perceptron loops forever.

### 2.6 Limitations

- **Not linearly separable? Infinite loop.** No noise tolerance.
- Finds *some* separator, not the *best* one \u2014 the answer depends on iteration order.
- No probability output; just a hard $\pm 1$.

These three shortcomings motivate the next three modules (regression \u2192 probabilistic \u2192 optimize).

---

## Module 3: Linear Regression (OLS)

### 3.1 Modeling assumption

Labels are a **linear function** of the features plus **Gaussian noise**:

$$y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$

This gives us the conditional distribution

$$P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(\mathbf{w}^\top \mathbf{x}_i - y_i)^2}{2\sigma^2}\right)$$

### 3.2 Maximum-likelihood derivation of squared loss

$$\begin{aligned}
\mathbf{w}_{\text{MLE}} &= \arg\max_{\mathbf{w}} \prod_{i=1}^n P(y_i \mid \mathbf{x}_i; \mathbf{w}) \\
                        &= \arg\max_{\mathbf{w}} \sum_{i=1}^n \log P(y_i \mid \mathbf{x}_i; \mathbf{w})           && \text{log turns product $\to$ sum} \\
                        &= \arg\max_{\mathbf{w}} \sum_i \Bigl[\log \tfrac{1}{\sqrt{2\pi\sigma^2}} - \tfrac{(\mathbf{w}^\top \mathbf{x}_i - y_i)^2}{2\sigma^2}\Bigr] \\
                        &= \arg\min_{\mathbf{w}} \frac{1}{n}\sum_{i=1}^n (\mathbf{w}^\top \mathbf{x}_i - y_i)^2 && \text{drop constants, flip sign, average}
\end{aligned}$$

Gaussian noise assumption $\Longleftrightarrow$ mean-squared-error (MSE) loss. Not a coincidence \u2014 different noise assumptions give different losses (e.g. Laplace noise $\to$ absolute loss).

**Physical intuition:** treat each data point as a bead connected to the line by a spring; MSE finds the line with minimum total spring energy.

### 3.3 Closed form via matrix calculus

Stack rows into $\mathbf{X} \in \mathbb{R}^{n\times d}$ and labels into $\mathbf{y} \in \mathbb{R}^n$:

$$\mathcal{L}(\mathbf{w}) = \tfrac{1}{n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = \tfrac{1}{n}\bigl(\mathbf{w}^\top \mathbf{X}^\top \mathbf{X}\mathbf{w} - 2\mathbf{w}^\top \mathbf{X}^\top \mathbf{y} + \mathbf{y}^\top \mathbf{y}\bigr)$$

Set the gradient to zero:

$$\nabla_{\mathbf{w}} \mathcal{L} = \tfrac{2}{n}\bigl(\mathbf{X}^\top \mathbf{X}\mathbf{w} - \mathbf{X}^\top \mathbf{y}\bigr) = 0 \quad\Longrightarrow\quad \boxed{\,\mathbf{w}_{\text{MLE}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}\,}$$

This is the **normal equation**. It exists when $\mathbf{X}^\top \mathbf{X}$ is invertible (i.e., columns of $\mathbf{X}$ are linearly independent).

### 3.4 Python \u2014 closed form vs. `lstsq`

```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 200, 3
X = rng.standard_normal((n, d))
true_w = np.array([2.5, -1.0, 0.7])
y = X @ true_w + rng.normal(0, 0.5, size=n)     # Gaussian noise

# 1. Explicit normal equation
w_ne = np.linalg.inv(X.T @ X) @ X.T @ y

# 2. Numerically safer: solve the linear system
w_solve = np.linalg.solve(X.T @ X, X.T @ y)

# 3. Best in practice: lstsq (uses SVD, handles rank deficiency)
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)

print(np.round(np.stack([w_ne, w_solve, w_lstsq, true_w]), 3))
```

> **Practical tip:** never invert $\mathbf{X}^\top \mathbf{X}$ in production code \u2014 use `np.linalg.lstsq` or `scipy.linalg.lstsq`. They handle rank-deficient / ill-conditioned problems gracefully.

---

## Module 4: Logistic Regression

### 4.1 From regression to classification via the sigmoid

Linear regression outputs any real number; classification wants a **probability**. Squash the linear score through the **sigmoid**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$$P(y = +1 \mid \mathbf{x}; \mathbf{w}) = \sigma(\mathbf{w}^\top \mathbf{x}), \qquad P(y = -1 \mid \mathbf{x}; \mathbf{w}) = \sigma(-\mathbf{w}^\top \mathbf{x})$$

With the sign convention $y \in \{+1,-1\}$ both cases fuse into one clean expression:

$$P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \sigma(y_i\,\mathbf{w}^\top \mathbf{x}_i) = \frac{1}{1 + \exp(-y_i \mathbf{w}^\top \mathbf{x}_i)}$$

### 4.2 Two elegant sigmoid identities (needed for the gradient)

1. $\sigma(-z) = 1 - \sigma(z)$ \u2014 guarantees $P(+1\mid\mathbf{x}) + P(-1\mid\mathbf{x}) = 1$.
2. $\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$ \u2014 derivative expressible in the sigmoid itself.

**Proof of (2)** \u2014 quick chain-rule exercise:

$$\sigma'(z) = -\frac{1}{(1 + e^{-z})^2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1 - \sigma(z))$$

### 4.3 MLE $\to$ logistic loss (cross-entropy)

$$\mathbf{w}_{\text{MLE}} = \arg\min_{\mathbf{w}} \sum_{i=1}^n \log\!\bigl(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}_i}\bigr)$$

This function is **convex** but has **no closed form**. It behaves like a rounded-bottom bowl: quadratic near the optimum, roughly linear (hinge-like) far from it. Convex + no closed form \u2014 the classic setup for gradient descent.

### 4.4 Gradient of the logistic loss

Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$ and $1 - \sigma(z) = \sigma(-z)$:

$$\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = -\sum_{i=1}^n \sigma(-y_i \mathbf{w}^\top \mathbf{x}_i)\, y_i \mathbf{x}_i$$

**Interpretation:** the weight of example $i$ in the gradient is $\sigma(-y_i \mathbf{w}^\top \mathbf{x}_i)$ \u2014 the model's *current probability of getting it wrong*. Confidently-correct points barely move $\mathbf{w}$; near-misses and misclassifications dominate. This is why logistic regression is **robust to easy points** and adaptive to the hard ones \u2014 unlike the perceptron, which treats every mistake identically.

### 4.5 Python \u2014 logistic regression from scratch

```python
import numpy as np

def sigmoid(z):
    # Numerically stable sigmoid \u2014 avoids overflow in exp(-z) for large |z|
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def logistic_loss(X, y, w):
    z = y * (X @ w)
    return np.mean(np.logaddexp(0.0, -z))     # log(1 + exp(-z)) \u2014 numerically stable

def logistic_grad(X, y, w):
    z = y * (X @ w)
    return -(X.T @ (sigmoid(-z) * y)) / len(y)

def train_logistic(X, y, lr=0.1, epochs=2000):
    w = np.zeros(X.shape[1])
    for t in range(1, epochs + 1):
        w -= lr * logistic_grad(X, y, w)
    return w

# Toy 2-D data \u2014 labels in {+1, -1}
rng = np.random.default_rng(0)
Xpos = rng.normal(loc=+1.5, size=(50, 2))
Xneg = rng.normal(loc=-1.5, size=(50, 2))
X = np.vstack([Xpos, Xneg])
y = np.concatenate([np.ones(50), -np.ones(50)])
X_aug = np.hstack([X, np.ones((100, 1))])     # bias trick

w = train_logistic(X_aug, y)
acc = np.mean(np.sign(X_aug @ w) == y)
print(f"train accuracy = {acc:.2%}   loss = {logistic_loss(X_aug, y, w):.4f}")
```

### 4.6 Generative vs. discriminative \u2014 Naive Bayes vs. Logistic Regression

Both end up with a **sigmoidal** conditional $P(y\mid\mathbf{x})$, but they arrive there differently:

| Aspect | Naive Bayes (generative) | Logistic Regression (discriminative) |
|--------|---------------------------|---------------------------------------|
| Models | $P(\mathbf{x}\mid y)$ then Bayes-flips to $P(y\mid\mathbf{x})$ | $P(y\mid\mathbf{x})$ directly |
| Training | Counting / closed form | Gradient descent on log loss |
| Assumptions | Strong (feature independence, distribution family) | Weak (linear decision boundary only) |
| Best when | Very little data, assumptions hold | Lots of data, arbitrary feature distributions |
| Handles outliers | Poorly \u2014 they distort the fitted Gaussians | Well \u2014 loss re-weights hard points |

**Rule of thumb:** small data + trustworthy modeling \u2192 Naive Bayes. Big data + \u201cwhatever\u201d features \u2192 Logistic Regression.

---

## Module 5: Gradient Descent

### 5.1 The update rule

For a differentiable loss $\mathcal{L}(\mathbf{w})$:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla \mathcal{L}(\mathbf{w})$$

- $\nabla \mathcal{L}$ points in the direction of **steepest ascent** \u2014 subtract it to go **downhill**.
- $\alpha > 0$ is the **learning rate** (step size).

### 5.2 Why does it work? \u2014 Taylor argument

Near $\mathbf{w}$, a first-order Taylor expansion gives

$$\mathcal{L}(\mathbf{w} + \mathbf{s}) \approx \mathcal{L}(\mathbf{w}) + \nabla \mathcal{L}(\mathbf{w})^\top \mathbf{s}$$

Choosing $\mathbf{s} = -\alpha \nabla \mathcal{L}(\mathbf{w})$:

$$\mathcal{L}(\mathbf{w} - \alpha \nabla \mathcal{L}) \approx \mathcal{L}(\mathbf{w}) - \alpha\,\underbrace{\|\nabla \mathcal{L}\|^2}_{\ge 0} \;<\; \mathcal{L}(\mathbf{w})$$

So as long as $\alpha$ is small enough for the Taylor approximation to hold, **loss strictly decreases** each step. On convex losses (linear/logistic regression), local minima *are* global minima and gradient descent converges to the optimum.

### 5.3 Choosing the learning rate

| $\alpha$ too small | $\alpha$ just right | $\alpha$ too large |
|---------------------|---------------------|---------------------|
| Converges, but painfully slowly | Steady, monotone descent | Zig-zags across the bowl \u2014 or **diverges** entirely |

A safe schedule: $\alpha_t = t_0 / t$ (learning rate decays with each step). Large early steps for coarse progress, small later steps for fine-tuning.

### 5.4 Applied to the losses in this course

| Loss | Gradient step $\mathbf{w} \leftarrow \mathbf{w} - \alpha\,\nabla\mathcal{L}(\mathbf{w})$ |
|------|------------------------------------------------------------------------------------------|
| MSE (linear reg.) | $\mathbf{w} - \alpha\cdot\tfrac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ |
| Logistic | $\mathbf{w} + \alpha\cdot\tfrac{1}{n}\mathbf{X}^\top\bigl(\sigma(-y\odot \mathbf{X}\mathbf{w}) \odot \mathbf{y}\bigr)$ |

### 5.5 Toy example \u2014 minimize $f(w) = w^2$

```python
import numpy as np

def f(w):     return w ** 2
def grad(w):  return 2 * w

w = 10.0
history = [w]
for t in range(1, 26):
    lr = 0.1                  # try 0.001 (slow), 0.1 (good), 1.1 (diverges)
    w = w - lr * grad(w)
    history.append(w)

print([round(h, 4) for h in history[:6]], "\u2026", round(history[-1], 6))
# [10.0, 8.0, 6.4, 5.12, 4.096, 3.2768] \u2026 0.000472
```

### 5.6 Sanity-check gradients with finite differences

Any time you derive a gradient by hand, verify numerically:

```python
def numerical_grad(f, w, eps=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = eps
        g[i] = (f(w + e) - f(w - e)) / (2 * eps)
    return g

# Check logistic_grad against a numerical estimate
w = np.random.default_rng(1).standard_normal(X_aug.shape[1])
g_analytic = logistic_grad(X_aug, y, w)
g_numeric  = numerical_grad(lambda ww: logistic_loss(X_aug, y, ww), w)
print("max abs diff:", np.max(np.abs(g_analytic - g_numeric)))   # ~1e-8
```

If the max difference is not tiny (\u226410\u207b\u2075 or so), your analytic gradient is wrong.

---

## Cheat sheet \u2014 pick the right tool

```
Do you want a probability?               \u2500\u252c\u2500 yes \u2192 Logistic Regression
                                          \u2514\u2500 no  \u2192 continue
Is the target continuous (a number)?     \u2500\u252c\u2500 yes \u2192 Linear Regression (OLS)
                                          \u2514\u2500 no  \u2192 continue
Is the data linearly separable
and noise-free?                          \u2500\u252c\u2500 yes \u2192 Perceptron (fast, guaranteed)
                                          \u2514\u2500 no  \u2192 Logistic Regression + regularization
```

**One-line summaries you can quote:**

- *Perceptron:* fix each mistake by nudging $\mathbf{w}$ toward it. Converges in $\le 1/\gamma^2$ updates if a margin $\gamma$ exists.
- *Linear regression:* Gaussian-noise MLE $\Rightarrow$ MSE loss $\Rightarrow$ closed form $\mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$.
- *Logistic regression:* sigmoid MLE $\Rightarrow$ log-loss (convex, no closed form) $\Rightarrow$ minimize with gradient descent.
- *Gradient descent:* $\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla \mathcal{L}$; Taylor guarantees descent when $\alpha$ is small.

---

# Appendix: Raw Study Notes

> Original bullet-point brain-dump, preserved verbatim.

Linear Classifiers

    Works with binary labels where data points of different classes can be separated by a hyperplane
Perceptron
    Method to adjust the hyperplane
    The hyperplane is initially set to w = 0
    Fix mistakes aggressively
    Asks was label classified correctly?  Yes=do nothing.  No=move line a little
    w(weights) = w  +  (learning rate, usually 1) * y (label, +1, -1) * x (feature vector) 
    perceptron update: w = w + 1 * x * y
    Perceptron:  
    def perceptron(xs, ys):
        """
        function w=perceptron(xs,ys);
    
        Returns the weight vector learned by the Perceptron classifier.
    
        Input:
            xs: n input vectors of d dimensions (nxd matrix)
            ys: n labels (-1 or +1)
    
        Output:
            w: weight vector (d)
            b: bias term
        """
        n, d = xs.shape  # So we have n input vectors with d dimensions
        w = np.zeros(d)
        b = 0.0
    
        # YOUR CODE HERE
        while True:
            cnt = 0
            for x, y in zip(xs,ys):
                resp = np.dot(w,x) + b
                if y * resp <=0:
                    w = w + y * x
                    b = b + y
                    cnt += 1
            if cnt == 0:
                break
        # END OF YOUR CODE
    
        return (w, b)
    The perceptron is arguably the first algorithm to have a strong formal guarantee: If a data set is linearly separable (i.e., the two classes can be perfectly split using a hyperplane in -dimensional space), the perceptron will find a separating hyperplane in a finite number of iterations. If the data is not linearly separable, it will loop forever.
    it doesn't work if this isn't the case, i.e. one cannot place a linear line between the classes.  It is also very sensitive to noisy data
    The hyperplane is iteratively adjusted (or "tilted") using one misclassified point at a time.
    The final hyperplane perfectly separates the two classes
    Not all data sets that are linearly separable can be separated with a hyperplane that passes through the origin. For the perceptron to converge in this scenario, you must account for an offset value (or bias) that will adjust the position of your hyperplane in either a positive or negative direction.  Can update bias term (b) (make 2D become 3D)
Linear Regression
    Linear regression, also often referred to as ordinary least squares (OLS), is a model similar to the perceptron classifier; however, instead of classification, it can be used for regression.
    models the relationship between your labels (y) and the features (x) as a linear function
    Gradually increase probability of correct labels
    Switching to logistic regression = replace sign() with sigmoid + replace update rule with gradient descent on MLE loss
MLE
Comes in when you ask: “Can I turn this classification problem into a probabilistic model?”
If you want MLE:
    Switch to logistic regression (if currently perceptron)
    Now you're modeling: 
    probability of labels
    uncertainty
    noise

Model Assumption
    Data is generated from a linear function (line or hyperplane) plus Gaussian noise
    Each observed point (x_iⓜ,y_i )is: 
    Centered around the true linear model
    With noise distributed as a Gaussian (normal distribution)
Goal
    Find weights w(the slope/hyperplane) that best explain the data
MLE Idea
    Maximize the probability of observing the data:
∏_i▒〖P(〗 y_i∣x_i,w)
    This means choosing wso that the observed data is most likely
Log Trick
    Products of probabilities are hard → take log
    Log turns: 
    Product → Sum
∑_i▒〖log⁡P(〗 y_i∣x_i,w)

Gaussian Likelihood
    Each P(y_i∣x_i,w)is Gaussian:
∝exp⁡(ⓜ-(w^T x_i-y_i )^2/(2σ^2 ))
Simplification
    Log of exponential cancels:
log⁡(e^something)=something
    Constant terms (not depending on w) are dropped
Result
    Maximizing likelihood becomes: 
    Minimizing:
∑_i▒( w^T x_i-y_i )^2

Interpretation
    This is the Mean Squared Error (MSE) loss
    You are minimizing the squared differences between: 
    prediction w^T x_i
    actual label y_i
Geometric View
    Errors are vertical distances from points to line
    Minimize sum of squared distances
Physical Analogy
    Points connected to line by springs
    Find wthat minimizes total tension
Python example
# setup fake data
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
X = np.linspace(0, 10, 50)
true_w = 2.5
noise = np.random.normal(0, 2, size=X.shape)

y = true_w * X + noise
# prediction function
def predict(X, w):
return w @ X
# MLE gives us
def mse_loss(X, y, w):
y_pred = predict(X, w)
return np.mean((y_pred - y) ** 2)

# find the best w
w_values = np.linspace(0, 5, 100)
losses = [mse_loss(X, y, w) for w in w_values]

best_w = w_values[np.argmin(losses)]
print("Best w:", best_w)
# visualize
plt.scatter(X, y, label="Data")
plt.plot(X, predict(X, best_w), color="red", label="Fitted line")
plt.legend()
plt.show()

LOSS FUNCTION
    It’s the mean of (dot product of weights and features − label)²
    Single prediction
    y_pred = np.dot(w, x) # w^T x
    error = y_pred - y
    loss = error ** 2
    Whole data set
    def mse_loss(X, y, w):
    y_pred = X @ w # vector of predictions
    return np.mean((y_pred - y) ** 2)

Logistic Regression
    Logistic regression is a linear classifier with modeling assumptions related to the multinomial Naive Bayes classifier. 
    While the name 'Logistic Regression' suggests that it solves a regression problem, Logistic Regression actually solves a classification problem -- it only allows you to estimate the probability of a categorical label, which must not be continuous.
     Linear Regression and Logistic Regression are not related to each other. 
    classification algorithm that predicts: P(y=1∣x).  not just a label, but a probability.
    Model: P(y=1∣x)=σ(wTx)
    Where:
    w^T x= linear score
    σ(⋅)= sigmoid function
    if P(y=1 | x) >= 0.5:
    predict 1
    else:
    predict 0
    
    Compute a linear combination of features 
 Pass it through a squashing function (sigmoid) 
 Output becomes a probability

🔷 Logistic Regression (Discriminative Model)
Discriminative model (Logistic Regression): models how to separate classes
Logistic regression says:
“Given this input x, what’s the probability it belongs to class 1?”
It doesn’t try to model:
    how x was created
    or what features look like per class

👉 Directly learns P(y | x)
👉 Uses sigmoid + gradient descent
import numpy as np

# Sigmoid function: converts score -> probability
def sigmoid(z):
return 1 / (1 + np.exp(-z))

# Compute predictions (probabilities)
def predict_proba_lr(X, w):
return sigmoid(X @ w) # linear model + sigmoid

# Convert probability to class (0 or 1)
def predict_lr(X, w):
probs = predict_proba_lr(X, w)
return (probs >= 0.5).astype(int)

# Cross-entropy loss (MLE objective)
def loss_lr(X, y, w):
z = X @ w
return -np.mean(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))

# Gradient of loss
def gradient_lr(X, y, w):
preds = sigmoid(X @ w)
return X.T @ (preds - y) / len(y)

# Train using gradient descent
def train_lr(X, y, lr=0.1, epochs=1000):
w = np.zeros(X.shape[1]) # initialize weights

for _ in range(epochs):
grad = gradient_lr(X, y, w)
w -= lr * grad # update step

return w


Sigmoid Function
def sigmoid(z):
return 1 / (1 + np.exp(-z))

It converts:
Input z	Output
very negative	≈ 0
0	0.5
very positive	≈ 1
👉 So it maps any real number → probability (0 to 1)

Multinomial Naive Bayes
Instead of modeling:
P(y∣x)

it models:
P(x∣y)"and" P(y)

This is called a generative model.
    Features are independent
    Used for:
    Text classification
    Word counts

# naive bayes prediction (conceptual)
score = np.log(P_y)

for j in features:
score += x[j] * np.log(P_x_given_y[j])

# choose class with highest score

Multinomial Naive Bayes (Generative Model)
Generative model (Naive Bayes): models how the data is produced

👉 Learns P(x | y) and P(y)
👉 Uses counts + probabilities (no gradient descent)
This version assumes:
    Inputs are counts (like word counts)
    Often used in text classification

import numpy as np

class MultinomialNB:
def __init__(self):
self.class_log_prior = None # log P(y)
self.feature_log_prob = None # log P(x_j | y)

def fit(self, X, y):
n_samples, n_features = X.shape
n_classes = len(np.unique(y))

# Initialize storage
self.class_log_prior = np.zeros(n_classes)
self.feature_log_prob = np.zeros((n_classes, n_features))

# Loop over classes
for c in range(n_classes):
X_c = X[y == c] # all samples in class c

# Compute P(y=c)
self.class_log_prior[c] = np.log(len(X_c) / n_samples)

# Count features (Laplace smoothing added)
feature_counts = X_c.sum(axis=0) + 1
total_count = feature_counts.sum()

# Compute P(x_j | y=c)
self.feature_log_prob[c] = np.log(feature_counts / total_count)

def predict(self, X):
# Compute scores for each class
scores = []

for c in range(len(self.class_log_prior)):
# log P(y=c) + sum_j x_j * log P(x_j | y=c)
score = self.class_log_prior[c] + X @ self.feature_log_prob[c]
scores.append(score)

scores = np.vstack(scores).T # shape: (n_samples, n_classes)

return np.argmax(scores, axis=1) # pick highest probability
``


COMPARRISON
Step	Logistic Regression	Naive Bayes
Model	X @ w	probabilities
Training	gradient descent	counting
Output	sigmoid(probability)	log-probabilities
Data type	numeric features	counts (typically)

Use Case	Model
Small data + text	Naive Bayes ✅
More data, fewer assumptions	Logistic Regression ✅
Want probabilities	Both
Want flexibility	Logistic Regression

Question	Naive Bayes	Logistic Regression
How is data generated?	✅ models it	❌ ignores it
How to classify?	via Bayes rule	directly predicts
Learns	distributions	

Logistic Regression:
    Learns wdirectly by minimizing loss (MLE)
    Flexible, fewer assumptions
    “Let’s directly learn a function that outputs probability.”
    “Let me directly learn weights that best predict labels.”
    
Naive Bayes:
    Computes windirectly from probabilities
    Strong independence assumptions
    “Let’s model how data is generated, then convert to probability.”
    “Let me estimate how features are distributed per class, then infer the label.”
    
Sigmoid
“Convert scores into valid probabilities.”
Logistic regression uses sigmoid to turn a linear model into probabilities, while multinomial Naive Bayes builds those probabilities from feature distributions—but both end up with similar decision boundaries.

⚖️ Trade-offs
Naive Bayes (Generative)
✅ Works well with small data
✅ Fast, simple
❌ Strong independence assumptions

Logistic Regression (Discriminative)
✅ More flexible
✅ Often better accuracy
❌ Needs more data
Think of it like this:

🧠 Generative (Naive Bayes)
class -> generates data
   y   ->   x

🎯 Discriminative (Logistic Regression)
data -> predicts class
  x  ->   y

Gradient Descent
    doesn't simply ask if the prediction was wrong but "how wrong?"
    Gradient descent repeatedly adjusts weights in the direction that reduces error the fastest.
    process tries to find the lowest loss (w = w - a * gradient where a is the learning rate, or how big the step should be).   Gradient Descent can work even when data overlaps and can use probability instead of a simple binary answer.  Some of its downsides are that it can be slower than perceptrons and one must choose the learning rate (too small can take an incredibly long time and too big it might bounce around an never find the lowest point). 
    Gradient descent is a method to minimize a function (loss) by repeatedly stepping in the direction of steepest decrease.
    Linear regression → minimize MSE
    Logistic regression → minimize cross-entropy
    But how do we actually find the best w?
    Ecco gradient descent
Imagine:
    You’re standing on a mountain
    You want to go down to the lowest point (minimum loss)
What do you do?
👉 Look at the slope and step downward.
You have some loss function:
L(w)

Gradient descent updates:
w=w-η⋅∇L(w)

Where:
    ∇L(w): gradient (slope)
    η: learning rate (step size)

🔑 Why subtract the gradient?
    Gradient points uphill
    You want to go downhill
So:
"new " w="old " w-"step"

example
import numpy as np

def f(w):
return w**2

def grad_f(w):
return 2*w # derivative of w^2

w = 10 # start far from minimum
lr = 0.1 # learning rate

for i in range(20):
w = w - lr * grad_f(w)
print(f"Step {i}, w = {w}")

This converges toward: w = 0

APPLY TO LINEAR REGRESSION
def gradient(X, y, w):
n = len(y)
return (2/n) * X.T @ (X @ w - y)

def train_linear_regression(X, y, lr=0.01, epochs=1000):
w = np.zeros(X.shape[1])

for _ in range(epochs):
grad = gradient(X, y, w)
w -= lr * grad # gradient descent step

return w

Applying to Logistic Regression
Loss from MLE
def sigmoid(z):
return 1 / (1 + np.exp(-z))

def gradient_logistic(X, y, w):
preds = sigmoid(X @ w)
return X.T @ (preds - y) / len(y)

def train_logistic(X, y, lr=0.1, epochs=1000):
w = np.zeros(X.shape[1])

for _ in range(epochs):
grad = gradient_logistic(X, y, w)
w -= lr * grad

return w


🔗 Connecting to Everything You've Learned
You already know:
MLE →
gives loss
Linear regression →
MSE
Logistic regression →
cross-entropy

Gradient Descent →
👉 actually minimizes those losses

🧠 Full Pipeline
Assumption → (MLE) → Loss → (Gradient Descent) → Weights
Example:
Gaussian noise → MSE → Gradient descent → Linear regression solution

🔥 Key Insight
👉 Gradient descent is not model-specific
It works for:
    linear regression
    logistic regression
    neural networks
    almost everything in ML

Step	Linear Regression	Logistic Regression	Perceptron
Problem	Predict number	Predict probability	Separate classes
Output	Continuous	Probability + class	Class
Model	w^T x	σ(w^T x)	sign(w^T x)
Loss	MSE	Cross-entropy (MLE)	Mistake-driven
Training	Closed form / GD	Gradient descent	Updates on errors


