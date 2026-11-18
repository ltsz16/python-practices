# Estimating Probability Distributions
**Cornell University — Prof. Kilian Weinberger**

---

## Statistics Refresher

### Bayes' Rule

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

- $P(A|B)$: likelihood of event $A$ given $B$
- $P(A)$: likelihood of observing $A$ independently
- $P(B) \neq 0$

**Example**: Two boxes — Box A (1 green, 1 red, 1 orange, 2 blue), Box B (1 green, 4 blue). Pick a random box and draw a blue block. What's the probability it came from Box A?

$$P(\text{Box A}|\text{blue}) = \frac{P(\text{blue}|\text{Box A})P(\text{Box A})}{P(\text{blue})} = \frac{(2/5)(1/2)}{6/10} = \frac{1}{3}$$

### Chain Rule

The joint probability of multiple events:

$$P(A, B, C, D) = P(A)P(B|A)P(C|A,B)P(D|A,B,C)$$

### Distributions

**Normal (Gaussian)**:
$$f(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
- Expectation: $\mu$, Variance: $\sigma^2$
- PDF has maximum at the mean ("bell curve")

**Binomial**:
$$P(x = k; p) = \binom{n}{k} p^k (1-p)^{n-k}$$
- Describes $k$ successes from $n$ independent trials (e.g., coin flips)
- Expectation: $np$, Variance: $np(1-p)$

---

## Module 1: Estimate Distributions from Data

### 1.1 Bayes Optimal Classifier

Data is always drawn from some probability distribution $P(y|\mathbf{x})$. If you know this distribution, you can make optimal predictions.

**Bayes Optimal Classifier** — assigns the most likely label for a given input:

$$\hat{y} = h_{\text{opt}}(\mathbf{x}) = \underset{y}{\arg\max}\ P(y|\mathbf{x})$$

**Error rate** (irreducible error):

$$\epsilon_{\text{BayesOpt}} = 1 - P(h_{\text{opt}}(\mathbf{x})|\mathbf{x}) = 1 - P(\hat{y}|\mathbf{x})$$

- No classifier with the same features can achieve lower error — this is the lower bound
- Called "irreducible error" because it reflects true label ambiguity in the data
- Cannot be used in practice since the true distribution $P(y|\mathbf{x})$ is rarely known

**Example** (spam): If $P(+1|\mathbf{x}) = 0.8$ and $P(-1|\mathbf{x}) = 0.2$, predict spam (+1). Error rate = 0.2.

### 1.2 Maximum Likelihood Estimation (MLE)

Since the true distribution is unknown, estimate it from data. **MLE** finds parameters $\theta$ that make the observed data as likely as possible.

$$\hat{\theta} = \underset{\theta \in [0,1]}{\arg\max}\ P(D|\theta)$$

**Two-step process**:
1. Write the likelihood $P(D|\theta)$
2. Take the log (log-likelihood), then differentiate and set to zero

**Why log?** Products of many small probabilities underflow to zero. Log turns products into sums:
$$\hat{\theta} = \underset{\theta}{\arg\max}\ P(D|\theta) = \underset{\theta}{\arg\max}\ \log(P(D|\theta))$$

### 1.3 MLE for Binomial Distribution (Coin Toss)

Given $n_H$ heads and $n_T$ tails, the likelihood is:

$$P(D|\theta) = \binom{n_H + n_T}{n_H} \theta^{n_H} (1-\theta)^{n_T}$$

**Log-likelihood**:
$$\log(P(D|\theta)) = \log\binom{n_H+n_T}{n_H} + n_H \log(\theta) + n_T \log(1-\theta)$$

**Taking derivative and setting to zero**:
$$\frac{\partial}{\partial\theta}\left[n_H \log(\theta) + n_T \log(1-\theta)\right] = 0$$
$$\Rightarrow \frac{n_H}{\theta} - \frac{n_T}{1-\theta} = 0 \Rightarrow \theta = \frac{n_H}{n_H + n_T}$$

**Result**: MLE estimate is simply the fraction of heads observed. If 6 heads out of 10 tosses, $\hat{\theta} = 0.6$.

### 1.4 Laplace (Plus-One) Smoothing

**Problem with small datasets**: If all $n$ flips come up heads, MLE gives $\hat{\theta} = 1.0$ — assigning zero probability to tails, which is clearly wrong.

**Solution**: Inject "hallucinated" samples based on prior belief before observing data.

$$\hat{\theta}_{\text{smoothed}} = \frac{n_H + s}{n_H + n_T + 2s}$$

Where $s$ is the smoothing parameter:
- $s = 0$: pure MLE
- $s = 1$: Laplace/plus-one smoothing (adds 1 of each outcome)

**Effect**:
- If prior is correct → faster convergence, need less data
- If prior is wrong → biases the estimate, requires more data to overcome

**Example**: 6 heads in 6 tosses. Pure MLE = 100%. With +1 smoothing (1H, 1T hallucinated): $\frac{7}{8} = 87.5\%$ — still high, but no longer zero probability for tails.

> **Rule**: Smoothing helps when the prior is approximately correct. It hurts when the prior is wrong.

---

## Module 2: Classification with Naive Bayes

### 2.1 MLE Curse of Dimensionality

MLE estimates $P(y|\mathbf{x})$ by counting: how many times did we see input $\mathbf{x}$ with label $y$?

$$P(y|\mathbf{x}) = \frac{\sum_{i=1}^{n} I(y_i = y \text{ and } \mathbf{x}_i = \mathbf{x})}{\sum_{i=1}^{n} I(\mathbf{x}_i = \mathbf{x})}$$

**Problem**: In high-dimensional spaces, the same exact $\mathbf{x}$ is almost never seen twice.

| Features ($d$) | Possible feature vectors |
|----------------|--------------------------|
| 2 binary | $2^2 = 4$ |
| 20 binary | $2^{20} = 1{,}048{,}576$ |
| 1,000,000 binary | $2^{1,000,000}$ (more than electrons in the universe) |

Every user/input becomes completely unique. MLE (with or without smoothing) breaks down — we can never estimate $P(y|\mathbf{x})$ from data we never saw.

### 2.2 Naive Bayes Assumption

**Key insight**: Apply Bayes' rule to flip the problem:

$$P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})}$$

- $P(y)$: easy to estimate (just count label frequencies — 1D problem)
- $P(\mathbf{x}|y)$: still high-dimensional and hard...

**Unless** we assume features are **conditionally independent given the label**:

$$\hat{P}(\mathbf{x}|y) = \prod_{\alpha=1}^{d} P(x_\alpha | y)$$

This decomposes one $d$-dimensional problem into $d$ independent 1-dimensional problems — each trivially solved with MLE.

> **Why "naive"?** Real-world features are rarely truly independent. We introduce a known error for the sake of tractability. Sometimes the error is small; sometimes it's large — you must judge case-by-case.

**Example where it holds**: Symptoms of a disease. If you know a patient has Ebola, fever and rash are independently caused by the disease — knowing one doesn't predict the other.

**Example where it fails**: Spam email words. Knowing "sale" is in the email makes "price" more likely, even conditioned on spam.

### 2.3 Naive Bayes Classifier

Full derivation:

$$h(\mathbf{x}) = \underset{y}{\arg\max}\ P(y|\mathbf{x})$$
$$= \underset{y}{\arg\max}\ \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})}$$
$$= \underset{y}{\arg\max}\ P(\mathbf{x}|y)P(y) \quad (P(\mathbf{x}) \text{ doesn't depend on } y)$$
$$= \underset{y}{\arg\max}\ \prod_{\alpha=1}^{d} P(x_\alpha|y) \cdot P(y) \quad \text{(Naive Bayes assumption)}$$
$$= \underset{y}{\arg\max}\ \sum_{\alpha=1}^{d} \left(\log P(x_\alpha|y) + \log P(y)\right) \quad \text{(log is monotonic)}$$

**Why sum logs instead of multiplying probabilities?** Multiplying many small values causes floating-point underflow to zero. Summing logs avoids this without changing which label wins.

### 2.4 Categorical Naive Bayes

When features take discrete values from $K_\alpha$ categories (imagine each feature is a die with $K_\alpha$ sides):

$$P([\mathbf{x}]_\alpha = j | y = c) = [\theta_{jc}]_\alpha \quad \text{such that} \sum_{j=1}^{K_\alpha} [\theta_{jc}]_\alpha = 1$$

**MLE estimate with smoothing parameter $s$**:

$$[\hat{\theta}_{jc}]_\alpha = \frac{\sum_{i=1}^{n} I(y_i = c) I([\mathbf{x}_i]_\alpha = j) + s}{\sum_{i=1}^{n} I(y_i = c) + s K_\alpha}$$

| $s$ value | Estimation technique |
|-----------|---------------------|
| $s = 0$ | MLE (no smoothing) |
| $s > 0$ | MAP (maximum a posteriori) |
| $s = 1$ | Laplace smoothing |

**Class prior** (label probability, independent of features):

$$\hat{\pi}_c = \hat{P}(y = c) = \frac{1}{n} \sum_{i=1}^{n} I(y_i = c)$$

**Full Naive Bayes prediction**:

$$h(\mathbf{x}) = \underset{c}{\arg\max}\ \hat{\pi}_c \prod_{\alpha=1}^{d} [\hat{\theta}_{jc}]_\alpha$$

---

## Module 3: Build a Baby Name Classifier

### 3.1 Problem Setup

Given a baby name, classify its gender (boy = $+1$, girl = $-1$). Apply Naive Bayes in 4 steps:

1. **Feature extraction** — convert each name to a binary feature vector $\mathbf{x} \in \{0,1\}^d$
2. **Class prior** — estimate $P(y = +1)$ and $P(y = -1)$ from training data
3. **Conditional probability** — estimate $P([\mathbf{x}]_\alpha | y)$ for each feature $\alpha$ and label
4. **Predict** — for a test name, compute log-likelihoods and pick the larger one

### 3.2 Feature Extraction via Hashing

Raw names → bag of string features → fixed-length binary vector via hashing.

**Example features for "Natalia"**:
- `PrefixNa` (name starts with "Na-")
- `Endingia` (name ends in "-ia")

**Hash function**: maps each feature string to a dimension index $\in \{0, \ldots, d-1\}$. Set that dimension to 1. Result: sparse binary vector.

```python
# Conceptual process
features = ["PrefixNa", "Endingia"]
x = np.zeros(d)
for f in features:
    x[hash(f) % d] = 1
```

**Example feature vectors** (3 features: first letter 'a', first letter vowel, last letter vowel):

| Name | $[\mathbf{x}]_1$ | $[\mathbf{x}]_2$ | $[\mathbf{x}]_3$ |
|------|-----------------|-----------------|-----------------|
| Alicia | 1 | 1 | 1 |
| Eve | 0 | 1 | 1 |
| Aiden | 1 | 1 | 0 |
| Fred | 0 | 0 | 0 |

### 3.3 Training: NaiveBayesPY

Estimate class priors from training data:

```python
def NaiveBayesPY(y_train):
    # Count fraction of each label
    P_boy  = sum(y == +1 for y in y_train) / len(y_train)
    P_girl = sum(y == -1 for y in y_train) / len(y_train)
    return P_boy, P_girl
```

If 600 boys and 600 girls: $P(y=+1) = P(y=-1) = 0.5$.

### 3.4 Training: NaiveBayesPXY

Estimate conditional probabilities per feature per label (with Laplace smoothing):

$$P([\mathbf{x}]_\alpha = j | y = c) = \frac{\text{\# samples with label } c \text{ having feature } \alpha = j}{\text{\# samples with label } c}$$

**Empirical example** (fraction of names with feature = 1):

| | Feature 1 (starts 'a') | Feature 2 (vowel start) | Feature 3 (vowel end) |
|---|---|---|---|
| Boy ($y=+1$) | 0.09 | 0.17 | 0.30 |
| Girl ($y=-1$) | 0.16 | 0.24 | 0.73 |

### 3.5 Prediction

For binary features, the conditional probability of a test feature value $x_1$:

$$P([\mathbf{x}]_1 | y=1) = x_1 \cdot P([\mathbf{x}]_1=1|y=1) + (1-x_1) \cdot P([\mathbf{x}]_1=0|y=1)$$

**Log-posterior** (Bayes' Rule):

$$\log P(y=c|\mathbf{x}) \propto \log P(y=c) + \sum_{\alpha=1}^{d} \log P([\mathbf{x}]_\alpha | y=c)$$

**Worked example** (with $\log P(y) = \log(0.5) \approx -0.693$ for both):

| Name | log-score (boy) | log-score (girl) | Prediction |
|------|-----------------|------------------|------------|
| Alicia | −6.077 | −4.267 | Girl |
| Eve | −3.763 | −2.609 | Girl |
| Aiden | −5.230 | −5.262 | Boy |
| Fred | −1.330 | −2.451 | Boy |

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Bayes Optimal Classifier | $\hat{y} = \arg\max_y P(y\|\mathbf{x})$ — best possible classifier; sets the error lower bound |
| MLE | Find $\theta$ maximizing $P(D\|\theta)$; use log-likelihood to simplify |
| Binomial MLE | $\hat{\theta} = n_H / (n_H + n_T)$ — fraction of observed successes |
| Laplace smoothing | Add hallucinated samples to prevent zero probabilities on small data |
| MLE curse of dimensionality | In high-$d$ spaces, no two inputs match — MLE fails entirely |
| Naive Bayes assumption | Features are conditionally independent given label; decomposes $d$-D → $d$ × 1-D |
| NB classifier | $h(\mathbf{x}) = \arg\max_y \sum_\alpha \log P(x_\alpha\|y) + \log P(y)$ |
| Categorical NB | Each feature modeled as a categorical die; smoothing via parameter $s$ |
| Feature hashing | Convert text features to fixed-size binary vectors efficiently |
| Always use logs | Multiplying many probabilities underflows; sum logs instead |

---

# Extended Derivations & Python Lab

The sections above cover the concepts. This appendix adds (1) a few extra derivations from the PDF that reward a closer look, and (2) runnable Python code for every module so you can experiment.

## A. Extra Derivations from the PDF

### A.1 Chain rule — worked numerical example

Ten blocks in one box: 2 green, 1 red, 1 orange, 6 blue. Draw without replacement: blue → red → orange → green.

$$P(A, B, C, D) = P(A)\,P(B\mid A)\,P(C\mid A,B)\,P(D\mid A,B,C) = \frac{6}{10}\cdot\frac{1}{9}\cdot\frac{1}{8}\cdot\frac{2}{7} = \frac{1}{420}$$

The chain rule is what powers the Naive Bayes derivation: without any independence assumption, $P(\mathbf{x}\mid y) = \prod_\alpha P(x_\alpha \mid y, x_1, \dots, x_{\alpha-1})$. Naive Bayes drops the conditioning on prior features to get $\prod_\alpha P(x_\alpha \mid y)$.

### A.2 Why the Binomial likelihood table peaks at $\hat\theta = n_H/(n_H+n_T)$

For 6 heads in 10 tosses, the PDF tabulates $P(D\mid\theta)$ at a few values of $\theta$:

| $\theta$ | $P(D\mid\theta) = \binom{10}{6}\theta^6(1-\theta)^4$ |
|----------|------------------------------------------------------|
| 0.5 | $\approx 0.205$ |
| **0.6** | $\approx \mathbf{0.251}$ ← maximum |
| 0.75 | $\approx 0.146$ |
| 1.00 | $0$ (zero tails is impossible under this $\theta$'s tail probability $(1-\theta)^4 = 0$) |

The zero at $\theta = 1$ is exactly why *pure* MLE assigns zero probability to unseen outcomes, and is the motivation for Laplace smoothing.

### A.3 Smoothing as MAP with a Beta prior

Plus-$s$ smoothing on a binomial is equivalent to Maximum A Posteriori (MAP) estimation with a $\mathrm{Beta}(s+1, s+1)$ prior:

$$P(\theta) \propto \theta^{s}(1-\theta)^{s} \quad\Longrightarrow\quad \hat\theta_{\text{MAP}} = \frac{n_H + s}{n_H + n_T + 2s}$$

- $s = 0$ ⇒ uniform prior ⇒ MAP = MLE.
- $s = 1$ ⇒ $\mathrm{Beta}(2,2)$ (a mild bump around $\theta = 0.5$) ⇒ Laplace / "add-one" smoothing.

This is why the PDF labels $s>0$ as the **MAP** estimator: you are secretly injecting a Beta prior on the coin's bias.

### A.4 Formalizing the curse of dimensionality with a Venn diagram

Let $B$ = training samples with $\mathbf{x}_i = \mathbf{x}$ and $C = B \cap \{y_i = y\}$. Then

$$P(y\mid \mathbf{x}) = \frac{|C|}{|B|}$$

In high $d$ *both* $|B|$ and $|C|$ collapse to $0$ almost surely at a fixed test point, so the estimator is $0/0$. This is the pure curse of dimensionality: **you can only estimate probabilities on cells that were actually visited**.

---

## B. Python Lab

Everything here runs on plain NumPy. Copy any block into a REPL or a notebook.

### B.1 Bayes' rule — verify the box example by simulation

```python
import numpy as np

rng = np.random.default_rng(0)

# Box A: 1G, 1R, 1O, 2B ; Box B: 1G, 4B
box_a = ['G','R','O','B','B']
box_b = ['G','B','B','B','B']

trials = 200_000
from_a_given_blue = 0
total_blue = 0
for _ in range(trials):
    box = rng.choice(['A','B'])
    draw = rng.choice(box_a if box == 'A' else box_b)
    if draw == 'B':
        total_blue += 1
        if box == 'A':
            from_a_given_blue += 1

print(f"P(A|blue) ~ {from_a_given_blue/total_blue:.4f}   (analytic: {1/3:.4f})")
```

### B.2 Plot the binomial likelihood (matches the PDF figure)

```python
import numpy as np
from math import comb

def binomial_likelihood(theta, n_H, n_T):
    return comb(n_H + n_T, n_H) * theta**n_H * (1 - theta)**n_T

thetas = np.linspace(0, 1, 501)
lik    = [binomial_likelihood(t, 6, 4) for t in thetas]
theta_hat = thetas[int(np.argmax(lik))]
print(f"argmax theta = {theta_hat:.2f}   L(argmax) = {max(lik):.3f}")
# argmax theta = 0.60   L(argmax) = 0.251   <- matches PDF exactly

# Optional plot
# import matplotlib.pyplot as plt
# plt.plot(thetas, lik); plt.axvline(theta_hat, ls='--'); plt.show()
```

### B.3 Watch smoothing "kick in" as the dataset grows

```python
import numpy as np
rng = np.random.default_rng(42)
true_theta = 0.5
sizes = [2, 5, 10, 50, 500, 5000]

print(f"{'n':>6}  {'MLE':>8}  {'Laplace s=1':>12}  {'Prior s=10':>12}")
for n in sizes:
    flips = rng.binomial(1, true_theta, size=n)
    nH, nT = flips.sum(), n - flips.sum()
    mle   = nH / (nH + nT)
    lap1  = (nH + 1)  / (nH + nT + 2)
    lap10 = (nH + 10) / (nH + nT + 20)
    print(f"{n:>6}  {mle:>8.3f}  {lap1:>12.3f}  {lap10:>12.3f}")
```

Small $n$ ⇒ the smoothed estimates hug 0.5. Large $n$ ⇒ they all converge to the truth.

### B.4 Empirically demonstrate the curse of dimensionality

Draw $n = 1000$ Bernoulli($0.5$) vectors in dimension $d$ and count duplicate collisions with a random test point:

```python
import numpy as np

rng = np.random.default_rng(0)
n = 1000

for d in [2, 5, 10, 20, 50]:
    X    = rng.integers(0, 2, size=(n, d))
    test = rng.integers(0, 2, size=d)
    matches = np.sum(np.all(X == test, axis=1))
    print(f"d={d:>3}  identical training matches to test point: {matches:>5}/{n}")
# d=  2  ~ hundreds of matches
# d= 20  ~ 0 matches   <- MLE has no data to estimate P(y|x)
```

### B.5 Naive Bayes from scratch (Bernoulli/Categorical)

```python
import numpy as np

class BernoulliNaiveBayes:
    """Categorical NB for binary features, with add-s smoothing."""
    def __init__(self, smoothing=1.0):
        self.s = smoothing

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        s = self.s
        # class prior pi_c
        self.log_prior_ = np.log(
            np.array([(y == c).mean() for c in self.classes_])
        )
        # theta_{jc,alpha} for j=1 (feature = 1)
        self.log_p1_ = np.zeros((len(self.classes_), d))
        self.log_p0_ = np.zeros((len(self.classes_), d))
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            n_c = len(Xc)
            p1  = (Xc.sum(axis=0) + s) / (n_c + 2 * s)   # K_alpha = 2
            self.log_p1_[k] = np.log(p1)
            self.log_p0_[k] = np.log(1 - p1)
        return self

    def predict_log_proba(self, X):
        X = np.asarray(X)
        # log P(x|c) = sum_alpha [ x_a * log p1 + (1-x_a) * log p0 ]
        scores = X @ self.log_p1_.T + (1 - X) @ self.log_p0_.T
        return scores + self.log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

# --- Ebola/fever toy example from the PDF ---
# 14 patients: 8 have Ebola (label=1), 6 don't (label=0).
# Among 8 Ebola: 2 have fever. Among 6 non-Ebola: 1 has fever.
X = np.array(
    [[1]]*2 + [[0]]*6      # Ebola:    2 fevers, 6 no-fever
  + [[1]]*1 + [[0]]*5      # No-Ebola: 1 fever,  5 no-fever
)
y = np.array([1]*8 + [0]*6)

model = BernoulliNaiveBayes(smoothing=0).fit(X, y)
print("P(fever | Ebola)   =", np.exp(model.log_p1_[list(model.classes_).index(1)][0]))
print("P(fever | ~Ebola)  =", np.exp(model.log_p1_[list(model.classes_).index(0)][0]))
# 1/4 and 1/6 -- exactly the PDF numbers.
```

### B.6 Naive Bayes on the "superhero" toy dataset (Activity: Naive Bayes in Action)

```python
import numpy as np

# Features: Cape, Mask, Underwear-outside
X = np.array([
    [0, 1, 0],   # Spider-Man    good
    [1, 0, 1],   # Superman      good
    [1, 1, 1],   # Batman        good
    [0, 0, 0],   # Joker         bad
    [1, 0, 0],   # Penguin       bad
    [0, 1, 0],   # Riddler       bad
])
y = np.array([1, 1, 1, 0, 0, 0])   # 1 = Good, 0 = Bad

model = BernoulliNaiveBayes(smoothing=1).fit(X, y)
test_names = ["Cape+Mask+Undies", "No cape, no mask, no undies"]
tests = np.array([[1,1,1], [0,0,0]])
for name, x, pred in zip(test_names, tests, model.predict(tests)):
    print(f"{name:>32}  ->  {'Good' if pred==1 else 'Bad'}")
```

### B.7 Baby-name classifier — end-to-end

Feature hashing + Bernoulli NB reproducing the walkthrough in the PDF.

```python
import numpy as np

def name_features(name):
    """Bag-of-features string extraction (add more for your project)."""
    n = name.lower()
    feats = [
        f"first={n[0]}",
        f"last={n[-1]}",
        f"prefix2={n[:2]}",
        f"suffix2={n[-2:]}",
        f"suffix3={n[-3:]}",
        f"has_a={'a' in n}",
        f"has_y={'y' in n}",
        f"vowels={sum(c in 'aeiouy' for c in n)}",
    ]
    return feats

def hash_features(name, d=1024):
    x = np.zeros(d, dtype=int)
    for f in name_features(name):
        x[hash(f) % d] = 1
    return x

def train_nb(names_pos, names_neg, d=1024, smoothing=1.0):
    X = np.stack([hash_features(n, d) for n in names_pos + names_neg])
    y = np.array([+1] * len(names_pos) + [-1] * len(names_neg))
    model = BernoulliNaiveBayes(smoothing=smoothing).fit(X, y)
    model._d = d
    return model

def predict_name(model, name):
    x = hash_features(name, model._d).reshape(1, -1)
    logp = model.predict_log_proba(x)[0]
    pred = model.classes_[np.argmax(logp)]
    return ("boy" if pred == +1 else "girl"), logp

# Tiny toy training set — expand from your girls.train / boys.train files.
boys  = ["John", "Michael", "Robert", "David", "James", "Aiden", "Fred",
         "William", "Joseph", "Charles"]
girls = ["Mary", "Jennifer", "Linda", "Alicia", "Eve", "Sophia",
         "Emily", "Olivia", "Ava", "Isabella"]

model = train_nb(boys, girls)
for name in ["Alicia", "Eve", "Aiden", "Fred", "Taylor"]:
    label, logp = predict_name(model, name)
    print(f"{name:>10}  ->  {label:<4}   log-scores={logp.round(3)}")
```

### B.8 Sanity-check computing the PDF's baby-name log-scores by hand

The PDF gives conditional probabilities for the 3-feature example:

| Feature | $P(x_\alpha=1\mid \text{boy})$ | $P(x_\alpha=1\mid \text{girl})$ |
|---------|--------------------------------:|--------------------------------:|
| starts 'a' | 0.09 | 0.16 |
| vowel start | 0.17 | 0.24 |
| vowel end | 0.30 | 0.73 |

```python
import numpy as np

p1 = {'boy':  np.array([0.09, 0.17, 0.30]),
      'girl': np.array([0.16, 0.24, 0.73])}
log_prior = np.log(0.5)   # 600 boys, 600 girls

def score(x, cls):
    p = p1[cls] * x + (1 - p1[cls]) * (1 - x)
    return log_prior + np.log(p).sum()

names = {'Alicia': [1,1,1], 'Eve': [0,1,1], 'Aiden': [1,1,0], 'Fred': [0,0,0]}
for name, x in names.items():
    b, g = score(np.array(x), 'boy'), score(np.array(x), 'girl')
    print(f"{name:>7}  boy={b:6.3f}  girl={g:6.3f}  -> {'boy' if b>g else 'girl'}")
# Alicia  boy=-6.077  girl=-4.267  -> girl
# Eve     boy=-3.763  girl=-2.609  -> girl
# Aiden   boy=-5.230  girl=-5.262  -> boy
# Fred    boy=-1.330  girl=-2.451  -> boy
```

Matches the PDF's numbers to three decimals — a good numerical unit test for your own implementation.

---

## C. Common pitfalls (memorize these)

1. **Never multiply probabilities in code — always sum log-probabilities.** Products of hundreds of numbers in $[0,1]$ underflow to $0$ in float64 astonishingly fast.
2. **Never set `smoothing=0` in Naive Bayes on sparse binary features.** One unseen feature/label combination sends $\log P = -\infty$ and poisons the whole prediction. Use `s=1` unless you have a good reason.
3. **Naive Bayes gives you a decision, not a calibrated probability.** Because features aren't really independent, the *arg-max* is often right but the *magnitudes* of $P(y\mid \mathbf{x})$ are usually overconfident (near 0 or 1).
4. **`np.log(0)` warnings are a bug, not a warning.** Clip probabilities to $[\varepsilon, 1-\varepsilon]$ or (better) use `np.logaddexp`.
5. **Feature hashing can collide.** Two different strings can map to the same dimension. In practice this is fine for large `d` (say $2^{18}$) and actually acts as a mild regularizer, but it's worth knowing when debugging.
