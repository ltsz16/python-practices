# Understanding the Expected Label in Machine Learning

This is one of the most important ideas in machine learning. The
**expected label** is what lets us separate **noise** (randomness in the
data) from what a model can actually learn.

Let's build it from the ground up.

------------------------------------------------------------------------

# Step 1. Why do we need an expected label?

Suppose you're predicting house prices.

Your input features are

\[ x = (`\text{bedrooms}`{=tex},; `\text{square feet}`{=tex}) \]

Imagine two houses that are **identical**:

-   3 bedrooms
-   2000 sq ft

One sold for

\[ \$400,000 \]

Another sold for

\[ \$430,000 \]

Why?

Maybe

-   better landscaping
-   motivated seller
-   nicer neighbors
-   timing
-   negotiation
-   pure luck

These aren't included in your features.

So **the same input x does not always produce the same y.**

Instead,

\[ y \]

is a **random variable**.

------------------------------------------------------------------------

# Step 2. The conditional distribution

Instead of saying

> "A 3-bedroom house costs \$420,000"

we say

> "Given these features, there is a probability distribution over
> possible prices."

Mathematically

\[ P(y\|x) \]

means

> the probability of seeing label (y) given features (x).

For our example,

  Price     Probability
  --------- -------------
  400,000   0.2
  410,000   0.3
  420,000   0.3
  430,000   0.2

This is the conditional distribution.

------------------------------------------------------------------------

# Step 3. The expected label

The expected label is simply the **average** of that distribution.

The notation is

\[ `\bar `{=tex}y(x) \]

(read as "y bar of x")

The definition is

\[ `\bar `{=tex}y(x)=E\_{P(y\|x)}\[y\] \]

This means

> Take the expectation (average) of all possible labels according to
> their probabilities.

------------------------------------------------------------------------

# Step 4. The integral

For continuous outputs,

\[ `\boxed{\bar y(x)=\int yP(y|x)\,dy}`{=tex} \]

This is exactly the definition of expectation.

Every possible label contributes

\[ `\text{value}`{=tex} `\times `{=tex}`\text{probability}`{=tex} \]

to the average.

------------------------------------------------------------------------

# Step 5. Why multiply by probability?

Suppose

  Price   Probability
  ------- -------------
  400k    0.2
  410k    0.3
  420k    0.3
  430k    0.2

The expected label is

\[ 400(0.2)+410(0.3)+420(0.3)+430(0.2)=415 \]

Therefore

\[ `\boxed{\bar y(x)=415,000}`{=tex} \]

Notice **415,000 may never actually occur.** It is simply the center of
the distribution.

------------------------------------------------------------------------

# Step 6. Discrete vs Continuous

For discrete labels,

\[ `\boxed{\bar y(x)=\sum_y yP(y|x)}`{=tex} \]

For continuous labels,

\[ `\boxed{\bar y(x)=\int yP(y|x)\,dy}`{=tex} \]

The integral is simply the continuous version of the sum.

------------------------------------------------------------------------

# Step 7. What does this mean intuitively?

Imagine asking:

> "If I could repeatedly observe identical houses with these exact
> features, what average selling price would I get?"

That average is

\[ `\bar `{=tex}y(x) \]

It is the "true underlying value" for those features.

------------------------------------------------------------------------

# Step 8. Why is this useful?

Suppose your model predicts

\[ h(x)=418,000 \]

The true expected label is

\[ `\bar `{=tex}y(x)=415,000 \]

Sometimes the observed price is 430k.

Sometimes it is 402k.

Sometimes it is 417k.

Those differences are **noise.**

Your model should be compared to

\[ `\bar `{=tex}y(x) \]

because that is what is actually learnable.

------------------------------------------------------------------------

# Python Example 1 (Discrete)

``` python
import numpy as np

prices = np.array([400000, 410000, 420000, 430000])
probabilities = np.array([0.2, 0.3, 0.3, 0.2])

expected_price = np.sum(prices * probabilities)

print(expected_price)
```

Output:

``` text
415000.0
```

The multiplication computes each value times its probability, and
`sum()` adds them together.

------------------------------------------------------------------------

# Python Example 2 (Simulation)

``` python
import numpy as np

prices = np.array([400000, 410000, 420000, 430000])
probabilities = np.array([0.2, 0.3, 0.3, 0.2])

samples = np.random.choice(
    prices,
    size=100000,
    p=probabilities
)

print(samples.mean())
```

Typical output:

``` text
414972.3
```

As the number of samples grows, the average converges to **415000**.

------------------------------------------------------------------------

# Python Example 3 (Continuous)

Suppose

\[ y\|x `\sim `{=tex}N(415000,;10000\^2) \]

``` python
import numpy as np

samples = np.random.normal(
    loc=415000,
    scale=10000,
    size=100000
)

print(samples.mean())
```

Typical output:

``` text
414989.7
```

Again, the sample mean approaches the true expected label.

------------------------------------------------------------------------

# The Big Picture

There are **three different "labels"** to keep straight:

1.  **Observed label** (y): the actual value in your dataset (includes
    randomness/noise).
2.  **Expected label** (`\bar `{=tex}y(x)=E\[y`\mid `{=tex}x\]): the
    average value you would get over infinitely many observations with
    the same features.
3.  **Model prediction** (h(x)): what your trained model predicts.

Bias measures how far (h(x)) is from the expected label. The remaining
differences between the observed labels and the expected label are
**irreducible noise**, which no machine learning model can eliminate.
