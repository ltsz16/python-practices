# Bias, Variance, and Noise

## A Detailed Guide to Generalization Error and the Bias--Variance Decomposition

> These notes expand upon the material shown in the screenshots and
> explain every major equation with intuition and Python examples.

------------------------------------------------------------------------

# Table of Contents

1.  Review: Expected Label
2.  Generalization Error
3.  Why a Learned Model is Random
4.  Expected Hypothesis
5.  Generalization Error of a Learning Algorithm
6.  Bias--Variance Decomposition
7.  Python Simulation
8.  Practical Interpretation
9.  Summary

------------------------------------------------------------------------

# 1. Review: Expected Label

For a feature vector (x), the label (y) is often **not deterministic**.

Instead there is a conditional probability distribution

\[ P(y\|x) \]

The expected label is

\[ `\bar `{=tex}y(x)=E\[y\|x\] \]

or

\[ `\bar `{=tex}y(x)=`\int `{=tex}yP(y\|x),dy \]

This is the average label over infinitely many observations having the
same features.

Think of it as the "true signal" hidden underneath random variation.

------------------------------------------------------------------------

# 2. Generalization Error

Suppose we collect a training dataset

\[ D={(x_1,y_1),...,(x_n,y_n)} \]

A learning algorithm

\[ A \]

takes the dataset and produces a model

\[ h_D=A(D) \]

Notice the subscript.

The model depends on the dataset.

If we trained on different data we would obtain a different model.

The generalization error is

\[ `\epsilon`{=tex}(h) = E\_{P(x,y)} \[(h(x)-y)\^2\] \]

This is simply the average squared prediction error over the true
population---not just the training data.

## Python

``` python
import numpy as np

y_true=np.array([4,6,8,10])
y_pred=np.array([5,5,9,8])

mse=np.mean((y_pred-y_true)**2)
print(mse)
```

The square penalizes large errors more heavily.

------------------------------------------------------------------------

# 3. Why is the Model Random?

Students often wonder:

> "My model is deterministic. Why do the notes call it random?"

The answer is that the **training data are random**.

Imagine drawing different training sets from the same population.

Dataset 1

    House A
    House B
    House C

produces

    Model #1

Dataset 2

    House D
    House E
    House F

produces

    Model #2

Therefore

\[ h_D \]

is itself a random variable because

\[ D \]

is random.

------------------------------------------------------------------------

# 4. Expected Hypothesis

Average all possible models trained on all possible datasets.

This produces the expected hypothesis

\[ `\bar `{=tex}h = E_D\[h_D\] = `\int `{=tex}h_DP\^n(D),dD \]

where

\[ P\^n(D) \]

is the probability of drawing dataset (D).

Think of this as

> "If I trained forever on infinitely many random datasets, what average
> prediction would I obtain?"

## Python

``` python
import numpy as np
from sklearn.linear_model import LinearRegression

predictions=[]

for seed in range(100):

    rng=np.random.default_rng(seed)

    x=rng.uniform(0,10,40).reshape(-1,1)
    y=3*x.flatten()+5+rng.normal(0,2,40)

    model=LinearRegression().fit(x,y)

    predictions.append(model.predict([[5]])[0])

print(np.mean(predictions))
```

Each dataset creates a slightly different model.

The average prediction approximates the expected hypothesis.

------------------------------------------------------------------------

# 5. Generalization Error of the Learning Algorithm

Previously we evaluated

\[ `\epsilon`{=tex}(h_D) \]

for one trained model.

Now we evaluate the **algorithm itself**.

Imagine repeating:

1.  Draw a dataset.
2.  Train.
3.  Test.
4.  Repeat forever.

The expected performance becomes

\[ `\epsilon`{=tex}(`\bar `{=tex}h) = E\_{x,y,D} \[(h_D(x)-y)\^2\] \]

The expectation now averages over

-   possible datasets
-   possible test points

because both are random.

------------------------------------------------------------------------

# 6. Bias--Variance Decomposition

The key identity is

\[ (h_D-y) = (h_D-`\bar `{=tex}h) + (`\bar `{=tex}h-`\bar `{=tex}y) +
(`\bar `{=tex}y-y) \]

This simply adds and subtracts useful quantities.

Squaring and taking expectations gives

\[ `\boxed{
Error
=
Variance
+
Bias^2
+
Noise
}`{=tex} \]

Specifically,

Variance

\[ E\[(h_D-`\bar `{=tex}h)\^2\] \]

Bias²

\[ E\[(`\bar `{=tex}h-`\bar `{=tex}y)\^2\] \]

Noise

\[ E\[(`\bar `{=tex}y-y)\^2\] \]

------------------------------------------------------------------------

# Understanding Each Term

## Variance

Measures how much the model changes if we retrain using different
datasets.

High variance means overfitting.

Example:

Decision trees grown without pruning.

------------------------------------------------------------------------

## Bias

Measures how far the average model is from the true function.

High bias means underfitting.

Example:

Trying to fit a straight line to quadratic data.

------------------------------------------------------------------------

## Noise

Noise is randomness already present in reality.

No algorithm can remove it.

Examples

-   measurement error
-   missing variables
-   random human behavior

------------------------------------------------------------------------

# Complete Simulation

``` python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng=np.random.default_rng(0)

x_test=np.linspace(-3,3,100).reshape(-1,1)

all_predictions=[]

for seed in range(200):

    rng=np.random.default_rng(seed)

    x=rng.uniform(-3,3,40).reshape(-1,1)

    y=np.sin(x[:,0])+rng.normal(0,.3,40)

    tree=DecisionTreeRegressor(max_depth=4)

    tree.fit(x,y)

    all_predictions.append(tree.predict(x_test))

pred=np.array(all_predictions)

expected_prediction=pred.mean(axis=0)

variance=((pred-expected_prediction)**2).mean(axis=0)

print(variance.mean())
```

This estimates the variance term by repeatedly training on new datasets.

------------------------------------------------------------------------

# Practical Interpretation

  Model                  Bias       Variance
  ---------------------- ---------- ------------------
  Linear Regression      High       Low
  Small Decision Tree    Medium     Medium
  Deep Decision Tree     Low        High
  Random Forest          Low        Lower than Tree
  Large Neural Network   Very Low   Potentially High

Regularization generally increases bias while decreasing variance.

Cross-validation helps choose the best tradeoff.

Ensemble methods reduce variance without substantially increasing bias.

------------------------------------------------------------------------

# Visual Intuition

    Observed Label

               y

               *
              * *
            *  *   *

    ----------------------------

    Expected Label

            ---------

    Model Predictions

         -----
          ------
           ------

The observed labels fluctuate because of noise.

The expected label is the underlying signal.

Different trained models fluctuate around the expected hypothesis.

------------------------------------------------------------------------

# Final Summary

There are four important quantities:

-   Observed label (y)
-   Expected label (`\bar `{=tex}y(x))
-   Learned model (h_D)
-   Expected hypothesis (`\bar `{=tex}h(x))

The total expected prediction error can be decomposed as

\[ `\boxed{
Error
=
Bias^2
+
Variance
+
Noise
}`{=tex} \]

where

-   **Bias** measures systematic error caused by overly simple
    assumptions.
-   **Variance** measures sensitivity to different training datasets.
-   **Noise** is irreducible randomness in the data.

Understanding this decomposition explains why larger models overfit, why
regularization works, why ensembles help, and why no model can ever
achieve zero error on noisy real-world data.
