A good way to answer these questions is to use a simple decision process:

## Step 0: Exploratory Data Analysis (EDA) and Data Cleaning

Inspect data               -> EDA

Remove irrelevant features -> Feature Selection (e.g. remove columns)

Reduce dimensions          -> Dimensionality Reduction (usually create new dimensions from multiple ones)

Create better features     -> Feature Engineering

Remove correlated features -> Multicollinearity Analysis

- Features are data input (e.g. dimensions)
- Labels are data output

Bad records are handled with Data Cleaning. Strange records are investigated with Outlier/Anomaly Detection. Unnecessary columns are handled with Feature Selection. Too many columns are handled with Dimensionality Reduction.
**Ways to handle outliers**
Remember not all anomalies are mistakes (e.g. fraud, defects, high range salary)
* Remove (e.g. Temp = 3000F is clearly wrong)
* Cap/Winsorize (e.g. anything above 99th percentile becomes 99th percentile)
* Transform (e.g. income list of 1000, 2000, 5000 and 10000000.  Apply log to all to compress extreme values)
* Use tools less sensitive to outliers.  Huber Loss, Absolute Loss, Decision Trees, Random Forest
## Step 1: Is it Classification or Regression?
* Classification → Predicting categories (spam/not spam, yes/no, cat/dog)
* Regression → Predicting numbers (house price, temperature, sales)
## Step 2: How noisy is the data?
* Clean data → Aggressive losses are OK.
* Noisy/outlier-heavy data → Use more robust losses.
## Step 3: Pick your model

```
START: What are you trying to predict?
│
├─ Are you predicting a NUMBER? (price, revenue, temperature)
│  │
│  └─ YES → REGRESSION
│     │
│     ├─ Relationship appears LINEAR? ("increase in x → increase in y")
│     │  └─ YES → Linear Regression
│     │
│     ├─ Relationship appears LINEAR?
│     │  └─ NO
│     │     │
│     │     ├─ Need interpretability?
│     │     │  └─ YES → Decision Tree
│     │     │
│     │     ├─ Need interpretability?
│     │     │  └─ NO
│     │     │     │
│     │     │     ├─ Large dataset with complex patterns?
│     │     │     │  └─ YES → Random Forest
│     │     │     │
│     │     │     └─ Lots of local examples matter?
│     │     │        └─ YES → kNN Regression
│     │
│     └─ (No clear pattern?)
│        └─ Consider: Random Forest or Neural Network
│
└─ Are you predicting a CATEGORY? (spam/not spam, fraud/not fraud)
   │
   └─ YES → CLASSIFICATION
      │
      ├─ Need probabilities?
      │  └─ YES → Logistic Regression
      │
      ├─ Relationship approximately LINEAR?
      │  └─ YES → Logistic Regression
      │
      ├─ Need maximum margin?
      │  └─ YES → SVM
      │
      ├─ Need interpretability?
      │  └─ YES → Decision Tree
      │
      ├─ Need highest accuracy on tabular data?
      │  └─ YES → Random Forest
      │
      ├─ Small dataset?
      │  └─ YES → kNN
      │
      └─ Very complex boundaries?
         └─ YES → Random Forest or SVM
```

## Step 4: Do you care about feature selection or interpretability?
* Yes → Use L1 regularization.
* No → Use L2 regularization.
* Want both → Use Elastic Net (L1 + L2).

### Regression Problems
| Situation                                | Loss              | Regularizer           |
| ---------------------------------------- | ----------------- | --------------------- |
| Standard regression                      | MSE (Square Loss) | None or L2            |
| Regression with overfitting risk         | MSE               | L2 (Ridge)            |
| Need feature selection                   | MSE               | L1 (Lasso)            |
| Need feature selection + stable solution | MSE               | L1 + L2 (Elastic Net) |
| Noisy labels/outliers                    | Huber Loss        | L2                    |
| Many outliers                            | Absolute Loss     | L1 or L2              |

### Classification Problems
| Situation                              | Loss             | Regularizer  |
| -------------------------------------- | ---------------- | ------------ |
| Need probabilities                     | Logistic Loss    | L2           |
| Need probabilities + feature selection | Logistic Loss    | L1           |
| Standard SVM classifier                | Hinge Loss       | L2           |
| Sparse SVM                             | Hinge Loss       | L1           |
| Noisy classification data              | Logistic Loss    | L2           |
| Boosting / focus on hard examples      | Exponential Loss | Usually none |


## Keyword helps
| Keywords                    | Likely Choice          |
| --------------------------- | ---------------------- |
| "predict a number"          | Regression             |
| "predict a category"        | Classification         |
| "probabilities"             | Logistic Loss          |
| "maximum margin"            | Hinge Loss + L2        |
| "outliers" / "noisy labels" | Huber or Absolute Loss |
| "feature selection"         | L1                     |
| "sparse model"              | L1                     |
| "interpretability"          | L1                     |
| "stable solution"           | L2                     |
| "correlated features"       | Elastic Net            |
| "general-purpose/default"   | L2                     |
| "House Prices"              | Regression             |
| "Predict a Number"          | Regression             |
| "Spam Detection"            | Classification         |
| "Probability"               | Logistic Loss          |
| "Maximum Margin"            | Hinge Loss + L2        |
| "SVM"                       | Hinge Loss + L2        |
| "Interpretability"          | L1                     |
| "Sparse"                    | L1                     |
| "Feature Selection"         | L1                     |
| "Important Features"        | L1                     |
| "Gene Selection"            | L1 or Elastic Net      |
| "Correlated Features"       | Elastic Net            |
| "Unique Solution"           | L2 or Elastic Net      |
| "Noisy Labels"              | Logistic Loss / Huber  |
| "Outliers"                  | Huber or Absolute Loss |
| "General Purpose"           | L2                     |

## Decision Tree
START
  |
  |-- Are you predicting a NUMBER?
  |      (price, revenue, temperature, age)
  |
  |      YES --> REGRESSION
  |                |
  |                |-- Does the data have significant outliers
  |                |   or noisy labels?
  |                |
  |                |      YES
  |                |        |
  |                |        |-- Need smooth optimization?
  |                |        |       |
  |                |        |       YES --> Huber Loss
  |                |        |                  |
  |                |        |                  |-- Need feature selection?
  |                |        |                         |
  |                |        |                         YES --> Elastic Net
  |                |        |                         NO  --> L2
  |                |        |
  |                |        NO --> Absolute Loss
  |                |
  |                NO
  |                |
  |                |-- Need feature selection or
  |                |   explain important features?
  |                |
  |                |       YES --> MSE + L1
  |                |                  (Lasso)
  |                |
  |                |       Need sparse + stable?
  |                |       --> MSE + L1 + L2
  |                |           (Elastic Net)
  |                |
  |                NO --> MSE + L2
  |                           (Ridge)
  |
  |
  |-- Are you predicting a CATEGORY?
         (spam/not spam, fraud/not fraud)

         YES --> CLASSIFICATION
                    |
                    |-- Need probability outputs?
                    |   ("80% likely spam")
                    |
                    |      YES --> Logistic Loss
                    |                 |
                    |                 |-- Need feature selection?
                    |                        |
                    |                        YES --> L1
                    |                        NO  --> L2
                    |
                    |
                    |-- Need maximum margin?
                    |   (classic SVM)
                    |
                    |      YES --> Hinge Loss + L2
                    |
                    |
                    |-- Need sparse/interpretable classifier?
                    |
                    |      YES --> Hinge Loss + L1
                    |              OR
                    |              Logistic Loss + L1
                    |
                    |
                    |-- Lots of mislabeled or noisy examples?
                    |
                    |      YES --> Logistic Loss + L2
                    |
                    |
                    |-- Want to aggressively focus on
                    |   misclassified examples?
                    |
                    |      YES --> Exponential Loss
                    |               (AdaBoost)
                    |
                    |      WARNING:
                    |      Avoid if many outliers/noise.

## Cheat Sheet
**Regression**
Default Regression
    -> MSE + L2
    -> Ridge Regression

Need Feature Selection
    -> MSE + L1
    -> Lasso

Need Feature Selection + Stability
    -> MSE + L1 + L2
    -> Elastic Net

Noisy Data / Outliers
    -> Huber Loss + L2

Extreme Outliers
    -> Absolute Loss + L2

**Classification**
Need Probabilities
    -> Logistic Loss + L2

Need Probabilities + Feature Selection
    -> Logistic Loss + L1

Need Maximum Margin
    -> Hinge Loss + L2
    -> SVM

Need Sparse Classifier
    -> Hinge Loss + L1
       or
       Logistic Loss + L1

Need Robustness to Noise
    -> Logistic Loss + L2

Want Aggressive Learning
    -> Exponential Loss
       (AdaBoost)

**If stuck**
Regression?
    -> Use MSE + L2

Classification?
    -> Use Logistic Loss + L2

Need feature selection?
    -> Change L2 to L1

Need feature selection + stability?
    -> Elastic Net

Need maximum margin?
    -> Hinge Loss + L2

Need robustness to outliers?
    -> Huber Loss
---
**Examples**

Scenario 1

Predict house prices from square footage, bedrooms, and location.

Regression
No mention of outliers
Accuracy is main goal

Answer:

Loss: MSE (Square Loss)
Regularizer: L2
Model: Ridge Regression

Why?

MSE is easy to optimize.
L2 helps prevent overfitting.
Scenario 2

Predict house prices, but some houses have strange sale prices because of family transfers or tax reasons.

Answer:

Loss: Huber Loss (or Absolute Loss)
Regularizer: L2

Why?

Outliers would heavily affect MSE.
Huber is more robust.
Scenario 3

Use 20,000 gene measurements to predict whether someone has a disease. Researchers want to know which genes matter.

Answer:

Loss: Logistic Loss
Regularizer: L1

Why?

Classification problem.
L1 gives sparse weights.
Non-zero coefficients identify important genes.
Scenario 4

Same gene problem, but many genes are highly correlated and researchers want a unique, stable solution.

Answer:

Loss: Logistic Loss
Regularizer: Elastic Net (L1 + L2)

Why?

L1 performs feature selection.
L2 stabilizes correlated features.
Scenario 5

Build a spam detector and output a probability that an email is spam.

Answer:

Loss: Logistic Loss
Regularizer: L2

Why?

Logistic regression produces probabilities.
L2 is the standard choice.
Scenario 6

Build a text classifier with 100,000 words as features. You want a small set of important words.

Answer:

Loss: Logistic Loss or Hinge Loss
Regularizer: L1

Why?

L1 drives most word weights to zero.
Easier to interpret.
Scenario 7

Build a classifier with the highest-margin decision boundary.

Answer:

Loss: Hinge Loss
Regularizer: L2

Why?

This is the classic SVM formulation.
L2 regularization gives maximum-margin behavior.