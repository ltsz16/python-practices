# How to Pick a Machine Learning Algorithm

Choosing an algorithm is often easier than students think.

The trick is to identify:

1. Classification or Regression?
2. Interpretability or Accuracy?
3. Small or Large Dataset?
4. Linear or Nonlinear Relationships?
5. Need Probabilities?
6. Need Feature Selection?

---

# Master Decision Tree

```text
START
  |
  |-- Predicting a NUMBER?
  |
  |      YES
  |       |
  |       v
  |   REGRESSION
  |       |
  |       |-- Relationship appears linear?
  |       |
  |       |      YES
  |       |       |
  |       |       v
  |       |   Linear Regression
  |       |
  |       |      NO
  |       |
  |       |-- Need interpretability?
  |       |
  |       |      YES
  |       |       |
  |       |       v
  |       |   Decision Tree
  |       |
  |       |      NO
  |       |
  |       |-- Need strong predictive accuracy?
  |               |
  |               v
  |          Random Forest
  |
  |
  |-- Predicting a CATEGORY?
          |
          v
     CLASSIFICATION
          |
          |-- Need probabilities?
          |
          |      YES
          |       |
          |       v
          |   Logistic Regression
          |
          |-- Need maximum margin?
          |
          |      YES
          |       |
          |       v
          |      SVM
          |
          |-- Need interpretability?
          |
          |      YES
          |       |
          |       v
          |   Decision Tree
          |
          |-- Highest accuracy on tabular data?
          |
                  |
                  v
             Random Forest
```
viz
```text
                 ALGORITHM CHEAT SHEET

Predict a Number
      |
      +--> Linear Relationship
      |         |
      |         +--> Linear Regression
      |
      +--> Nonlinear Relationship
                |
                +--> Explainability
                |         |
                |         +--> Decision Tree
                |
                +--> Accuracy
                          |
                          +--> Random Forest


Predict a Category
      |
      +--> Need Probabilities
      |         |
      |         +--> Logistic Regression
      |
      +--> Maximum Margin
      |         |
      |         +--> SVM
      |
      +--> Explainability
      |         |
      |         +--> Decision Tree
      |
      +--> Best Prediction Accuracy
                |
                +--> Random Forest
```

---

# Algorithm Cheat Sheet
```text
                           STUCK?

                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v

      Regression                           Classification

          |                                       |
          v                                       v

Linear Regression                     Need Probabilities?
                                              |
                      +-----------------------+----------------------+
                      |                                              |
                    YES                                             NO
                      |                                              |
                      v                                              v

           Logistic Regression                         Need Maximum Margin?
                                                               |
                                       +-----------------------+------------------+
                                       |                                          |
                                     YES                                         NO
                                       |                                          |
                                       v                                          v

                                      SVM                                Random Forest


If question says:

"Interpretability"      -> Decision Tree

"Highest Accuracy"      -> Random Forest

"Similar Neighbors"     -> kNN

"Text Classification"   -> Naive Bayes or SVM

"Images"                -> Neural Network

"Large Complex Data"    -> Neural Network
```

## Linear Regression

### Problem Type

```text
Regression
```

### Use When

```text
Relationship is approximately linear
Need interpretability
Fast baseline model
```

### Example

```text
Predict:
- House prices
- Revenue
- Demand
```

### Pros

✅ Fast

✅ Interpretable

✅ Simple

### Cons

❌ Cannot model complex nonlinear patterns

---

# Logistic Regression

### Problem Type

```text
Classification
```

### Use When

```text
Need probabilities
Need interpretability
Relationship is roughly linear
```

### Example

```text
Spam Detection

P(Spam) = 0.92
```

### Pros

✅ Produces probabilities

✅ Fast

✅ Easy to explain

### Cons

❌ Linear decision boundary

---

# k-Nearest Neighbors (kNN)

### Problem Type

```text
Classification or Regression
```

### Core Idea

```text
Find similar examples
and vote
```

### Use When

```text
Small dataset

Similar examples should
have similar labels
```

### Example

```text
Recommend products
Movie recommendations
Simple pattern matching
```

### Pros

✅ Easy to implement

✅ No training required

✅ Can model nonlinear boundaries

### Cons

❌ Slow predictions

❌ Doesn't scale well

❌ Sensitive to irrelevant features

---

# Decision Tree

### Problem Type

```text
Classification or Regression
```

### Core Idea

```text
Series of if/then rules
```

### Example

```text
Income > 100K?

      YES
       |
      Buy

      NO
       |
      Age > 40?
```

### Use When

```text
Interpretability is important
Business rules are important
Need explainable decisions
```

### Pros

✅ Visual

✅ Human-readable

✅ Handles nonlinear relationships

### Cons

❌ Easily overfits

❌ Can be unstable

---

# Random Forest

### Problem Type

```text
Classification or Regression
```

### Core Idea

```text
Train many trees
Combine predictions
```

### Example

```text
Tree 1 -> Yes
Tree 2 -> Yes
Tree 3 -> No

Majority Vote -> Yes
```

### Use When

```text
Need high accuracy
Tabular data
Complex nonlinear relationships
```

### Pros

✅ Usually very accurate

✅ Robust

✅ Less overfitting than trees

✅ Handles nonlinear patterns

### Cons

❌ Less interpretable

❌ Larger model

---

# Support Vector Machine (SVM)

### Problem Type

```text
Classification
```

(also supports regression through SVR)

### Core Idea

```text
Maximize margin
between classes
```

### Use When

```text
Medium-sized datasets

High-dimensional data

Text classification

Maximum-margin solution
```

### Pros

✅ Strong performance

✅ Works well in high dimensions

✅ Powerful with kernels

### Cons

❌ Harder to explain

❌ Can be expensive on huge datasets

---

# Kernel SVM

### Use When

```text
Data is nonlinear

Need complex decision boundaries

Dataset isn't enormous
```

### Common Kernels

```text
Linear
Polynomial
RBF
```

### Pros

✅ Powerful

✅ Flexible

✅ Learns nonlinear patterns

### Cons

❌ Slower

❌ More tuning required

---

# Naive Bayes

### Problem Type

```text
Classification
```

### Use When

```text
Text classification

Document classification

Email filtering
```

### Example

```text
Spam Detection
Sentiment Analysis
```

### Pros

✅ Extremely fast

✅ Works surprisingly well

✅ Small datasets

### Cons

❌ Strong independence assumptions

---

# Neural Networks

### Problem Type

```text
Classification
Regression
```

### Use When

```text
Large datasets

Complex patterns

Images

Audio

Natural language
```

### Pros

✅ Extremely powerful

✅ State-of-the-art on many problems

### Cons

❌ Requires lots of data

❌ Less interpretable

❌ More computationally expensive

---

# Random Forest vs Decision Tree

## Question

Need explanations?

Choose:

```text
Decision Tree
```

---

## Question

Need the best accuracy?

Choose:

```text
Random Forest
```

---

# Logistic Regression vs SVM

## Question

Need probabilities?

Choose:

```text
Logistic Regression
```

---

## Question

Need maximum margin?

Choose:

```text
SVM
```

---

# kNN vs Random Forest

## Question

Small dataset and local similarity?

Choose:

```text
kNN
```

---

## Question

Large tabular dataset and best performance?

Choose:

```text
Random Forest
```

---

# Exam Keyword Cheat Sheet

```text
Probability
    -> Logistic Regression

Maximum Margin
    -> SVM

Linear Relationship
    -> Linear Regression

Explainability
    -> Decision Tree

If/Then Rules
    -> Decision Tree

Similar Neighbors
    -> kNN

Text Classification
    -> Naive Bayes or SVM

High-Dimensional Data
    -> SVM

Feature Importance
    -> Random Forest

Best Tabular Performance
    -> Random Forest

Image Recognition
    -> Neural Network

Large Complex Dataset
    -> Neural Network

Reduce Tree Overfitting
    -> Random Forest
```

---

# Quick Comparison

```text
                Accuracy  Interp.  Speed   Nonlinear

Linear Reg.       ★★       ★★★★★    ★★★★★      ★

Logistic Reg.     ★★★      ★★★★★    ★★★★★      ★

kNN               ★★★      ★★       ★★         ★★★★

Decision Tree     ★★★      ★★★★★    ★★★★       ★★★★

Random Forest     ★★★★★    ★★       ★★★        ★★★★★

SVM               ★★★★     ★★       ★★★        ★★★★★

Naive Bayes       ★★★      ★★★      ★★★★★      ★

Neural Networks   ★★★★★    ★         ★         ★★★★★
```

---

# 10-Second Rule

```text
Predict a Number?
    -> Linear Regression

Predict a Category + Probability?
    -> Logistic Regression

Need Explainability?
    -> Decision Tree

Need Accuracy on Tabular Data?
    -> Random Forest

Need Maximum Margin?
    -> SVM

Need Similarity-Based Learning?
    -> kNN

Need Text Classification?
    -> Naive Bayes or SVM

Need Deep Learning / Images?
    -> Neural Network
```

---

# One-Page Mental Model

```text
Linear Regression
    -> Numbers

Logistic Regression
    -> Probabilities

kNN
    -> Similar neighbors

Decision Tree
    -> Explainable rules

Random Forest
    -> Accurate trees

SVM
    -> Maximum margin

Naive Bayes
    -> Text and simple classification

Neural Networks
    -> Large complex datasets
```

## One-Line Summary

```text
Use Linear/Logistic Regression for simple interpretable models, Decision Trees for explanations, Random Forests for tabular accuracy, SVMs for maximum-margin classification, kNN for similarity-based learning, Naive Bayes for text, and Neural Networks for large complex datasets.
```