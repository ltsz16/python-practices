import csv
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#model = Perceptron()
# model = svm.SVC()
model = KNeighborsClassifier(n_neighbors=5)
# model = GaussianNB()

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# Separate data into training and testing groups
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# Train model on training set
X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
model.fit(X_training, y_training)

# Make predictions on the testing set
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)

# Compute how well we performed
correct = 0
incorrect = 0
total = 0
for actual, predicted in zip(y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")

not_reduced = True

# Reduce 4D data to 2D
pca = PCA(n_components=2)
if not_reduced:
    X_testing_2d = pca.fit_transform(X_testing)
else:
    # Visualize with first 2 features
    X_testing_2d = [row["evidence"][:2] for row in testing]

# Create scatter plot
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(X_testing_2d):
    color = 'blue' if y_testing[i] == "Authentic" else 'red'
    plt.scatter(x, y, color=color, s=50, alpha=0.6)

# Plot decision boundary
x_min = min(p[0] for p in X_testing_2d) - 0.5
x_max = max(p[0] for p in X_testing_2d) + 0.5
y_min = min(p[1] for p in X_testing_2d) - 0.5
y_max = max(p[1] for p in X_testing_2d) + 0.5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Create data for prediction (only use first 2 features, fill others with mean)
X_training_2d = [row["evidence"][:2] for row in training]
feature3_mean = np.mean([row["evidence"][2] for row in training])
feature4_mean = np.mean([row["evidence"][3] for row in training])

Z_pred = model.predict(np.c_[xx.ravel(), yy.ravel(), 
                        np.full(xx.ravel().shape, feature3_mean),
                        np.full(xx.ravel().shape, feature4_mean)])
# Convert predictions to numeric values (0 for Authentic, 1 for Counterfeit)
Z = np.array([0 if pred == "Authentic" else 1 for pred in Z_pred])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[0, 1, 2], colors=['lightblue', 'lightcoral'], alpha=0.5)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Banknote Classification - {type(model).__name__}')
plt.legend(['Authentic', 'Counterfeit'])
plt.tight_layout()
plt.savefig(f'banknote_classification_{type(model).__name__}.png', dpi=100)
print(f"Plot saved to banknote_classification_{type(model).__name__}.png")
