# STEP 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# STEP 2: Load the dataset
data = load_breast_cancer()

# Convert dataset to DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


# STEP 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# STEP 4: Create and train the model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)


# STEP 5: Make predictions
y_pred = model.predict(X_test)


# STEP 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# STEP 7: Visualize confusion matrix
plt.figure(figsize=(5, 4))
plt.imshow(confusion_matrix(y_test, y_pred), cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.show()

