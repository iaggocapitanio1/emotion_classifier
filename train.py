import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import settings

# Load data from the text file

DATA_PATH = settings.DATA_DIR  / 'data.txt'


data = np.loadtxt(DATA_PATH)
# Split data into features (X) and labels (y)
X = data[:, :-1] # Features are all columns except the last one
Y = data[:, -1] # Labels are the last column

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True ,stratify=Y)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(x_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))

with open('./model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

