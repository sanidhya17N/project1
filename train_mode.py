import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('Language-Detection.csv')

# Preprocess the dataset
df['Language'] = df['Language'].str.lower()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Language'], test_size=0.2, random_state=42)

# Train the language detection model
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer
joblib.dump(classifier, 'language_detection_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the test dataset
X_test_tfidf = vectorizer.transform(X_test)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Get the unique language labels
labels = np.unique(y_test)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)