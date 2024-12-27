# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('emails.csv')
print(df.head())
print(df.columns)
print(df.isnull().sum().sum())  # Total missing values in the dataset
print(df.describe())  # Basic statistics of numerical columns
# Preprocessing: Remove unnecessary columns and prepare the dataset for feature selection
# Assuming 'Prediction' is the target column and the rest are features

# Separate features and target variable
X = df.drop(columns=['Email No.', 'Prediction'], errors='ignore')  # Drop unnecessary columns
y = df['Prediction']

# Display the shape of the features and target
print(X.shape)
print(y.shape)
# Combine all text features into a single column (excluding Email No. and Prediction)
text_features = df.drop(['Email No.', 'Prediction'], axis=1, errors='ignore')
X = text_features.astype(str).agg(' '.join, axis=1)
y = df['Prediction']

print("Dataset shape:", df.shape)
print("\
Sample of combined text:")
print(X.head())
print("\
Target variable distribution:")
print(y.value_counts()) # Step 2: Pre-processing and TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000, stop_words='english') # Step 4: Apply Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Step 5: Generate Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\
Classification Report:")
print(class_report)X_tfidf = tfidf.fit_transform(X) # Import necessary libraries
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionary to store results
results = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    results['Accuracy'].append(accuracy_score(y_test, y_pred))
    results['Precision'].append(precision_score(y_test, y_pred, average='weighted'))
    results['Recall'].append(recall_score(y_test, y_pred, average='weighted'))
    results['F1-Score'].append(f1_score(y_test, y_pred, average='weighted'))

# Create comparison DataFrame
comparison_df = pd.DataFrame(results, index=classifiers.keys())
print("Algorithm Comparison:")
print(comparison_df)

# Create visualization
plt.figure(figsize=(12, 6))
comparison_df.plot(kind='bar', width=0.8)
plt.title('Algorithm Performance Comparison')
plt.xlabel('Algorithms')
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Create confusion matrices for each classifier
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices Comparison')

for i, (name, clf) in enumerate(classifiers.items()):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show() # Import additional libraries
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Initialize classifiers including J48 (Decision Tree)
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Dictionary to store results
results = {
    'Accuracy': [],
    'Training Time': [],
    'Testing Time': [],
    'Error Rate': []
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    # Training time
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    # Testing time
    test_start = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - test_start
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    
   # Store results
    results['Accuracy'].append(accuracy)
    results['Training Time'].append(train_time)
    results['Testing Time'].append(test_time)
    results['Error Rate'].append(error_rate)



# Create time comparison plot
plt.figure(figsize=(12, 6))
comparison_df[['Training Time', 'Testing Time']].plot(kind='bar', width=0.8)
plt.title('Algorithm Time Comparison')
plt.xlabel('Algorithms')
plt.ylabel('Time (seconds)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
