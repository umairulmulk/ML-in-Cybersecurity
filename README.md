# Email Spam Classifier Using Decision Tree

This project demonstrates a simple implementation of an email spam classifier using the **DecisionTreeClassifier** from scikit-learn. The dataset is sourced from the "SpamAssassin Public Corpus," with email text being vectorized using **HashingVectorizer** and transformed using **TfidfTransformer**.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset Structure](#dataset-structure)
- [Code Explanation](#code-explanation)
  - [Step 1: Load Dataset](#step-1-load-dataset)
  - [Step 2: Assign Labels](#step-2-assign-labels)
  - [Step 3: Train-Test Split](#step-3-train-test-split)
  - [Step 4: Text Vectorization](#step-4-text-vectorization)
  - [Step 5: Model Training](#step-5-model-training)
  - [Step 6: Evaluation](#step-6-evaluation)
- [Final Output](#final-output)

## Prerequisites
Make sure you have the following Python libraries installed:

```bash
pip install scikit-learn
```

Additionally, you need the "SpamAssassin Public Corpus" dataset. Place the dataset in the following structure:

```
spamassassin-public-corpus/
spam/
spam1.txt
spam2.txt
ham/
ham1.txt
ham2.txt
```

## Dataset Structure
- `spam/`: Contains spam emails.
- `ham/`: Contains non-spam (ham) emails.

## Code Explanation

### Step 1: Load Dataset
The `load_emails_from_folder` function reads email text files from a given directory. Emails are loaded as plain text.

```python
def load_emails_from_folder(folder_path):
    emails = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='latin1') as file:
            emails.append(file.read())
    return emails
```

The spam and ham email directories are loaded as follows:
```python
spam_emails = load_emails_from_folder(spam_path)
ham_emails = load_emails_from_folder(ham_path)
```

### Step 2: Assign Labels
Spam emails are labeled as `0`, and ham emails are labeled as `1`. These labels are combined with the email content:
```python
email_contents = spam_emails + ham_emails
email_labels = [0] * len(spam_emails) + [1] * len(ham_emails)
```

### Step 3: Train-Test Split
The dataset is split into 80% training and 20% testing using `train_test_split`:
```python
X_train, X_test, y_train, y_test = train_test_split(
    email_contents, email_labels, test_size=0.2, random_state=42)
```

### Step 4: Text Vectorization
We use `HashingVectorizer` to convert text into sparse numerical feature vectors and `TfidfTransformer` to scale them based on term frequency-inverse document frequency (TF-IDF):
```python
vectorizer = HashingVectorizer(n_features=2**12)
tfidf_transformer = TfidfTransformer()

X_train_vec = tfidf_transformer.fit_transform(vectorizer.transform(X_train))
X_test_vec = tfidf_transformer.transform(vectorizer.transform(X_test))
```

### Step 5: Model Training
A Decision Tree classifier is trained on the transformed data:
```python
classifier = DecisionTreeClassifier()
classifier.fit(X_train_vec, y_train)
```

### Step 6: Evaluation
The trained model is evaluated using accuracy and a confusion matrix:
```python
y_pred = classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
```

## Final Output
The code outputs the following:
1. **Model Accuracy**: The percentage of correctly classified emails.
2. **Confusion Matrix**: A matrix showing the counts of true positives, false positives, true negatives, and false negatives.

Example output:
```
Model Accuracy: 0.85
Confusion Matrix:
 [[25  5]
  [ 3 27]]
```

- **Accuracy of our model: 0.97**
- **The Confusion Matrix**:
  - True Positives (Spam correctly identified): `298`
  - False Positives (Ham misclassified as Spam): `6`
  - True Negatives (Ham correctly identified): `13`
  - False Negatives (Spam misclassified as Ham): `523`

## How to Run
1. Place the dataset in the required structure.
2. Adjust the `spam_path` and `ham_path` variables to point to your dataset.
3. Run the script to train and evaluate the model.

## For a detailed explanation of the project and the steps involved, check out the full article on Medium: 
[Email Spam Classifier with Decision Tree](https://medium.com/@umairm142/introduction-ee2512a061b6)
