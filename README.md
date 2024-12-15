# Spam Email Classification Using Decision Tree Classifier


This project builds a spam email classifier using a Decision Tree Classifier integrated with a HashingVectorizer and TfidfTransformer. The dataset used is the SpamAssassin Public Corpus, which includes separate folders for spam and ham (non-spam) emails.

Features
Preprocesses email content into numerical representations using HashingVectorizer and TfidfTransformer.
Uses a Decision Tree Classifier for training and prediction.
Evaluates model performance using accuracy and a confusion matrix.
Installation
Prerequisites
Python 3.7 or higher
Install the required libraries by running:
bash
Copy code
pip install -U scikit-learn
Dataset Setup
Download the SpamAssassin Public Corpus from SpamAssassin Corpus.
Extract the dataset and organize it into the following directory structure:
arduino
Copy code
spamassassin-public-corpus/
  spam/
    spam1.txt
    spam2.txt
    ...
  ham/
    ham1.txt
    ham2.txt
    ...
Project Steps
Step 1: Loading the Dataset
The script reads the email files from the spam and ham directories:

spam/ contains spam emails.
ham/ contains non-spam emails.
The function load_emails_from_folder iterates through each folder, reads email files, and returns a list of email contents.

Step 2: Assigning Labels
Spam emails are assigned the label 0.
Ham emails are assigned the label 1.
Both spam and ham emails are combined into a single list, email_contents, while their labels are stored in email_labels.

Step 3: Splitting Data
The data is split into training and test sets using an 80-20 split:

80% for training the model.
20% for testing the model's performance.
The train_test_split function ensures that both the training and test sets contain a mix of spam and ham emails.

Step 4: Text Vectorization
Email content, being text data, is converted into numerical representations using:

HashingVectorizer:
Converts the text into a sparse matrix of word frequencies using a hashing mechanism.
The number of features is set to 
2
12
=
4096
2 
12
 =4096 to balance memory efficiency and accuracy.
TfidfTransformer:
Transforms the word frequency matrix into Term Frequency-Inverse Document Frequency (TF-IDF) values.
These values represent how important a word is to an email compared to the entire dataset.
Step 5: Training the Model
The Decision Tree Classifier is trained using:

X_train_vec: The TF-IDF transformed training data.
y_train: The labels for the training data.
The fit method of the classifier trains the model to classify spam and ham emails.

Step 6: Evaluating the Model
After training, the model is tested on the test dataset:

Predictions:
The classifier predicts labels for X_test_vec.
Accuracy:
The accuracy_score function computes the percentage of correct predictions.
Confusion Matrix:
The confusion_matrix function provides a summary of:
True Positives (correctly identified spam emails).
True Negatives (correctly identified ham emails).
False Positives (ham emails misclassified as spam).
False Negatives (spam emails misclassified as ham).
How to Run the Script
Ensure the dataset directory paths (spam_path and ham_path) are correctly set in the script.
Run the script:
bash
Copy code
python spam_classifier.py
View the output:
Accuracy: Overall performance of the model.
Confusion Matrix: Detailed classification results.
Sample Output
text
Copy code
Model Accuracy: 0.92
Confusion Matrix:
 [[189  11]
  [ 14 186]]
Project Structure
arduino
Copy code
spam-classifier/
│
├── spamassassin-public-corpus/
│   ├── spam/
│   ├── ham/
│
├── spam_classifier.py
├── README.md
Dependencies
Python 3.7 or higher
Required libraries:
scikit-learn
os
Notes
Ensure the dataset contains a sufficient number of files in both spam and ham directories for meaningful results.
Modify the paths in the script if the dataset is located in a different directory.
Further Improvements
Use more advanced classifiers (e.g., Random Forest, Gradient Boosting).
Add hyperparameter tuning for the Decision Tree Classifier.
Explore deep learning techniques for text classification.
