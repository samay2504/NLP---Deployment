import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle


def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']

    # Extract Features with CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the data

    # Save the CountVectorizer to a file
    pickle.dump(cv, open('transform.pkl', 'wb'))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Evaluate the model
    score = clf.score(X_test, y_test)
    print(f"Model Accuracy: {score}")

    # Save the model to a file
    filename = 'nlp_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))

    return clf


# Train the model and save it
clf = predict()

# Alternatively, save using joblib (optional)
joblib.dump(clf, 'NB_spam_model.pkl')

# Load the model (example of how to load it)
NB_spam_model = open('NB_spam_model.pkl', 'rb')
clf = joblib.load(NB_spam_model)
