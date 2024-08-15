from flask import Flask, render_template, request
import pickle

# Load the model and vectorizer from disk
model_path = 'D:/Projects/NLP Deploy/nlp_model.pkl'
vectorizer_path = 'D:/Projects/NLP Deploy/tranform.pkl'

with open(model_path, 'rb') as model_file:
    clf = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
