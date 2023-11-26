from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Load the model
model = joblib.load('navie_bayes_model.joblib')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.joblib')

# Load NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)
        return render_template('index.html', prediction=prediction[0], text=text)

if __name__ == '__main__':
    app.run(debug=True)
