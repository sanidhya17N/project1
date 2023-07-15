import os
import json
from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('language_detection_model.pkl')

# Mapping of labels to languages
label_mapping = {
    'english': 'English',
    'malayalam': 'Malayalam',
    'hindi': 'Hindi',
    'tamil': 'Tamil',
    'kannada': 'Kannada',
    'french': 'French',
    'spanish': 'Spanish',
    'portugeese': 'Portugeese',
    'italian': 'Italian',
    'russian': 'Russian',
    'sweedish': 'Sweedish',
    'dutch': 'Dutch',
    'arabic': 'Arabic',
    'turkish': 'Turkish',
    'german': 'German',
    'danish': 'Danish',
    'greek': 'Greek',
    # Add more mappings for other languages if necessary
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = json.loads(request.data)
    text = data['text']
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    label = label_mapping[prediction]
    return jsonify({'language': label})

if __name__ == '__main__':
    app.run()


