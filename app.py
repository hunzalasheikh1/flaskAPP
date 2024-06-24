from flask import Flask, request, render_template
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import ssl

# Bypass SSL certificate verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Initialize the Flask application
app = Flask(__name__)

# Load the preprocessed DataFrame
df = pd.read_csv('cleaned_pubmed_dataset.csv')

# Define a function to summarize text
def summarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Create a CountVectorizer to count word frequencies
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    word_count = vectors.sum(axis=0)
    words = vectorizer.get_feature_names_out()
    
    # Create a dictionary of word frequencies
    word_freq = {words[i]: word_count[i] for i in range(len(words))}
    
    # Score each sentence based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = 0
        for word in sentence.split():
            if word in word_freq:
                score += word_freq[word]
        sentence_scores[i] = score
    
    # Sort sentences by score and select the top N
    top_n = int(len(sentences) * 0.3)
    summarized_sentences = [sentences[i] for i in sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]]
    
    # Join the summarized sentences back into a single string
    summary = ' '.join(summarized_sentences)
    
    return summary

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling text input and summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        # Get the input text
        text = request.form['text']
        
        # Summarize the text
        summary = summarize_text(text)
        
        return render_template('result.html', original_text=text, summary=summary)

# Run the Flask application
if __name__ == '__main__':
    nltk.download('punkt')
    app.run(debug=True, port=5000)  
